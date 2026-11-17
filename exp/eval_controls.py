"""Linear-probe evaluation of the camera-ready control runs (output/controls).

Script version of exp/benchmark_multipl_cross.ipynb, i.e. the notebook that produced
Tables 2 and 3 (pixel runs of output/pixels, none|slurpp|ppg). The original
notebook is NOT modified. Same protocol, with no change:
  * backbones DINOv2 ViT-B/14 (252 px), CLIP ViT-B/32 (224), MoCoV3 ViT-B (224);
  * 5 probe seeds (0..4), Adam lr 1e-3, 5000 steps, batch 256, last checkpoint;
  * grid matched in {False, True} (contrast matching on real-image moments)
    x std_mode in {none, scalar}; full-data reference with std_mode="none";
  * same environment variables (HOME/TORCH_HOME/HF_HOME under $WORK) so the same
    cached weights are used.
Helpers (cells 3-11) are copied verbatim; the main loop is cell 13, with paths
and sources passed as arguments. Only differences: tqdm.auto instead of
tqdm.notebook (display only), argparse, an optional backbone pre-check.

Output: raw CSV with exactly the columns of eval_raw.csv
  arch, IPC, Variant, dseed, pseed, matched, std_mode, img_mean, img_std,
  F1 macro (%), F1 weighted (%)
The paper slice (scalar std, no matching) is matched == False & std_mode == "scalar".

Usage:
  python exp/eval_controls.py --out eval_raw_controls.csv
  python exp/eval_controls.py \
      --source $WORK/projects/GradientDistillation/output/controls/results/aqua20/dinov2_vitb \
               '^dinov2_vitb_aqua20_(?:none_frozen|affine|slurpp_t1|none_t3)_ipc\\d+_s\\d+$' \
      --source $WORK/projects/GradientDistillation/output/pixels/results/aqua20/dinov2_vitb \
               '^dinov2_vitb_aqua20_(?:none|slurpp|ppg)_ipc1_s\\d+$'
"""
import argparse
import os

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--work", default=os.environ.get("WORK", "/lustre/fswork/projects/rech/rbw/ucw75ke"),
                 help="Jean Zay $WORK (default: $WORK, else the notebook's hard-coded value)")
_ap.add_argument("--project-root", default=None,
                 help="default: <work>/projects/GradientDistillation")
_ap.add_argument("--data-root", default=None,
                 help="ImageFolder root with train/ and test/ (default: <work>/datasets/aqua20/data/aqua20)")
_ap.add_argument("--source", nargs=2, action="append", metavar=("DIR", "REGEX"),
                 help="run directory + regex on run folder names; repeatable. "
                      "Default: <project-root>/output/controls/results/aqua20/dinov2_vitb with the 4 control variants")
_ap.add_argument("--out", default="eval_raw_controls.csv", help="raw CSV path")
_ap.add_argument("--archs", nargs="+", default=["dinov2_vitb", "clip_vitb", "mocov3_vitb"])
_ap.add_argument("--std-modes", nargs="+", default=["none", "scalar"], choices=["none", "scalar", "perdim"])
_ap.add_argument("--matched", nargs="+", default=["False", "True"], choices=["False", "True"])
_ap.add_argument("--match-mode", default="real", choices=["real", "fixed"])
_ap.add_argument("--no-full", action="store_true",
                 help="skip the full-data reference probes (rows Variant=='full')")
_ap.add_argument("--no-precheck", action="store_true",
                 help="do not load every backbone once before the (long) main loop")
ARGS = _ap.parse_args()

# ---- environment: identical to the notebook (cell 1) ----
os.environ["XFORMERS_DISABLED"] = "1"

import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

WORK = ARGS.work
PROJECT_ROOT = ARGS.project_root or f"{WORK}/projects/GradientDistillation"

os.environ["HOME"] = WORK                       # au cas ou d'autres libs en dependent
os.environ["TORCH_HOME"] = f"{WORK}/.cache/torch"
os.environ["HF_HOME"] = f"{WORK}/.cache/huggingface"
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

for _p in (f"{PROJECT_ROOT}/src", PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- constants: identical to the notebook (cell 2), DATA_ROOT parameterizable ----
DATA_ROOT = ARGS.data_root or f"{WORK}/datasets/aqua20/data/aqua20"
NUM_CLASSES = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESOLUTION = 252

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

# stats utilisees A LA DISTILLATION en espace pixel (DINOv2 = ImageNet)
_SRC_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_SRC_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

ARCHS = {
    "dinov2_vitb": dict(res=RESOLUTION, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "clip_vitb":   dict(res=224,        mean=CLIP_MEAN,     std=CLIP_STD),
    "mocov3_vitb": dict(res=224,        mean=IMAGENET_MEAN, std=IMAGENET_STD),
}


# ---- verbatim: exp/benchmark_multipl_cross.ipynb, cell 3 ----
@dataclass
class Backbone:
    name: str
    model: nn.Module
    forward: Callable[[torch.Tensor], torch.Tensor]
    mean: tuple
    std: tuple
    res: int
    feat_dim: int


# ---- verbatim: exp/benchmark_multipl_cross.ipynb, cell 4 ----
def load_backbone(name: str) -> Backbone:
    cfg = ARCHS[name]
    if name == "dinov2_vitb":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        fwd = lambda x: m(x)                                    # CLS, 768
    elif name == "clip_vitb":
        import clip
        m = clip.load("ViT-B/32")[0].visual.float()
        fwd = lambda x: m(x)                                    # 512
    elif name == "mocov3_vitb":
        from src.models.moco_vision_tansformer import VisionTransformerMoCoV3
        m = VisionTransformerMoCoV3.from_pretrained("nyu-visionx/moco-v3-vit-b", num_classes=0)
        fwd = lambda x: m(x)                                    # 768
    else:
        raise ValueError(name)
 
    m.eval().requires_grad_(False).to(DEVICE)
    with torch.no_grad():
        feat_dim = fwd(torch.zeros(1, 3, cfg["res"], cfg["res"], device=DEVICE)).shape[-1]
    return Backbone(name, m, fwd, cfg["mean"], cfg["std"], cfg["res"], feat_dim)
 
 
def make_transform(bb: Backbone):
    return transforms.Compose([
        transforms.Resize(bb.res),
        transforms.CenterCrop(bb.res),
        transforms.ToTensor(),
        transforms.Normalize(bb.mean, bb.std),
    ])


# ---- verbatim: exp/benchmark_multipl_cross.ipynb, cell 5 ----
@torch.no_grad()
def extract_features(loader, bb: Backbone, desc="Extracting features"):
    feats, labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        feats.append(bb.forward(x.to(DEVICE)).float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


# ---- verbatim: exp/benchmark_multipl_cross.ipynb, cell 6 ----
def detect_image_space(x: torch.Tensor, tol: float = 0.05) -> str:
    """'unit' si deja dans [0,1] (runs latents), 'imagenet' sinon (runs pixel)."""
    lo, hi = x.min().item(), x.max().item()
    if lo >= -tol and hi <= 1.0 + tol:
        return "unit"
    if lo < -0.5 and hi > 1.5:
        return "imagenet"
    raise ValueError(
        f"espace indetermine : min={lo:.3f} max={hi:.3f} std={x.std().item():.3f}. "
        "Verifier ce que get_to_save ecrit dans data.pth."
    )
 
 
def syn_to_pixel(x: torch.Tensor, space: str) -> torch.Tensor:
    """-> [0,1], quel que soit l'espace de stockage."""
    if space == "unit":
        return x.clamp(0.0, 1.0)
    return (x * _SRC_STD + _SRC_MEAN).clamp(0.0, 1.0)
 
 
@torch.no_grad()
def syn_features(imgs: torch.Tensor, bb: Backbone, space: str, bs: int = 128):
    mean = torch.tensor(bb.mean, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(bb.std, device=DEVICE).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(imgs), bs):
        x = syn_to_pixel(imgs[i:i + bs], space).to(DEVICE)
        if x.shape[-1] != bb.res:
            x = F.interpolate(x, size=bb.res, mode="bicubic", align_corners=False).clamp(0, 1)
        out.append(bb.forward((x - mean) / std).float().cpu())
    return torch.cat(out)


# ---- verbatim: exp/benchmark_multipl_cross.ipynb, cell 7 ----
def standardize(train_feats, test_feats, mode="scalar"):
    """Centrage par dimension + mise a l'echelle.
 
    mode='scalar' : une seule echelle globale. Corrige la disparite d'echelle
        inter-backbones (MoCoV3 a des features de norme bien plus faible) sans
        estimer d ecarts-types sur n << d, ce qui amplifierait du bruit a IPC 1.
    mode='perdim' : standardisation classique. A n'utiliser qu'en ablation de
        robustesse : le biais d'estimation depend de l'IPC.
    Stats calculees sur le train uniquement (= le dataset distille).
    """
    mu = train_feats.mean(0, keepdim=True)
    tr, te = train_feats - mu, test_feats - mu
    if mode == "none":
        return train_feats, test_feats
    if mode == "scalar":
        s = tr.std().clamp_min(1e-6)
    elif mode == "perdim":
        s = tr.std(0, keepdim=True).clamp_min(1e-6)
    else:
        raise ValueError(f"mode inconnu : {mode!r}")
    return tr / s, te / s
 
 
def _infinite(loader):
    """Flux infini de batches AVEC re-melange a chaque passage.
 
    itertools.cycle memoriserait les batches du premier passage et les
    rejouerait a l'identique, annulant le shuffle du DataLoader.
    """
    while True:
        yield from loader


# ---- verbatim: exp/benchmark_multipl_cross.ipynb, cell 8 ----

def train_linear_probe(train_feats, train_labels, test_feats, test_labels,
                       feat_dim=768, max_steps=2000, lr=1e-3,
                       batch_size=256, eval_every=500, seed=0,
                       standardize_feats="scalar", verbose=False):
    """Budget en PAS d'optimisation, donc comparable entre IPC 1 et full.
 
    Aucune selection sur le test : c'est le dernier checkpoint qui est retenu.
    """
    if standardize_feats != "none":
        train_feats, test_feats = standardize(train_feats, test_feats, mode=standardize_feats)
 
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    head = nn.Linear(feat_dim, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
 
    g = torch.Generator()
    g.manual_seed(seed)
    bs = min(batch_size, len(train_labels))
    loader = DataLoader(TensorDataset(train_feats, train_labels),
                        batch_size=bs, shuffle=True, generator=g, drop_last=False)
    test_loader = DataLoader(TensorDataset(test_feats, test_labels),
                             batch_size=512, shuffle=False)
 
    @torch.no_grad()
    def evaluate():
        head.eval()
        preds, ys = [], []
        for feats, y in test_loader:
            preds.append(head(feats.to(DEVICE)).argmax(1).cpu())
            ys.append(y)
        preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
        head.train()
        return (f1_score(ys, preds, average="macro"),
                f1_score(ys, preds, average="weighted"))
 
    history = {}
    stream = _infinite(loader)
    run_loss, run_correct, run_n = 0.0, 0, 0
 
    for step in tqdm(range(1, max_steps + 1), desc="Probe", leave=False):
        feats, y = next(stream)
        feats, y = feats.to(DEVICE), y.to(DEVICE)
        logits = head(feats)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        run_loss += loss.item() * len(y)
        run_correct += (logits.argmax(1) == y).sum().item()
        run_n += len(y)
 
        if step % eval_every == 0 or step == max_steps:
            f1m, f1w = evaluate()
            history[step] = {"loss": run_loss / run_n,
                             "train_acc": run_correct / run_n,
                             "f1_macro": f1m, "f1_weighted": f1w}
            if verbose:
                h = history[step]
                tqdm.write(f"step {step:>5}/{max_steps} | loss {h['loss']:.4f} | "
                           f"train acc {h['train_acc']*100:.1f}% | "
                           f"F1 macro {h['f1_macro']*100:.1f}% | "
                           f"F1 weighted {h['f1_weighted']*100:.1f}%")
            run_loss, run_correct, run_n = 0.0, 0, 0
 
    # garde-fou : le protocole suppose que le probe a converge
    ckpts = sorted(history)
    if len(ckpts) >= 2:
        drift = abs(history[ckpts[-1]]["f1_macro"] - history[ckpts[-2]]["f1_macro"]) * 100
        if drift > 0.5:
            print(f"[warn] probe non converge : dF1={drift:.2f} pt entre step "
                  f"{ckpts[-2]} et {ckpts[-1]} (seed={seed}) -> augmenter max_steps")
 
    return head, history
 
 
def final_metrics(history: dict) -> dict:
    """Dernier checkpoint. Jamais le max sur le test."""
    last = max(history)
    return {k: float(history[last][k]) for k in ("f1_macro", "f1_weighted", "train_acc")}


# ---- verbatim: exp/benchmark_multipl_cross.ipynb, cell 9 ----
_SEED_RE = re.compile(r"_s(\d+)")
_IPC_RE = re.compile(r"_ipc(\d+)")
 
 
def parse_run_name(name: str) -> tuple[int | None, str, int | None]:
    """(ipc, variant, seed) depuis un nom de dossier de run distille.
 
    Tolere : sentinelle 'full_data', slash final, seed dupliquee en fin
    (..._s3407_s3407), descripteur de variante arbitraire, IPC absent.
    """
    name = name.rstrip("/")
    if name == "full_data":
        return None, "full", None
 
    seeds = _SEED_RE.findall(name)
    if not seeds:
        raise ValueError(f"run_name sans seed reconnaissable : {name!r}")
    if len(set(seeds)) > 1:
        raise ValueError(f"seeds divergentes dans {name!r} : {seeds}")
 
    m = _IPC_RE.search(name)
    ipc = int(m.group(1)) if m else None
 
    variant = name[: name.index(f"_s{seeds[0]}")] or "baseline"
    variant = _IPC_RE.sub("", variant)          # sinon 'ipc1' et 'ipc3' = 2 variantes
    return ipc, variant, int(seeds[0])


# ---- verbatim: exp/benchmark_multipl_cross.ipynb, cell 10 ----
class DistilledRun:
    def __init__(self, base_dir: str | Path, run_name: str):
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.data_path = self.base_dir / run_name / "data.pth"
        if not self.data_path.exists():
            raise FileNotFoundError(f"data.pth absent pour {run_name!r} (run incomplet ?)")

        syn = torch.load(self.data_path, map_location="cpu", mmap=True)
        im = syn["images"]
        self.space = detect_image_space(im)
        self.ipc, self.variant, self.dseed = parse_run_name(run_name)
        print(f"[{run_name}] space={self.space} ipc={self.ipc} "
              f"min={im.min():.3f} max={im.max():.3f} std={im.std():.3f}")
        del syn, im                      # rien ne survit a __init__

    def evaluate(self, bb: Backbone, test_feats, test_labels,
                 seeds: Iterable[int] = (0, 1, 2, 3, 4), **probe_kw) -> list[dict]:
        syn = torch.load(self.data_path, map_location="cpu", mmap=True)
        feats = syn_features(syn["images"], bb, self.space)
        labels = syn["labels"].clone()
        del syn
        gc.collect()

        out = []
        for s in seeds:
            _, hist = train_linear_probe(
                feats, labels, test_feats, test_labels,
                feat_dim=bb.feat_dim, seed=s, **probe_kw)
            out.append({"pseed": s, **final_metrics(hist)})

        del feats, labels
        gc.collect()
        return out


# ---- verbatim: exp/benchmark_multipl_cross.ipynb, cell 11 ----
def collect_runs(sources: list[tuple[str, str]]) -> list[DistilledRun]:
    """sources = [(base_dir, regex), ...] — permet de melanger latent et pixel."""
    runs = []
    for base_dir, pattern in sources:
        rx = re.compile(pattern)
        base = Path(base_dir)
        if not base.exists():
            print(f"[skip] repertoire absent : {base_dir}")
            continue
        for p in sorted(base.iterdir()):
            if not (p.is_dir() and rx.match(p.name)):
                continue
            try:
                runs.append(DistilledRun(base, p.name))
            except FileNotFoundError as e:
                print(f"[skip] {e}")
    return runs


# ---- sources (notebook cell 12, parameterized) ----
CONTROLS_DIR = f"{PROJECT_ROOT}/output/controls/results/aqua20/dinov2_vitb/"
CONTROLS_RX = r"^dinov2_vitb_aqua20_(?:none_frozen|affine|slurpp_t1|none_t3)_ipc\d+_s\d+$"
SOURCES = [tuple(s) for s in ARGS.source] if ARGS.source else [(CONTROLS_DIR, CONTROLS_RX)]


# ---- notebook cell 13 (evaluation + contrast-matching grid), unchanged except
#      ARCHS / STD_MODES / MATCHED / full reference taken from the arguments ----
PROBE_SEEDS = (0, 1, 2, 3, 4)
PROBE_KW = dict(max_steps=5000, lr=1e-3, eval_every=1000)  # standardize_feats moved to the grid

MATCH_MODE = ARGS.match_mode   # "real" -> real-image moments ; "fixed" -> (0.5, 0.2)
STD_MODES = tuple(ARGS.std_modes)
MATCHED = tuple(m == "True" for m in ARGS.matched)


@torch.no_grad()
def match_moments(imgs: torch.Tensor, target_mean: float, target_std: float) -> torch.Tensor:
    """Per-set affine rescaling to a common (mean, std), then clamp to [0,1]."""
    mu, sd = imgs.mean(), imgs.std().clamp_min(1e-6)
    return ((imgs - mu) / sd * target_std + target_mean).clamp(0.0, 1.0)


@torch.no_grad()
def real_moments(bb: Backbone, n_batches: int = 20) -> tuple[float, float]:
    """Mean and std of real training images in [0,1], as a matching target."""
    tf = transforms.Compose([
        transforms.Resize(bb.res),
        transforms.CenterCrop(bb.res),
        transforms.ToTensor(),          # no Normalize: we want [0,1]
    ])
    loader = DataLoader(datasets.ImageFolder(f"{DATA_ROOT}/train", transform=tf),
                        batch_size=64, shuffle=False, num_workers=8)
    s, ss, n = 0.0, 0.0, 0
    for i, (x, _) in enumerate(loader):
        s += x.sum().item(); ss += (x ** 2).sum().item(); n += x.numel()
        if i + 1 >= n_batches:
            break
    mean = s / n
    return mean, (ss / n - mean ** 2) ** 0.5


def main():
    print(f"DATA_ROOT={DATA_ROOT}\nDEVICE={DEVICE}\nSOURCES={SOURCES}\n"
          f"archs={ARGS.archs} std_modes={STD_MODES} matched={MATCHED} match_mode={MATCH_MODE}")
    for d in ("train", "test"):
        if not Path(f"{DATA_ROOT}/{d}").is_dir():
            raise SystemExit(f"missing {DATA_ROOT}/{d}")

    distilled_runs = collect_runs(SOURCES)
    print(f"\n{len(distilled_runs)} runs prets")
    if not distilled_runs:
        raise SystemExit("no run found: check --source")

    if not ARGS.no_precheck:      # fail fast if a backbone is not in the offline cache
        for arch in ARGS.archs:
            bb = load_backbone(arch)
            print(f"[precheck] {arch} ok (feat_dim={bb.feat_dim})")
            del bb
            gc.collect(); torch.cuda.empty_cache()

    CHECK_EVALUATE = True          # one-off consistency check vs run.evaluate
    rows = []
    for arch in ARGS.archs:
        bb = load_backbone(arch)
        tf = make_transform(bb)

        test_feats, test_labels = extract_features(
            DataLoader(datasets.ImageFolder(f"{DATA_ROOT}/test", transform=tf),
                       batch_size=64, shuffle=False, num_workers=8),
            bb, f"[{arch}] test")

        # --- full-data reference, same protocol, std_mode="none" only -----------
        if not ARGS.no_full:
            full_feats, full_labels = extract_features(
                DataLoader(datasets.ImageFolder(f"{DATA_ROOT}/train", transform=tf),
                           batch_size=64, shuffle=False, num_workers=8),
                bb, f"[{arch}] full train")

            for ps in PROBE_SEEDS:
                _, hist = train_linear_probe(full_feats, full_labels, test_feats, test_labels,
                                             feat_dim=bb.feat_dim, seed=ps,
                                             standardize_feats="none", **PROBE_KW)
                m = final_metrics(hist)
                rows.append({"arch": arch, "IPC": None, "Variant": "full", "dseed": None,
                             "pseed": ps, "matched": False, "std_mode": "none",
                             "img_mean": float("nan"), "img_std": float("nan"),
                             "F1 macro (%)": m["f1_macro"] * 100,
                             "F1 weighted (%)": m["f1_weighted"] * 100})
            del full_feats, full_labels
            gc.collect()

        tgt_mean, tgt_std = real_moments(bb) if MATCH_MODE == "real" else (0.5, 0.2)
        print(f"[{arch}] matching target: mean={tgt_mean:.4f} std={tgt_std:.4f}")

        # --- distilled runs over the (matched x std_mode) grid ------------------
        for run in distilled_runs:
            syn = torch.load(run.data_path, map_location="cpu", mmap=True)
            imgs = syn_to_pixel(syn["images"], run.space)
            labels = syn["labels"].clone()
            del syn
            gc.collect()

            for matched in MATCHED:
                x = match_moments(imgs, tgt_mean, tgt_std) if matched else imgs
                feats = syn_features(x, bb, space="unit")
                img_mean, img_std = float(x.mean()), float(x.std())
                if matched:
                    del x
                gc.collect()

                for std_mode in STD_MODES:
                    for ps in PROBE_SEEDS:
                        _, hist = train_linear_probe(
                            feats, labels, test_feats, test_labels,
                            feat_dim=bb.feat_dim, seed=ps,
                            standardize_feats=std_mode, **PROBE_KW)
                        m = final_metrics(hist)
                        rows.append({
                            "arch": arch, "IPC": run.ipc, "Variant": run.variant,
                            "dseed": run.dseed, "pseed": ps,
                            "matched": matched, "std_mode": std_mode,
                            "img_mean": img_mean, "img_std": img_std,
                            "F1 macro (%)": m["f1_macro"] * 100,
                            "F1 weighted (%)": m["f1_weighted"] * 100})

                # consistency check: manual path vs run.evaluate, baseline cell only
                if CHECK_EVALUATE and not matched and "none" in STD_MODES:
                    ref = [r["f1_macro"] * 100 for r in run.evaluate(
                        bb, test_feats, test_labels, seeds=PROBE_SEEDS,
                        standardize_feats="none", **PROBE_KW)]
                    here = [r["F1 macro (%)"] for r in rows[-len(PROBE_SEEDS) * len(STD_MODES):]
                            if r["std_mode"] == "none"]
                    dmax = max(abs(a - b) for a, b in zip(sorted(ref), sorted(here)))
                    print(f"  [check] {run.variant} ipc{run.ipc} s{run.dseed}: "
                          f"max |manual - evaluate| = {dmax:.3f} pt")
                    CHECK_EVALUATE = False   # once is enough

                del feats
                gc.collect()

            del imgs, labels
            gc.collect()

        del bb, test_feats, test_labels
        gc.collect(); torch.cuda.empty_cache()

        # checkpoint the CSV after each backbone (a walltime kill keeps finished archs)
        pd.DataFrame(rows).to_csv(ARGS.out, index=False)
        print(f"[{arch}] done, {len(rows)} rows so far -> {ARGS.out}")

    df = pd.DataFrame(rows)
    df.to_csv(ARGS.out, index=False)
    print(f"{len(df)} measurements -> {ARGS.out}")

    # ---- per-run then per-variant summary (notebook cells 14-15), printed only ----
    for std_mode in STD_MODES:
        d = df[(df["matched"] == False) & (df["std_mode"] == std_mode)]
        per_run = (d.groupby(["arch", "IPC", "Variant", "dseed"], dropna=False, observed=True)
                     .agg(macro=("F1 macro (%)", "mean")).reset_index())
        summary = (per_run.groupby(["arch", "IPC", "Variant"], dropna=False, observed=True)
                     .agg(macro_mean=("macro", "mean"), macro_std=("macro", "std"),
                          n=("macro", "count")).reset_index())
        print(f"\n=== matched=False, std_mode={std_mode} (F1 macro, mean/std over distillation seeds)")
        print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
