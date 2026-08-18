# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: pytorch-gpu-2.8.0_py3.12.11
#     language: python
#     name: module-conda-env-pytorch-gpu-2.8.0_py3.12.11
# ---

# %% [markdown]
# ## BASE

# %%
import os
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
from tqdm.notebook import tqdm
 
WORK = "/lustre/fswork/projects/rech/rbw/ucw75ke"
PROJECT_ROOT = f"{WORK}/projects/GradientDistillation"
 
os.environ["HOME"] = WORK                       # au cas ou d'autres libs en dependent
os.environ["TORCH_HOME"] = f"{WORK}/.cache/torch"
os.environ["HF_HOME"] = f"{WORK}/.cache/huggingface"
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
 
for _p in (f"{PROJECT_ROOT}/src", PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# %%
DATA_ROOT = f"{WORK}/datasets/aqua20/data/aqua20"
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


# %%
@dataclass
class Backbone:
    name: str
    model: nn.Module
    forward: Callable[[torch.Tensor], torch.Tensor]
    mean: tuple
    std: tuple
    res: int
    feat_dim: int


# %%
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
 


# %%
@torch.no_grad()
def extract_features(loader, bb: Backbone, desc="Extracting features"):
    feats, labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        feats.append(bb.forward(x.to(DEVICE)).float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)
 


# %%
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
    pix, out = syn_to_pixel(imgs, space), []
    for i in range(0, len(pix), bs):
        x = pix[i:i + bs].to(DEVICE)
        if x.shape[-1] != bb.res:
            x = F.interpolate(x, size=bb.res, mode="bicubic", align_corners=False).clamp(0, 1)
        out.append(bb.forward((x - mean) / std).float().cpu())
    return torch.cat(out)
 


# %%
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


# %%

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
 


# %%
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


# %%
class DistilledRun:
    def __init__(self, base_dir: str | Path, run_name: str):
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        data_path = self.base_dir / run_name / "data.pth"
        if not data_path.exists():
            raise FileNotFoundError(f"data.pth absent pour {run_name!r} (run incomplet ?)")
 
        self.syn_data: dict[str, torch.Tensor] = torch.load(data_path, map_location="cpu")
        im = self.syn_data["images"]
        self.space = detect_image_space(im)
        self.ipc, self.variant, self.dseed = parse_run_name(run_name)
        print(f"[{run_name}] space={self.space} ipc={self.ipc} "
              f"min={im.min():.3f} max={im.max():.3f} std={im.std():.3f}")
 
    def evaluate(self, bb: Backbone, test_feats, test_labels,
                 seeds: Iterable[int] = (0, 1, 2, 3, 4), **probe_kw) -> list[dict]:
        feats = syn_features(self.syn_data["images"], bb, self.space)
        out = []
        for s in seeds:
            _, hist = train_linear_probe(
                feats, self.syn_data["labels"], test_feats, test_labels,
                feat_dim=bb.feat_dim, seed=s, **probe_kw)
            out.append({"pseed": s, **final_metrics(hist)})
        return out


# %%
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


# %%
LATENT_DIR = "../output/latent_exp/results/aqua20/dinov2_vitb/"
PIXEL_DIR = "../output/ipc_ablation/results/aqua20/dinov2_vitb/"
 
SOURCES = [
    (LATENT_DIR,
     r"^dinov2_vitb_latent_(?:decoder_only|predlatent)_medoids_adam_lr1e-3_s\d+(?:_s\d+)?$"),
    (PIXEL_DIR, 
     r"^dinov2_vitb_ipc\d+_.*_s\d+$"),
]
 
distilled_runs = collect_runs(SOURCES)
print(f"\n{len(distilled_runs)} runs prets")

# %%

PROBE_SEEDS = (0, 1, 2, 3, 4)
PROBE_KW = dict(max_steps=5000, lr=1e-3, eval_every=1000, standardize_feats="scalar")
 
rows = []
for name in ARCHS:
    bb = load_backbone(name)
    tf = make_transform(bb)
 
    test_feats, test_labels = extract_features(
        DataLoader(datasets.ImageFolder(f"{DATA_ROOT}/test", transform=tf),
                   batch_size=64, shuffle=False),
        bb, f"[{name}] test")
 
    # reference full-data, meme protocole exactement
    full_feats, full_labels = extract_features(
        DataLoader(datasets.ImageFolder(f"{DATA_ROOT}/train", transform=tf),
                   batch_size=64, shuffle=False),
        bb, f"[{name}] full train")
 
    for s in PROBE_SEEDS:
        _, hist = train_linear_probe(full_feats, full_labels, test_feats, test_labels,
                                     feat_dim=bb.feat_dim, seed=s, **PROBE_KW)
        m = final_metrics(hist)
        rows.append({"arch": name, "IPC": None, "Variant": "full", "dseed": None,
                     "pseed": s,
                     "F1 macro (%)": m["f1_macro"] * 100,
                     "F1 weighted (%)": m["f1_weighted"] * 100})
    del full_feats, full_labels
    gc.collect()
 
    for run in distilled_runs:
        for m in run.evaluate(bb, test_feats, test_labels, seeds=PROBE_SEEDS, **PROBE_KW):
            rows.append({"arch": name, "IPC": run.ipc, "Variant": run.variant,
                         "dseed": run.dseed, "pseed": m["pseed"],
                         "F1 macro (%)": m["f1_macro"] * 100,
                         "F1 weighted (%)": m["f1_weighted"] * 100})
 
    del bb, test_feats, test_labels
    gc.collect()
    torch.cuda.empty_cache()
 
df = pd.DataFrame(rows)
df.to_csv("eval_raw.csv", index=False)
print(f"{len(df)} mesures brutes -> eval_raw.csv") 

# %%

per_run = (df.groupby(["arch", "IPC", "Variant", "dseed"], dropna=False, observed=True)
             .agg(macro=("F1 macro (%)", "mean"),
                  weighted=("F1 weighted (%)", "mean"))
             .reset_index())
 
summary = (per_run.groupby(["arch", "IPC", "Variant"], dropna=False, observed=True)
             .agg(macro_mean=("macro", "mean"), macro_std=("macro", "std"),
                  weighted_mean=("weighted", "mean"), weighted_std=("weighted", "std"),
                  n=("macro", "count"))
             .reset_index())
 
_variants = [v for v in summary["Variant"].dropna().unique() if v != "full"]
summary["Variant"] = pd.Categorical(summary["Variant"],
                                    categories=sorted(_variants) + ["full"], ordered=True)
summary = summary.sort_values(["arch", "IPC", "Variant"], na_position="last").reset_index(drop=True)
 
fmt = lambda mu, sd: f"{mu:.2f}" if pd.isna(sd) else f"{mu:.2f} ± {sd:.2f}"
summary["F1 macro"] = summary.apply(lambda r: fmt(r.macro_mean, r.macro_std), axis=1)
summary["F1 weighted"] = summary.apply(lambda r: fmt(r.weighted_mean, r.weighted_std), axis=1)
 
print(summary[["arch", "IPC", "Variant", "F1 macro", "F1 weighted", "n"]].to_string(index=False))
summary.to_csv("eval_summary.csv", index=False)

# %%
full_std = (df[df["Variant"] == "full"]
            .groupby("arch")["F1 macro (%)"].agg(["mean", "std"]))
print("full (dispersion inter-seeds de probe) :")
print(full_std.round(2), "\n")
 
# variance de probe seule, par variante : doit etre negligeable (< ~0.3 pt)
# devant la variance de distillation, sinon augmenter max_steps
probe_var = (df.groupby(["arch", "Variant", "dseed"], dropna=False, observed=True)["F1 macro (%)"]
               .std().groupby(level=[0, 1]).mean())
print("ecart-type moyen inter-seeds de probe :")
print(probe_var.round(2))
