"""
Phase 1 -- Faisabilite de l'optimisation en espace latent SLURPP.

Design retenu (d'apres la lecture du code reel de slurpp_pipeline.py /
dual_unet_condition.py / diffusers_utils.py) :

  pipe.__call__ et pipe.single_infer sont decores @torch.no_grad() : on ne
  peut donc jamais obtenir de gradient en appelant pipe(...) directement.
  Il faut reimplementer un forward "maison" qui reproduit single_infer SANS
  le no_grad (fonction `differentiable_decompose` ci-dessous).

  Comme denoising_steps=1 force pred_latent = torch.zeros(...) (pas de bruit
  a ensemencer), SLURPP n'a pas de "latent de generation" a la GLaD. La seule
  variable optimisable naturelle est z_u : le latent VAE de l'image sous-marine
  observee "u", partage par les deux branches (unet1 -> clear/J, unet2 ->
  bc/B + ill/T). C'est ce z_u que ce script optimise/teste.

  cld_clear.pth (checkpoint/cld/) n'est charge nulle part dans load_stage1 :
  self.vae_cld reste None et self.skip_connection reste False -> la branche
  "clear" (J) est decodee par le VAE generique de stable-diffusion-2/vae,
  pas par un decodeur dedie. A confirmer avec l'auteur du pipeline.

Tests :
  T1  Round-trip VAE : encode/decode des sorties SLURPP (J, T, B) d'un medoide,
      fidelite par composante (RMSE, PSNR), a 512 et 256, via le VAE partage.
  T2  Gradient flow "option A" (sans le reseau) : loss(vae.decode(z)).backward()
      pour un latent independant par composante -- verifie juste que le VAE
      seul est differentiable (sanity check rapide, bon marche).
  T2bis  Gradient flow "pred_latent post-step" : on fait tourner le reseau UNE
      fois (no_grad) pour obtenir pred_latent (12 canaux, apres le pas DDIM,
      avant le split+decode), puis on detache ce pred_latent et on l'optimise
      directement -- seul le VAE est dans le graphe de la boucle de
      distillation (pas les UNets). Contrainte physique = initialisation
      seulement (pas de garde-fou permanent comme en T3), mais sans la perte
      de round-trip qu'introduirait un re-encodage separe de J/T/B (cf T2).
  T3  Gradient flow "option B" (avec le reseau gele) : differentiable_decompose
      reproduit single_infer sans no_grad, verifie (a) la fidelite numerique
      vs l'inference standard (pipe(...) normal), et (b) que le gradient
      remonte jusqu'a z_u a travers les 2 UNets geles + le VAE.
  T4  Benchmark memoire/temps par step : option A / pred_latent post-step /
      option B (avec/sans gradient checkpointing sur pipe.unet.unet1 /
      pipe.unet.unet2), pour N images (~ IPC * classes).
  T5  Resolution : latent 64x64 (image 512) vs latent 32x32 (image 256),
      chemin decode -> interp vers 252, comparaison visuelle et RMSE.

Usage (login/compute node, env PyTorch IDRIS charge, BASE_CKPT_DIR exporte) :
    python latent_feasibility.py \
        --image /path/to/medoid.jpg \
        --slurpp-root $WORK/projects/slurpp_lib \
        --checkpoint $WORK/projects/SLURPP/slurpp/checkpoint \
        --n-images 20 \
        --outdir output/latent_feasibility

Chaque test est isole (try/except) : un echec n'empeche pas les suivants.
Rapport recapitulatif imprime + sauve en JSON.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from slurpp.diffusers_utils import load_stage1
from slurpp.io import normalize_imgs
from utils.config_util import recursive_load_config


# ============================================================
# Utilitaires
# ============================================================

REPORT = {}


def section(name):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")


def record(test, key, value):
    REPORT.setdefault(test, {})[key] = value
    print(f"  [{test}] {key} = {value}")


def guard(test_name):
    """Decorateur : isole chaque test, log l'exception, continue."""
    def deco(fn):
        def wrapped(*args, **kwargs):
            section(test_name)
            try:
                fn(*args, **kwargs)
                REPORT.setdefault(test_name, {})["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                REPORT.setdefault(test_name, {})["status"] = f"FAILED: {e}"
                traceback.print_exc()
        return wrapped
    return deco


def load_img01(path, size):
    """path -> tensor [1,3,size,size] en [0,1]."""
    pil = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()


def save_grid(tensors, names, path):
    imgs = [t.squeeze(0).clamp(0, 1).detach().cpu() for t in tensors]
    h = max(i.shape[1] for i in imgs)
    imgs = [F.interpolate(i.unsqueeze(0), size=(h, h), mode="bilinear",
                          align_corners=False).squeeze(0) for i in imgs]
    grid = torch.cat(imgs, dim=2)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"  saved {path}  ({' | '.join(names)})")


def rmse(a, b):
    return torch.sqrt(F.mse_loss(a.float(), b.float())).item()


def psnr(a, b):
    m = F.mse_loss(a.clamp(0, 1).float(), b.clamp(0, 1).float()).item()
    return 10 * np.log10(1.0 / max(m, 1e-12))


def cuda_mem_reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def cuda_peak_gb():
    return torch.cuda.max_memory_allocated() / 1024**3


# ============================================================
# Chargement SLURPP (reprend exactement SLURPPInitializer)
# ============================================================

def load_slurpp(slurpp_root, checkpoint_path, device):
    base_ckpt_dir = os.environ.get("BASE_CKPT_DIR")
    if base_ckpt_dir is None:
        raise EnvironmentError("BASE_CKPT_DIR non defini (doit contenir stable-diffusion-2/).")
    model_path = os.path.join(base_ckpt_dir, "stable-diffusion-2")

    cfg = recursive_load_config(os.path.join(slurpp_root, "config", "dual.yaml"))
    pipe, inputs_fields, outputs_fields, dual = load_stage1(model_path, checkpoint_path, cfg)
    assert dual, "ce script suppose dual=True (dual.yaml)"

    pipe = pipe.to(device)
    pipe.scheduler.config.timestep_spacing = "trailing"
    field_idx = {f: i for i, f in enumerate(outputs_fields)}
    print(f"  inputs_fields={inputs_fields}  outputs_fields={outputs_fields}")
    print(f"  unet1.in={pipe.unet.unet1.config['in_channels']} "
          f"out={pipe.unet.unet1.config['out_channels']}  "
          f"unet2.in={pipe.unet.unet2.config['in_channels']} "
          f"out={pipe.unet.unet2.config['out_channels']}")
    if getattr(pipe, "vae_cld", None) is not None:
        record("load", "vae_cld", "present (charge ailleurs ?)")
    else:
        record("load", "vae_cld", "None -> branche clear/J decodee par le VAE generique "
                                    "(cld_clear.pth semble inutilise dans ce chemin)")
    return pipe, inputs_fields, outputs_fields, field_idx


@torch.no_grad()
def slurpp_reference_decompose(pipe, inputs_fields, field_idx, img01, device):
    """Inference SLURPP standard (via pipe(...), non differentiable) -> dict J,T,B en [0,1]."""
    normalized = normalize_imgs(img01.to(device), device=device)
    inputs = [normalized] * len(inputs_fields)
    out = pipe(inputs, denoising_steps=1, show_progress_bar=False, is_dual=True)
    comp = {}
    for name, key in (("T", "ill"), ("B", "bc"), ("J", "clear")):
        i = field_idx[key]
        comp[name] = out[i:i + 1].clamp(0, 1).float().to(device)
    return comp


# ============================================================
# VAE helpers (VAE partage, cf. decode_images / encode_rgb du pipe)
# ============================================================

def vae_encode(pipe, x01):
    """[0,1] -> latent scale SLURPP (rgb_latent_scale_factor), deterministe (mode())."""
    x = x01 * 2.0 - 1.0
    posterior = pipe.vae.encode(x).latent_dist
    return posterior.mode() * pipe.rgb_latent_scale_factor


def vae_decode(pipe, z):
    """latent scale SLURPP -> [0,1], formule exacte de decode_images (sans clamp)."""
    x = pipe.vae.decode(z / pipe.rgb_latent_scale_factor).sample
    return x / 2.0 + 0.5


# ============================================================
# Forward differentiable "option B" : reproduit single_infer sans no_grad
# ============================================================

def compute_pred_latent(pipe, z_u, device):
    """
    z_u : [N,4,h,w] latent de l'image observee "u" (echelle rgb_latent_scale_factor),
          partage par les deux branches (comme a l'inference : [normalized]*2).

    Reproduit la partie "reseau" de single_infer() dual, 1 pas DDIM, SANS
    @torch.no_grad() : rgb_latents (fixe, deduit de z_u) -> dual-UNet -> DDIM
    step -> pred_latent (12 canaux, AVANT split + decode VAE).

    Retourne pred_latent, differentiable par rapport a z_u (si z_u.requires_grad).
    """
    N = z_u.shape[0]

    pipe.scheduler.set_timesteps(1, device=device)
    t = pipe.scheduler.timesteps[0]

    if pipe.empty_text_embed is None:
        pipe.encode_empty_text()
    text_embed = pipe.empty_text_embed.repeat(N, 1, 1).to(device)

    # meme image "u" fournie aux deux branches, comme a l'inference standard
    rgb_latents = torch.cat([z_u, z_u], dim=1)  # [N, 8, h, w]

    out_ch = pipe.unet.unet1.config["out_channels"] + pipe.unet.unet2.config["out_channels"]
    pred_latent = torch.zeros(N, out_ch, z_u.shape[-2], z_u.shape[-1],
                               device=device, dtype=z_u.dtype)  # 1 step => init a zero

    unet_input = torch.cat([rgb_latents, pred_latent], dim=1)  # [N, 20, h, w]
    noise_pred = pipe.unet(unet_input, t, encoder_hidden_states=text_embed).sample
    pred_latent = pipe.scheduler.step(noise_pred, t, pred_latent).prev_sample
    return pred_latent


def decode_pred_latent(pipe, pred_latent):
    """
    pred_latent : [N,12,h,w], AVANT split -- ordre confirme par
    DualUNetCondition.forward (cat([unet1_out(clear,4ch), unet2_out(bc+ill,8ch)]))
    et par decode_images (reshape batch*3, 4, H, W).

    Decode via le VAE partage -> (J, B, T) en [0,1] (non clampes), differentiable
    par rapport a pred_latent.
    """
    clear_lat, bc_lat, ill_lat = pred_latent.split(4, dim=1)
    J = vae_decode(pipe, clear_lat)
    B = vae_decode(pipe, bc_lat)
    T = vae_decode(pipe, ill_lat)
    return J, B, T


def differentiable_decompose(pipe, z_u, device):
    """
    z_u -> pred_latent (a travers le dual-UNet gele) -> (J, B, T) decodes.
    Enchainement complet de compute_pred_latent + decode_pred_latent, utilise
    par T3 (le reseau reste dans le graphe de bout en bout).
    """
    pred_latent = compute_pred_latent(pipe, z_u, device)
    return decode_pred_latent(pipe, pred_latent)


def compose_slurpp(J, T, B):
    return J * T + B


# ============================================================
# T1 -- Round-trip VAE sur les composantes SLURPP
# ============================================================

@guard("T1_roundtrip_vae")
def t1_roundtrip(pipe, comp, outdir, tag):
    for name, x in comp.items():
        with torch.no_grad():
            z = vae_encode(pipe, x)
            xr = vae_decode(pipe, z).clamp(0, 1)
        record("T1_roundtrip_vae", f"{tag}/{name}/rmse", round(rmse(x, xr), 5))
        record("T1_roundtrip_vae", f"{tag}/{name}/psnr_db", round(psnr(x, xr), 2))
        record("T1_roundtrip_vae", f"{tag}/{name}/latent_shape", tuple(z.shape))
        save_grid([x, xr, (x - xr).abs() * 5],
                  [f"{name} orig", f"{name} roundtrip", "abs diff x5"],
                  outdir / f"t1_{tag}_{name}.png")
    if all(k in comp for k in ("J", "T", "B")):
        with torch.no_grad():
            I = compose_slurpp(comp["J"], comp["T"], comp["B"]).clamp(0, 1)
            Jr = vae_decode(pipe, vae_encode(pipe, comp["J"])).clamp(0, 1)
            Tr = vae_decode(pipe, vae_encode(pipe, comp["T"])).clamp(0, 1)
            Br = vae_decode(pipe, vae_encode(pipe, comp["B"])).clamp(0, 1)
            Ir = compose_slurpp(Jr, Tr, Br).clamp(0, 1)
        record("T1_roundtrip_vae", f"{tag}/I_composed/rmse", round(rmse(I, Ir), 5))
        record("T1_roundtrip_vae", f"{tag}/I_composed/psnr_db", round(psnr(I, Ir), 2))
        save_grid([I, Ir], ["I compose orig", "I compose roundtrip"],
                  outdir / f"t1_{tag}_I.png")
    record("T1_roundtrip_vae", "NOTE",
           "J/T/B tous decodes par le VAE generique (vae_cld=None) -- "
           "a confirmer que cld_clear.pth n'est pas cense etre charge ici.")


# ============================================================
# T2 -- Gradient flow "option A" : VAE seul, sans le reseau
# ============================================================

@guard("T2_gradflow_vae_only")
def t2_gradflow_vae_only(pipe, comp):
    with torch.no_grad():
        z0 = vae_encode(pipe, comp["J"])
    z = z0.detach().clone().requires_grad_(True)
    J = vae_decode(pipe, z)
    T, B = comp["T"].detach(), comp["B"].detach()
    loss = compose_slurpp(J, T, B).mean()
    loss.backward()
    ok = z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    record("T2_gradflow_vae_only", "z_grad_ok", bool(ok))
    record("T2_gradflow_vae_only", "z_grad_norm", float(z.grad.norm()) if ok else None)
    assert ok, "pas de gradient sur z a travers vae.decode seul"


# ============================================================
# T2bis -- Gradient flow "pred_latent post-step" : reseau hors boucle,
# VAE seul dans le graphe de distillation
# ============================================================

@guard("T2bis_gradflow_predlatent_poststep")
def t2bis_gradflow_predlatent_poststep(pipe, img01, device, reference_comp):
    """
    Le reseau (2 UNets geles) tourne UNE fois en no_grad pour produire
    pred_latent (init "physique"). Ensuite pred_latent est detache et rendu
    optimisable : la boucle de distillation ne repasse plus que par le VAE
    (pas de UNet), moins cher que T3, mais sans garde-fou physique permanent
    (rien n'empeche pred_latent de deriver hors de ce que le reseau aurait
    genere pour une image coherente).
    """
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        normalized = normalize_imgs(img01.to(device), device=device)
        z0 = pipe.encode_rgb(normalized)
        pred_latent_init = compute_pred_latent(pipe, z0, device)  # 1 passage reseau, no_grad

    # fidelite : ce pred_latent decode doit reproduire la reference (meme
    # forward que T3, donc meme garantie de non-regression)
    with torch.no_grad():
        J0, B0, T0 = decode_pred_latent(pipe, pred_latent_init)
        J0, B0, T0 = J0.clamp(0, 1), B0.clamp(0, 1), T0.clamp(0, 1)
    for name, ref, mine in (("J", reference_comp["J"], J0),
                             ("B", reference_comp["B"], B0),
                             ("T", reference_comp["T"], T0)):
        record("T2bis_gradflow_predlatent_poststep",
               f"init_vs_reference/{name}/rmse", round(rmse(ref, mine), 5))

    # optimisation : pred_latent devient le parametre, seul le VAE est dans le graphe
    pred_latent = pred_latent_init.detach().clone().requires_grad_(True)
    J, B, T = decode_pred_latent(pipe, pred_latent)
    loss = compose_slurpp(J, T, B).mean()
    loss.backward()
    ok = (pred_latent.grad is not None and torch.isfinite(pred_latent.grad).all()
          and pred_latent.grad.abs().sum() > 0)
    record("T2bis_gradflow_predlatent_poststep", "pred_latent_grad_ok", bool(ok))
    record("T2bis_gradflow_predlatent_poststep", "pred_latent_grad_norm",
           float(pred_latent.grad.norm()) if ok else None)
    record("T2bis_gradflow_predlatent_poststep", "pred_latent_shape", tuple(pred_latent.shape))
    assert ok, "pas de gradient sur pred_latent a travers le VAE seul"


# ============================================================
# T3 -- Gradient flow "option B" : a travers le reseau dual-UNet gele
# ============================================================

@guard("T3_gradflow_frozen_unet")
def t3_gradflow_frozen_unet(pipe, img01, device, reference_comp):
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        normalized = normalize_imgs(img01.to(device), device=device)
        z0 = pipe.encode_rgb(normalized)

    with torch.no_grad():
        J0, B0, T0 = differentiable_decompose(pipe, z0, device)
        J0, B0, T0 = J0.clamp(0, 1), B0.clamp(0, 1), T0.clamp(0, 1)
    for name, ref, mine in (("J", reference_comp["J"], J0),
                             ("B", reference_comp["B"], B0),
                             ("T", reference_comp["T"], T0)):
        record("T3_gradflow_frozen_unet", f"reforward_vs_reference/{name}/rmse",
               round(rmse(ref, mine), 5))
    record("T3_gradflow_frozen_unet", "NOTE_fidelity",
           "rmse devrait etre proche de 0 (a la precision fp near) : "
           "confirme que differentiable_decompose reproduit bien single_infer.")

    z_u = z0.detach().clone().requires_grad_(True)
    J, B, T = differentiable_decompose(pipe, z_u, device)
    loss = compose_slurpp(J, T, B).mean()
    loss.backward()
    ok = z_u.grad is not None and torch.isfinite(z_u.grad).all() and z_u.grad.abs().sum() > 0
    record("T3_gradflow_frozen_unet", "z_u_grad_ok", bool(ok))
    record("T3_gradflow_frozen_unet", "z_u_grad_norm", float(z_u.grad.norm()) if ok else None)
    assert ok, "pas de gradient sur z_u a travers le dual-UNet + VAE"


# ============================================================
# T4 -- Benchmark memoire / temps par step
# ============================================================

def _bench(fn, n_warmup=2, n_iter=5):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    cuda_mem_reset()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter
    return dt, cuda_peak_gb()


@guard("T4_benchmark")
def t4_benchmark(pipe, comp, img01, device, n_images):
    with torch.no_grad():
        zJ = vae_encode(pipe, comp["J"]).repeat(n_images, 1, 1, 1)
        zT = vae_encode(pipe, comp["T"]).repeat(n_images, 1, 1, 1)
        zB = vae_encode(pipe, comp["B"]).repeat(n_images, 1, 1, 1)
    params = [z.detach().clone().requires_grad_(True) for z in (zJ, zT, zB)]

    def step_a():
        for p in params:
            if p.grad is not None:
                p.grad = None
        J, T, B = (vae_decode(pipe, p) for p in params)
        I = compose_slurpp(J, T, B)
        I = F.interpolate(I, size=(252, 252), mode="bilinear", align_corners=False)
        I.mean().backward()

    try:
        dt, mem = _bench(step_a)
        record("T4_benchmark", f"optionA/N{n_images}/ms_per_step", round(dt * 1e3, 1))
        record("T4_benchmark", f"optionA/N{n_images}/peak_gb", round(mem, 2))
    except torch.cuda.OutOfMemoryError:
        record("T4_benchmark", f"optionA/N{n_images}", "OOM -> chunker les decodes")
        torch.cuda.empty_cache()
    del params, zJ, zT, zB
    torch.cuda.empty_cache()

    # ---- option pred_latent post-step : reseau hors boucle (1 passage no_grad
    # pour l'init), puis VAE seul dans le graphe de distillation, N images ----
    with torch.no_grad():
        normalized = normalize_imgs(img01.to(device), device=device)
        z0_single = pipe.encode_rgb(normalized)
        pred_latent_init = compute_pred_latent(pipe, z0_single, device).repeat(n_images, 1, 1, 1)
    pred_latent_param = pred_latent_init.detach().clone().requires_grad_(True)

    def step_predlatent():
        if pred_latent_param.grad is not None:
            pred_latent_param.grad = None
        J, B, T = decode_pred_latent(pipe, pred_latent_param)
        I = compose_slurpp(J, T, B)
        I = F.interpolate(I, size=(252, 252), mode="bilinear", align_corners=False)
        I.mean().backward()

    try:
        dt, mem = _bench(step_predlatent)
        record("T4_benchmark", f"predlatent_poststep/N{n_images}/ms_per_step", round(dt * 1e3, 1))
        record("T4_benchmark", f"predlatent_poststep/N{n_images}/peak_gb", round(mem, 2))
    except torch.cuda.OutOfMemoryError:
        record("T4_benchmark", f"predlatent_poststep/N{n_images}", "OOM -> chunker les decodes")
        torch.cuda.empty_cache()
    del pred_latent_param, pred_latent_init
    torch.cuda.empty_cache()

    with torch.no_grad():
        z0 = pipe.encode_rgb(normalized).repeat(n_images, 1, 1, 1)

    for ckpt in (False, True):
        label = "ckpt_on" if ckpt else "ckpt_off"
        try:
            (pipe.unet.unet1.enable_gradient_checkpointing() if ckpt
             else pipe.unet.unet1.disable_gradient_checkpointing())
            (pipe.unet.unet2.enable_gradient_checkpointing() if ckpt
             else pipe.unet.unet2.disable_gradient_checkpointing())
            n_this = 1 if not ckpt else n_images
            z0_this = z0[:n_this]

            def step_b_n():
                z_u = z0_this.detach().clone().requires_grad_(True)
                J, B, T = differentiable_decompose(pipe, z_u, device)
                I = compose_slurpp(J, T, B)
                I = F.interpolate(I, size=(252, 252), mode="bilinear", align_corners=False)
                I.mean().backward()

            dt, mem = _bench(step_b_n, n_warmup=1, n_iter=3)
            record("T4_benchmark", f"optionB/N{n_this}/{label}/ms_per_step", round(dt * 1e3, 1))
            record("T4_benchmark", f"optionB/N{n_this}/{label}/peak_gb", round(mem, 2))
        except torch.cuda.OutOfMemoryError:
            record("T4_benchmark", f"optionB/{label}", f"OOM a N={n_images} -> chunker / accumuler")
            torch.cuda.empty_cache()
        except Exception as e:
            record("T4_benchmark", f"optionB/{label}", f"indisponible: {e}")
            torch.cuda.empty_cache()


# ============================================================
# T5 -- Question de resolution
# ============================================================

@guard("T5_resolution")
def t5_resolution(pipe, comp512, comp256, outdir):
    with torch.no_grad():
        J512 = vae_decode(pipe, vae_encode(pipe, comp512["J"])).clamp(0, 1)
        J512_252 = F.interpolate(J512, size=(252, 252), mode="bilinear", align_corners=False)
        J256 = vae_decode(pipe, vae_encode(pipe, comp256["J"])).clamp(0, 1)
        J256_252 = F.interpolate(J256, size=(252, 252), mode="bilinear", align_corners=False)
        ref = F.interpolate(comp512["J"], size=(252, 252), mode="bilinear", align_corners=False)
    record("T5_resolution", "J_512path_rmse_vs_ref", round(rmse(ref, J512_252), 5))
    record("T5_resolution", "J_256path_rmse_vs_ref", round(rmse(ref, J256_252), 5))
    record("T5_resolution", "latent_shape_512", tuple(vae_encode(pipe, comp512["J"]).shape))
    record("T5_resolution", "latent_shape_256", tuple(vae_encode(pipe, comp256["J"]).shape))
    save_grid([ref, J512_252, J256_252],
              ["ref 252", "via latent64 (512)", "via latent32 (256)"],
              outdir / "t5_resolution_J.png")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="chemin d'un medoide (jpg/png)")
    ap.add_argument("--slurpp-root", required=True, help="depot CODE (slurpp_lib), contient config/dual.yaml")
    ap.add_argument("--checkpoint", required=True, help="depot POIDS : SLURPP/slurpp/checkpoint")
    ap.add_argument("--n-images", type=int, default=20, help="N pour le benchmark (IPC x classes)")
    ap.add_argument("--outdir", default="output/latent_feasibility")
    args = ap.parse_args()

    device = torch.device("cuda")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    section("Chargement SLURPP")
    pipe, inputs_fields, outputs_fields, field_idx = load_slurpp(
        args.slurpp_root, args.checkpoint, device)

    section("Decomposition SLURPP de reference (pipe standard, no_grad) -- 512 et 256")
    img512 = load_img01(args.image, 512)
    img256 = load_img01(args.image, 256)
    comp512 = slurpp_reference_decompose(pipe, inputs_fields, field_idx, img512, device)
    comp256 = slurpp_reference_decompose(pipe, inputs_fields, field_idx, img256, device)
    print(f"  composantes: {list(comp512.keys())}")

    t1_roundtrip(pipe, comp512, outdir, tag="512")
    t1_roundtrip(pipe, comp256, outdir, tag="256")
    t2_gradflow_vae_only(pipe, comp512)
    t2bis_gradflow_predlatent_poststep(pipe, img512, device, comp512)
    t3_gradflow_frozen_unet(pipe, img512, device, comp512)
    t4_benchmark(pipe, comp512, img512, device, args.n_images)
    t5_resolution(pipe, comp512, comp256, outdir)

    section("RAPPORT")
    print(json.dumps(REPORT, indent=2, default=str))
    with open(outdir / "report.json", "w") as f:
        json.dump(REPORT, f, indent=2, default=str)
    print(f"\nRapport sauve dans {outdir/'report.json'}")


if __name__ == "__main__":
    main()