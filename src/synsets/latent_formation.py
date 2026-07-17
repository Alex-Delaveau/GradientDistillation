from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from config import DistillCfg
from data.dataloaders import BaseRealDataset
from my_utils.device import DeviceSingleton
from my_utils.log_utils import log_images

from .base import BaseDistilledDataset

from src.initializer.priors import build_prior
from src.initializer.sample import build_sample_init, ordered_indices

from src.my_utils.log_utils import log


# ============================================================
# Decodage chunke exact (backward en deux passes)
# ============================================================
#
# La loss LGM (cosinus entre gradients du classifieur) n'est pas une somme
# par image : l'accumulation de gradient naive "loss par chunk" est fausse.
# ChunkedDecodeCompose fait a la place :
#   forward  : decode + compose en no_grad, chunk par chunk -> I (pas de
#              graphe conserve, pas d'OOM)
#   backward : re-decode chunk par chunk sous enable_grad et injecte le
#              gradient amont recu (vector-Jacobian product) -> gradient
#              EXACT sur pred_latent, pic memoire borne par chunk_size.
# La boucle d'entrainement (get_data -> loss -> backward -> step) reste
# inchangee.

class ChunkedDecodeCompose(torch.autograd.Function):
    """I = compose(vae_decode(clear), vae_decode(ill), vae_decode(bc)), chunke."""

    @staticmethod
    def forward(ctx, pred_latent: Tensor, decode_fn, compose_fn, chunk_size: int):
        ctx.decode_fn = decode_fn
        ctx.compose_fn = compose_fn
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(pred_latent)

        outs = []
        with torch.no_grad():
            for i in range(0, pred_latent.shape[0], chunk_size):
                chunk = pred_latent[i:i + chunk_size]
                J, B, T = decode_fn(chunk)
                outs.append(compose_fn(J, T, B))
        return torch.cat(outs, dim=0)

    @staticmethod
    def backward(ctx, grad_I: Tensor):
        (pred_latent,) = ctx.saved_tensors
        grad_latent = torch.zeros_like(pred_latent)
        for i in range(0, pred_latent.shape[0], ctx.chunk_size):
            chunk = pred_latent[i:i + ctx.chunk_size].detach().requires_grad_(True)
            with torch.enable_grad():
                J, B, T = ctx.decode_fn(chunk)
                I_chunk = ctx.compose_fn(J, T, B)
            (g,) = torch.autograd.grad(I_chunk, chunk,
                                       grad_outputs=grad_I[i:i + ctx.chunk_size])
            grad_latent[i:i + ctx.chunk_size] = g
        return grad_latent, None, None, None


class LatentFormationDataset(BaseDistilledDataset):
    """
    Distilled dataset parameterized in SLURPP's latent space (GLaD-style).

    latent_mode = "predlatent" (implemented):
        The frozen SLURPP dual-UNet runs ONCE per medoid at init to produce
        pred_latent [N, 12, h, w] (clear/bc/ill stacked, post DDIM step,
        pre VAE-decode). pred_latent is then THE optimized parameter; the
        distillation loop only goes through the (frozen) VAE decoder:
            pred_latent -> split(4) -> vae.decode x3 -> (J, B, T)
            I = compose(J, T, B) -> resize to cfg.syn_res
        Physics constraint acts as initialization only (no permanent
        network guard, contrary to the future "z_u" mode).

    Feasibility findings this design relies on (job 1932676):
        - VAE round-trips T/B/J faithfully at 512 (>= 38 dB) -> all-latent OK
        - 256/latent-32 loses 8 dB on J -> work at 512 / latent 64x64
        - gradient flows through DDIMScheduler.step and vae.decode
        - N=20 in a single graph OOMs on 80GB -> chunked two-pass backward

    Expected cfg fields (new):
        cfg.latent_mode        : "predlatent" (later: "independent", "z_u")
        cfg.latent_chunk       : images per decode chunk (default 2)
        cfg.latent_res         : SLURPP working resolution (default 512)
        cfg.lr                 : latent learning rate (needs its own tuning:
                                 pixel-space LRs are meaningless here)
    """

    def __init__(self, train_dataset: BaseRealDataset, cfg: DistillCfg,
                 backbone: torch.nn.Module = None, num_feat: int = None):
        super().__init__()
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.device = DeviceSingleton.get()
        self.N = cfg.ipc * train_dataset.num_classes
        self.chunk = getattr(cfg, "latent_chunk", 2)
        self.latent_res = getattr(cfg, "latent_res", 512)

        assert getattr(cfg, "latent_mode", "predlatent") == "predlatent", \
            "only latent_mode=predlatent is implemented for now"

        (self.pred_latent, self.syn_labels) = self.init_synset(backbone, num_feat)
        self.optimizer = self.init_optimizer()

    # ----- init -----

    def init_synset(self, backbone, num_feat):
        log(type(self).__name__, f"building sample init ({self.cfg.sample_init})")
        sample_init = build_sample_init(backbone, num_feat, self.cfg, self.train_dataset)

        log(type(self).__name__, "building SLURPP prior (latent mode requires slurpp)")
        assert self.cfg.prior_init == "slurpp", "latent formation requires prior_init=slurpp"
        prior_init = build_prior(self.cfg, self.device)
        self.compose = prior_init.compose  # I = J*T + B (SLURPP variant)
        self.formation_name = prior_init.formation_name()

        log(type(self).__name__, "selecting samples")
        self.sample_indices, label_list = ordered_indices(
            sample_init.get_indices(), self.train_dataset, self.cfg.ipc)
        syn_labels = torch.tensor(label_list, dtype=torch.long, device=self.device)

        # --- one frozen forward per medoid: image -> z_u -> dual-UNet -> pred_latent ---
        pipe = prior_init.pipe
        pipe.scheduler.config.timestep_spacing = "trailing"
        for p in pipe.unet.parameters():
            p.requires_grad_(False)
        for p in pipe.vae.parameters():
            p.requires_grad_(False)

        self._scale = pipe.rgb_latent_scale_factor

        log(type(self).__name__,
            f"computing pred_latent init for {len(self.sample_indices)} medoids "
            f"@ {self.latent_res}px (chunk={self.chunk})")
        latents = []
        with torch.no_grad():
            for i in self.sample_indices:
                img01 = self._load_medoid(i)                       # [1,3,R,R] in [0,1]
                z_u = self._encode_rgb(pipe, img01)                # [1,4,r,r]
                latents.append(self._compute_pred_latent(pipe, z_u))
        pred0 = torch.cat(latents, dim=0)                          # [N,12,r,r]

        self.latent_prior = pred0.detach().clone()                 # drift reference
        pred_latent = pred0.detach().clone().requires_grad_(True)

        # --- keep only the VAE decoder; free UNets/encoder/text stack ---
        self.vae = pipe.vae
        self._free_prior_pipeline(prior_init, sample_init, pipe)

        log(type(self).__name__,
            f"synset init done | pred_latent {tuple(pred_latent.shape)} "
            f"({pred_latent.numel() // self.N} params/img)")
        return pred_latent, syn_labels

    def _load_medoid(self, index) -> Tensor:
        """Real image -> [1,3,latent_res,latent_res] float in [0,1] on device."""
        from PIL import Image
        import numpy as np
        path = self.train_dataset.get_path(index)
        pil = Image.open(path).convert("RGB").resize(
            (self.latent_res, self.latent_res), Image.BICUBIC)
        arr = np.array(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def _encode_rgb(self, pipe, img01: Tensor) -> Tensor:
        from slurpp.io import normalize_imgs
        normalized = normalize_imgs(img01, device=self.device)
        return pipe.encode_rgb(normalized)

    def _compute_pred_latent(self, pipe, z_u: Tensor) -> Tensor:
        """Mirror of the validated feasibility forward (single_infer, 1 DDIM step)."""
        N = z_u.shape[0]
        pipe.scheduler.set_timesteps(1, device=self.device)
        t = pipe.scheduler.timesteps[0]
        if pipe.empty_text_embed is None:
            pipe.encode_empty_text()
        text_embed = pipe.empty_text_embed.repeat(N, 1, 1).to(self.device)

        rgb_latents = torch.cat([z_u, z_u], dim=1)
        out_ch = (pipe.unet.unet1.config["out_channels"]
                  + pipe.unet.unet2.config["out_channels"])
        pred = torch.zeros(N, out_ch, z_u.shape[-2], z_u.shape[-1],
                           device=self.device, dtype=z_u.dtype)
        unet_input = torch.cat([rgb_latents, pred], dim=1)
        noise_pred = pipe.unet(unet_input, t, encoder_hidden_states=text_embed).sample
        return pipe.scheduler.step(noise_pred, t, pred).prev_sample

    def init_optimizer(self):
        groups = [{"params": [self.pred_latent], "lr": self.cfg.lr}]
        return self.build_optimizer(groups)

    # ----- decode path -----

    def _vae_decode(self, z: Tensor) -> Tensor:
        x = self.vae.decode(z / self._scale).sample
        return x / 2.0 + 0.5

    def _decode_JBT(self, pred_latent_chunk: Tensor):
        clear_lat, bc_lat, ill_lat = pred_latent_chunk.split(4, dim=1)
        J = self._vae_decode(clear_lat)
        B = self._vae_decode(bc_lat)
        T = self._vae_decode(ill_lat)
        return J, B, T

    # ----- forward -----

    def get_data(self) -> Tuple[Tensor, Tensor]:
        I = ChunkedDecodeCompose.apply(
            self.pred_latent, self._decode_JBT, self.compose, self.chunk)
        I = F.interpolate(I, size=(self.cfg.syn_res, self.cfg.syn_res),
                          mode="bilinear", align_corners=False)
        if getattr(self.cfg, "clamp_I", True):
            self._sat_frac = (I > 1.0).float().mean().item()
            I = I.clamp(0.0, 1.0)
        return I, self.syn_labels

    # ----- research / save -----

    @torch.no_grad()
    def _decode_all(self):
        Js, Bs, Ts = [], [], []
        for i in range(0, self.N, self.chunk):
            J, B, T = self._decode_JBT(self.pred_latent[i:i + self.chunk])
            Js.append(J); Bs.append(B); Ts.append(T)
        return torch.cat(Js), torch.cat(Bs), torch.cat(Ts)

    @torch.no_grad()
    def get_snapshot(self) -> dict:
        J, B, T = self._decode_all()
        I = self.compose(J, T, B)
        if getattr(self.cfg, "clamp_I", True):
            I = I.clamp(0.0, 1.0)
        return {
            "I": I.cpu(), "J": J.clamp(0, 1).cpu(),
            "T": T.clamp(0, 1).cpu(), "B": B.clamp(0, 1).cpu(),
            "formation": self.formation_name,
            "latent_mode": "predlatent",
        }

    @torch.no_grad()
    def get_to_save(self) -> dict:
        J, B, T = self._decode_all()
        I = self.compose(J, T, B)
        if getattr(self.cfg, "clamp_I", True):
            I = I.clamp(0.0, 1.0)
        I_out = F.interpolate(I, size=(self.cfg.syn_res, self.cfg.syn_res),
                              mode="bilinear", align_corners=False)
        sample_paths = [self.train_dataset.get_path(i) for i in self.sample_indices]
        return {
            "syn_data": (I_out, self.syn_labels),
            "save_dict": {
                "pred_latent": self.pred_latent.detach(),
                "latent_prior": self.latent_prior,
                "syn_J": J.clamp(0, 1), "syn_T": T.clamp(0, 1), "syn_B": B.clamp(0, 1),
                "sample_indices": torch.tensor(self.sample_indices),
                "sample_paths": sample_paths,
                "sample_init": self.cfg.sample_init,
            },
        }

    def upkeep(self, step: int = None):
        pass  # no pyramid schedule in latent mode

    @torch.no_grad()
    def log_images(self, step: int = None):
        if self.N > 100:
            print("Warning: too many images to log")
            return
        I, _ = self.get_data()
        log_images(syn_images=I.detach().clone(), step=step)

    def get_save_dict(self):
        return {
            "pred_latent": self.pred_latent,
            "opt_state": self.optimizer.state_dict(),
        }

    def load_from_dict(self, load_dict: dict):
        with torch.no_grad():
            self.pred_latent.copy_(load_dict["pred_latent"])
        self.optimizer = self.init_optimizer()
        self.optimizer.load_state_dict(load_dict["opt_state"])
        log(type(self).__name__, "loaded from checkpoint")

    # ------ Metrics ------

    @torch.no_grad()
    def physics_metrics(self) -> dict:
        drift = torch.sqrt(F.mse_loss(self.pred_latent, self.latent_prior))
        per_comp = {}
        for name, sl in (("clear", slice(0, 4)), ("bc", slice(4, 8)), ("ill", slice(8, 12))):
            per_comp[f"latent/{name}_drift_rmse"] = torch.sqrt(F.mse_loss(
                self.pred_latent[:, sl], self.latent_prior[:, sl])).item()
        return {
            "latent/drift_rmse": drift.item(),
            "latent/norm": self.pred_latent.norm().item(),
            "physics/sat_frac": getattr(self, "_sat_frac", 0.0),
            **per_comp,
        }

    @torch.no_grad()
    def gradient_metrics(self, grads: dict) -> dict:
        out = super().gradient_metrics(grads)  # grad/norm_total
        g = self.pred_latent.grad
        out["grad/norm_latent"] = g.norm().item() if g is not None else 0.0
        if g is not None:
            for name, sl in (("clear", slice(0, 4)), ("bc", slice(4, 8)), ("ill", slice(8, 12))):
                out[f"grad/norm_latent_{name}"] = g[:, sl].norm().item()
        return out

    # ------ Cleanup ------

    def _free_prior_pipeline(self, prior_init, sample_init, pipe):
        """Keep only the VAE (decoder path); free UNets, text stack, encoder refs."""
        pipe.unet = None
        pipe.text_encoder = None
        pipe.tokenizer = None
        del prior_init, sample_init, pipe
        import gc
        gc.collect()
        torch.cuda.empty_cache()