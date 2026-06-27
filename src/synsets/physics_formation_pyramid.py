from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from config import DistillCfg
from data.dataloaders import BaseRealDataset
from my_utils.device import DeviceSingleton
from my_utils.log_utils import log_images

from .base import BaseDistilledDataset
from src.initializer.sample.medoids_init import MedoidInitializer
from src.initializer.sample.random_init import RandomInitializer
from src.initializer.priors.ppg_init import PPGInitializer
from src.initializer.priors.slurpp_init import SLURPPInitializer
from src.initializer.priors.none_init import NonePriorInitializer


def _log(msg: str) -> None:
    """Single-line progress log, flushed for live SLURM output."""
    print(f"[PhysicsFormation] {msg}", flush=True)


class PhysicsFormationDataset(BaseDistilledDataset):
    """
    Parameterized underwater image-formation model, with J encoded as a
    multi-resolution coarse-to-fine pyramid.

    Formation equation (carried by the prior initializer's `compose`):
        Koschmieder (PPG) : I = J*T + (1-T)*B
        SLURPP            : I = J*T + B        (B is the full additive term)

    T and B are physical priors (cfg.prior_init), reparameterized through
    logit -> sigmoid. Either can be frozen (cfg.freeze_T / cfg.freeze_B) to
    act as a fixed prior instead of being jointly optimized.
    """

    def __init__(self, train_dataset: BaseRealDataset, cfg: DistillCfg,
                 backbone: torch.nn.Module = None, num_feat: int = None):
        super().__init__()
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.device = DeviceSingleton.get()
        self.N = cfg.ipc * train_dataset.num_classes

        (self.pyramid_J, self.syn_T, self.syn_B,
         self.syn_labels) = self.init_synset(backbone, num_feat)
        self.optimizer = self.init_optimizer()

    # ----- builders -----

    def _build_sample_init(self, backbone, num_feat):
        if self.cfg.sample_init == "medoids":
            return MedoidInitializer(self.cfg, self.train_dataset, backbone, num_feat, self.cfg.seed)
        if self.cfg.sample_init == "random":
            return RandomInitializer(self.cfg, self.train_dataset, seed=self.cfg.seed)
        raise ValueError(f"sample_init unknown: {self.cfg.sample_init}")

    def _build_prior_init(self):
        if self.cfg.prior_init == "none":
            return NonePriorInitializer(self.cfg)
        if self.cfg.prior_init == "ppg":
            return PPGInitializer(self.cfg)
        if self.cfg.prior_init == "slurpp":
            slurpp_device = getattr(self.cfg, "slurpp_device", str(self.device))
            return SLURPPInitializer(
                self.cfg.slurpp_root,
                self.cfg.slurpp_checkpoint_path,
                device=self.device,
            )
        raise ValueError(f"prior_init unknown: {self.cfg.prior_init}")

    # ----- init -----

    def _ordered_indices(self, sample_idx: dict) -> Tuple[list, list]:
        """Flatten {class: [global indices]} into class-major order."""
        ordered, labels = [], []
        for c in range(self.train_dataset.num_classes):
            idxs = list(sample_idx[c])
            if len(idxs) < self.cfg.ipc:                 # class too small -> repeat
                idxs = (idxs * self.cfg.ipc)[:self.cfg.ipc]
            ordered += idxs[:self.cfg.ipc]
            labels += [c] * self.cfg.ipc
        return ordered, labels

    def _prep_prior(self, x: Tensor, channels: int, spatial: bool) -> Tensor:
        """native [0,1] -> [1, channels, syn_res, syn_res] (or [...,1,1] if not spatial)."""
        x = x.to(self.device).float()
        x = F.interpolate(x, (self.cfg.syn_res, self.cfg.syn_res),
                          mode="bilinear", align_corners=False)
        if x.shape[1] == 1 and channels == 3:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3 and channels == 1:
            x = x.mean(1, keepdim=True)
        if not spatial:
            x = x.mean(dim=(2, 3), keepdim=True)
        return x.clamp(1e-4, 1 - 1e-4)

    def _make_param(self, t: Tensor, trainable: bool) -> Tensor:
        return t.detach().clone().requires_grad_(trainable)

    def init_synset(self, backbone, num_feat):
        _log(f"building sample init ({self.cfg.sample_init})")
        sample_init = self._build_sample_init(backbone, num_feat)

        _log(f"building prior init ({self.cfg.prior_init})")
        prior_init = self._build_prior_init()
        self.compose = prior_init.compose
        self.formation_name = type(prior_init).__name__

        _log("selecting samples")
        ordered_idx, label_list = self._ordered_indices(sample_init.get_indices())
        self.sample_indices = ordered_idx
        syn_labels = torch.tensor(label_list, dtype=torch.long, device=self.device)

        _log(f"computing priors for {len(ordered_idx)} samples")
        T_list, B_list = [], []
        for i in ordered_idx:
            priors = prior_init.get_priors_from_path(self.train_dataset.get_path(i))
            T_list.append(self._prep_prior(priors["T"], self.cfg.t_channels, spatial=True))
            B_list.append(self._prep_prior(priors["B"], channels=3, spatial=self.cfg.b_spatial))

        T01 = torch.cat(T_list, 0)
        B01 = torch.cat(B_list, 0)

        # everything is parameterized through sigmoid -> store in logit space,
        # even when frozen
        syn_T = self._make_param(torch.logit(T01), trainable=not self.cfg.freeze_T)
        syn_B = self._make_param(torch.logit(B01), trainable=not self.cfg.freeze_B)

        # save prior init for rmse
        self.T_prior = T01.detach().clone()
        self.B_prior = B01.detach().clone()
        _log(f"priors ready | T {tuple(syn_T.shape)} (frozen={self.cfg.freeze_T}) "
             f"B {tuple(syn_B.shape)} (frozen={self.cfg.freeze_B})")

        pyramid_J = self._init_pyramid()

        # free the SLURPP/PPG pipeline (several GB) once priors are computed:
        # only self.compose (a staticmethod) is kept afterwards
        del prior_init, sample_init
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        _log("synset init done")
        return pyramid_J, syn_T, syn_B, syn_labels

    def _init_pyramid(self) -> List[Tensor]:
        pyramid_J, res = [], 1
        while res <= self.cfg.pyramid_start_res:
            level = torch.randn((self.N, 3, res, res), device=self.device)
            if self.cfg.init_mode == "zero":
                level = level * 0
            pyramid_J.insert(0, level)
            res *= 2
            if res > self.cfg.syn_res:
                res = self.cfg.syn_res
        pyramid_J = [p / len(pyramid_J) for p in pyramid_J]
        for p in pyramid_J:
            p.requires_grad_(True)
        _log(f"pyramid initialized with {len(pyramid_J)} level(s), "
             f"start_res={self.cfg.pyramid_start_res}")
        return pyramid_J

    def init_optimizer(self):
        lr_T = getattr(self.cfg, "lr_T", self.cfg.lr)
        lr_B = getattr(self.cfg, "lr_B", self.cfg.lr)
        groups = [{"params": self.pyramid_J, "lr": self.cfg.lr}]
        if self.syn_T.requires_grad:                       # frozen -> excluded from optimizer
            groups.append({"params": [self.syn_T], "lr": lr_T})
        if self.syn_B.requires_grad:
            groups.append({"params": [self.syn_B], "lr": lr_B})
        return torch.optim.Adam(groups)

    # ----- forward -----

    def decode_J(self, n_levels: int = None) -> Tensor:
        levels = self.pyramid_J if n_levels is None else self.pyramid_J[-n_levels:]
        result = torch.sum(torch.stack([
            F.interpolate(p, (self.cfg.syn_res, self.cfg.syn_res),
                          antialias=False, mode="bilinear")
            for p in levels
        ]), dim=0)
        if self.cfg.decorrelate_color:
            result = self.linear_decorrelate_color(result)
        return torch.sigmoid(2 * result)

    def get_data(self) -> Tuple[Tensor, Tensor]:
        J = self.decode_J()
        T = torch.sigmoid(self.syn_T)
        B = torch.sigmoid(self.syn_B)
        I = self.compose(J, T, B)
        if getattr(self.cfg, "clamp_I", True):
            self._sat_frac = (I > 1.0).float().mean().item()   # log via wandb
            I = I.clamp(0.0, 1.0)
        return I, self.syn_labels

    # ----- research / save -----

    @torch.no_grad()
    def get_snapshot(self) -> dict:
        T = torch.sigmoid(self.syn_T)
        B = torch.sigmoid(self.syn_B)
        J_levels = [self.decode_J(n_levels=k) for k in range(1, len(self.pyramid_J) + 1)]
        J = J_levels[-1]
        I = self.compose(J, T, B)
        if getattr(self.cfg, "clamp_I", True):
            I = I.clamp(0.0, 1.0)
        return {
            "I": I.cpu(), "J": J.cpu(), "T": T.cpu(), "B": B.cpu(),
            "J_levels": [j.cpu() for j in J_levels],
            "level_res": [p.shape[-1] for p in reversed(self.pyramid_J)],
            "frozen": {"T": not self.syn_T.requires_grad, "B": not self.syn_B.requires_grad},
            "formation": self.formation_name,
        }

    @torch.no_grad()
    def get_to_save(self) -> dict:
        J = self.decode_J()
        T = torch.sigmoid(self.syn_T)
        B = torch.sigmoid(self.syn_B)
        I = self.compose(J, T, B)
        if getattr(self.cfg, "clamp_I", True):
            I = I.clamp(0.0, 1.0)

        sample_paths = [self.train_dataset.get_path(i) for i in self.sample_indices]

        return {
            "syn_data": (I, self.syn_labels),
            "save_dict": {
                "syn_J": J,
                "syn_T": T,
                "syn_B": B,
                "syn_T_logit": self.syn_T.detach(),
                "syn_B_logit": self.syn_B.detach(),
                "sample_indices": torch.tensor(self.sample_indices),
                "sample_paths": sample_paths,
                "sample_init": self.cfg.sample_init,
            },
        }

    # ----- pyramid upkeep -----

    def extend_pyramid(self) -> bool:
        old_len = len(self.pyramid_J)
        new_len = old_len + 1
        old_res = self.pyramid_J[0].shape[-1]
        if old_res == self.cfg.syn_res:
            _log("pyramid already at max resolution, no extension")
            return False
        new_res = min(old_res * 2, self.cfg.syn_res)
        num_images = self.pyramid_J[-1].shape[0]

        # rescale in-place -> keep object identity (= Adam state of existing J levels)
        with torch.no_grad():
            for p in self.pyramid_J:
                p.mul_(old_len / new_len)

            if self.cfg.init_mode == "zero":
                new_layer = torch.sum(torch.stack([
                    F.interpolate(p, (new_res, new_res), antialias=False, mode="bilinear")
                    for p in self.pyramid_J
                ]), dim=0) / old_len
            else:
                new_layer = torch.randn((num_images, 3, new_res, new_res),
                                        device=self.device) / new_len

        new_layer.requires_grad_(True)
        self.pyramid_J.insert(0, new_layer)

        # rebuild (canonical structure [J, T?, B?]) + re-inject state by object identity.
        # existing J levels + T + B keep their state; new_layer starts empty (intended).
        saved_state = dict(self.optimizer.state)
        self.optimizer = self.init_optimizer()
        self.optimizer.state.update(saved_state)
        _log(f"pyramid extended: {old_len} -> {new_len} levels (new res {new_res})")
        return True

    def upkeep(self, step: int = None):
        if (step - 1) % self.cfg.pyramid_extent_it == 0 and step > 1:
            if self.extend_pyramid():
                self.log_images(step=step)

    @torch.no_grad()
    def log_images(self, step: int = None):
        if len(self.pyramid_J[0]) > 100:
            print("Warning: too many images to log")
            return
        I, _ = self.get_data()
        log_images(syn_images=I.detach().clone(), step=step)

    def get_save_dict(self):
        return {
            "pyramid_J": self.pyramid_J,
            "T": self.syn_T, "B": self.syn_B,
            "opt_state": self.optimizer.state_dict(),
        }

    def load_from_dict(self, load_dict: dict):
        saved_groups = len(load_dict["opt_state"]["param_groups"])
        cur_groups = 1 + int(self.syn_T.requires_grad) + int(self.syn_B.requires_grad)
        assert saved_groups == cur_groups, (
            f"freeze flags inconsistent with checkpoint "
            f"({saved_groups} saved groups vs {cur_groups} current)"
        )
        while len(self.pyramid_J) < len(load_dict["pyramid_J"]):
            self.extend_pyramid()
        with torch.no_grad():
            for p, loaded_p in zip(self.pyramid_J, load_dict["pyramid_J"]):
                p.copy_(loaded_p)
            self.syn_T.copy_(load_dict["T"])
            self.syn_B.copy_(load_dict["B"])
        self.optimizer.load_state_dict(load_dict["opt_state"])
        _log(f"loaded from checkpoint ({len(self.pyramid_J)} pyramid levels)")


    # ------ Metrics ------

    @torch.no_grad()
    def physics_metrics(self) -> dict:
        """Quantitative monitoring of T/B vs their physical prior."""
        T = torch.sigmoid(self.syn_T)
        B = torch.sigmoid(self.syn_B)

        t_drift = torch.sqrt(F.mse_loss(T, self.T_prior))
        b_drift = torch.sqrt(F.mse_loss(B, self.B_prior))

        return {
            "physics/T_drift_rmse": t_drift.item(),
            "physics/B_drift_rmse": b_drift.item(),
            "physics/T_mean": T.mean().item(),
            "physics/T_std": T.std().item(),
            "physics/B_mean": B.mean().item(),
            "physics/B_std": B.std().item(),
            "physics/sat_frac": getattr(self, "_sat_frac", 0.0),
        }