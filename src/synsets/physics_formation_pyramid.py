from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from config import DistillCfg
from data.dataloaders import BaseRealDataset
from my_utils.device import DeviceSingleton
from my_utils.log_utils import log_images

from .base import BaseDistilledDataset

from src.initializer.priors import build_prior, compute_TB_priors
from src.initializer.sample import build_sample_init, ordered_indices

from src.my_utils.log_utils import log
from src.synsets.syn_J import build_syn_J




class PhysicsFormationDataset(BaseDistilledDataset):
    """
    Distilled underwater dataset under the Jaffe-McGlamery image-formation model.

        I = compose(J, T, B)

    J (scene radiance) is delegated to a JParameterization strategy (pixel / pyramid / …)
    selected by cfg.distill_mode. T (transmission) and B (backscatter/veiling light) are
    initialized from a physical prior (cfg.prior_init: none / ppg / slurpp) and optionally
    frozen (cfg.freeze_T / cfg.freeze_B).

    The exact composition is carried by the prior initializer's `compose`:
        PPG      : I = J·T + (1-T)·B
        SLURPP   : I = J·T + B
    """

    def __init__(self, train_dataset: BaseRealDataset, cfg: DistillCfg,
                 backbone: torch.nn.Module = None, num_feat: int = None):
        super().__init__()
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.device = DeviceSingleton.get()
        self.N = cfg.ipc * train_dataset.num_classes

        (self.syn_J, self.syn_T, self.syn_B,
         self.syn_labels) = self.init_synset(backbone, num_feat)
        self.optimizer = self.init_optimizer()

    # ----- init -----


    def _make_param(self, t: Tensor, trainable: bool) -> Tensor:
        return t.detach().clone().requires_grad_(trainable)

    def init_synset(self, backbone, num_feat):
        log(type(self).__name__, f"building sample init ({self.cfg.sample_init})")
        sample_init = build_sample_init(backbone, num_feat, self.cfg, self.train_dataset)

        log(type(self).__name__, f"building prior init ({self.cfg.prior_init})")
        prior_init = build_prior(self.cfg, self.device)
        self.compose, self.formation_name = prior_init.compose, prior_init.formation_name()

        log(type(self).__name__, "selecting samples")
        self.sample_indices, label_list = ordered_indices(sample_init.get_indices(), self.train_dataset, self.cfg.ipc)
        syn_labels = torch.tensor(label_list, dtype=torch.long, device=self.device)

        log(type(self).__name__, f"computing priors for {len(self.sample_indices)} samples")
        T01, B01 = compute_TB_priors(self.cfg, self.train_dataset, prior_init,
                             self.sample_indices, device=self.device)

        # everything is parameterized through sigmoid, store in logit space,
        syn_T = self._make_param(torch.logit(T01), trainable=not self.cfg.freeze_T)
        syn_B = self._make_param(torch.logit(B01), trainable=not self.cfg.freeze_B)

        # save prior init for physics drift metrics
        self.T_prior = T01.detach().clone()
        self.B_prior = B01.detach().clone()
        log(type(self).__name__, f"priors ready | T {tuple(syn_T.shape)} (frozen={self.cfg.freeze_T}) "
             f"B {tuple(syn_B.shape)} (frozen={self.cfg.freeze_B})")

        syn_J = build_syn_J(self.cfg, self.N, self.device)

        # free the prior pipeline (several GB) once priors are computed:
        # only self.compose (a staticmethod) is kept afterwards
        self._free_prior_pipeline(prior_init, sample_init)

        log(type(self).__name__, "synset init done")
        return syn_J, syn_T, syn_B, syn_labels


    def init_optimizer(self):
        lr_T = getattr(self.cfg, "lr_T", self.cfg.lr)
        lr_B = getattr(self.cfg, "lr_B", self.cfg.lr)
        groups = [{"params": self.syn_J.parameters(), "lr": self.cfg.lr}]
        if self.syn_T.requires_grad:                       # frozen -> excluded from optimizer
            groups.append({"params": [self.syn_T], "lr": lr_T})
        if self.syn_B.requires_grad:
            groups.append({"params": [self.syn_B], "lr": lr_B})
        return torch.optim.Adam(groups)

    # ----- forward -----


    def get_data(self) -> Tuple[Tensor, Tensor]:
        J = self.syn_J.decode()
        T = torch.sigmoid(self.syn_T)
        B = torch.sigmoid(self.syn_B)
        I = self.compose(J, T, B)
        if getattr(self.cfg, "clamp_I", True):
            self._sat_frac = (I > 1.0).float().mean().item()
            I = I.clamp(0.0, 1.0)
        return I, self.syn_labels

    # ----- research / save -----

    @torch.no_grad()
    def get_snapshot(self) -> dict:
        T = torch.sigmoid(self.syn_T)
        B = torch.sigmoid(self.syn_B)
        J_levels, level_res = self.syn_J.snapshot_levels()
        J = J_levels[-1]
        I = self.compose(J, T, B)
        if getattr(self.cfg, "clamp_I", True):
            I = I.clamp(0.0, 1.0)
        return {
            "I": I.cpu(), "J": J.cpu(), "T": T.cpu(), "B": B.cpu(),
            "J_levels": [j.cpu() for j in J_levels],
            "level_res": level_res,
            "frozen": {"T": not self.syn_T.requires_grad, "B": not self.syn_B.requires_grad},
            "formation": self.formation_name,
        }

    @torch.no_grad()
    def get_to_save(self) -> dict:
        J = self.syn_J.decode()
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
                "syn_Tlogit": self.syn_T.detach(),
                "syn_Blogit": self.syn_B.detach(),
                "sample_indices": torch.tensor(self.sample_indices),
                "sample_paths": sample_paths,
                "sample_init": self.cfg.sample_init,
            },
        }

    # ----- pyramid upkeep -----

    def _rebuild_optimizer_keeping_state(self):
        saved_state = dict(self.optimizer.state)
        self.optimizer = self.init_optimizer()
        self.optimizer.state.update(saved_state)

    def upkeep(self, step: int = None):
        if (step - 1) % self.cfg.pyramid_extent_it == 0 and step > 1:
            if self.syn_J.extend():
                self._rebuild_optimizer_keeping_state()
                self.log_images(step=step)

    @torch.no_grad()
    def log_images(self, step: int = None):
        if self.N > 100:
            print("Warning: too many images to log")
            return
        I, _ = self.get_data()
        log_images(syn_images=I.detach().clone(), step=step)

    def get_save_dict(self):
        return {
            "syn_J": self.syn_J.state_dict(),
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

        j_state = load_dict["syn_J"]
        self.syn_J.load_state_dict(j_state)

        with torch.no_grad():
            self.syn_T.copy_(load_dict["T"])
            self.syn_B.copy_(load_dict["B"])

        self.optimizer = self.init_optimizer()
        self.optimizer.load_state_dict(load_dict["opt_state"])
        log(type(self).__name__, "loaded from checkpoint")


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
    
    # ------ Priors ------

    def _free_prior_pipeline(self, prior_init, sample_init):
        """Free the prior initializer and sample initializer to save memory."""
        del prior_init, sample_init
        import gc
        gc.collect()
        torch.cuda.empty_cache()
