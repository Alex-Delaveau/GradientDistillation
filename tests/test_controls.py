"""CPU checks for the camera-ready control variants (scripts/slurm/controls).

Run (local, CPU only):
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src:. uv run --with pytest pytest -q tests/test_controls.py

The real PhysicsFormationDataset is built; only the medoid sampler (needs a
backbone + the dataset) and, for SLURPP, the diffusion network are mocked.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

import synsets.physics_formation_pyramid as pfp
from config import DistillCfg
from src.initializer.priors import (AffinePriorInitializer, NonePriorInitializer,
                                    SLURPPInitializer, build_prior)

NUM_CLASSES, IPC, RES = 20, 1, 224
N = NUM_CLASSES * IPC

# exact flag sets of scripts/slurm/controls/distill_controls.slurm (+ paper "none")
COMMON = ["--dataset=aqua20", "--model=dinov2_vitb", "--formation_mode=physics",
          "--distill_mode=pyramid", "--sample_init=medoids", f"--syn_res={RES}",
          f"--ipc={IPC}", "--b_spatial=False", "--clamp_I=True", "--seed=3407"]
VARIANTS = {
    "none":        ["--prior_init=none", "--t_channels=1", "--freeze_T=False", "--freeze_B=False"],
    "none_frozen": ["--prior_init=none", "--t_channels=1", "--freeze_T=True", "--freeze_B=True"],
    "affine":      ["--prior_init=affine", "--t_channels=1", "--freeze_T=False", "--freeze_B=False"],
    "slurpp_t1":   ["--prior_init=slurpp", "--t_channels=1", "--freeze_T=False", "--freeze_B=False"],
    "none_t3":     ["--prior_init=none", "--t_channels=3", "--freeze_T=False", "--freeze_B=False"],
}
# variant -> (T shape, B shape, T trainable, B trainable)
EXPECTED = {
    "none":        ((N, 1, RES, RES), (N, 3, 1, 1), True, True),
    "none_frozen": ((N, 1, RES, RES), (N, 3, 1, 1), False, False),
    "affine":      ((N, 1, RES, RES), (N, 3, 1, 1), True, True),
    "slurpp_t1":   ((N, 1, RES, RES), (N, 3, 1, 1), True, True),
    "none_t3":     ((N, 3, RES, RES), (N, 3, 1, 1), True, True),
}


def make_cfg(variant):
    return DistillCfg(explicit_bool=True).parse_args(COMMON + VARIANTS[variant])


class _FakeTrainSet:
    num_classes = NUM_CLASSES

    def get_path(self, i):
        return f"/fake/{i}.png"


class _FakeSampleInit:
    def get_indices(self):
        return {c: [c] for c in range(NUM_CLASSES)}


class _FakeSLURPP:
    """Stands in for the SLURPP network: native 512x512, 3-channel T and B."""
    compose = staticmethod(SLURPPInitializer.compose)

    def formation_name(self):
        return "SLURPPInitializer"

    def get_priors_from_path(self, path):
        g = torch.Generator().manual_seed(hash(path) % 2**31)
        return {"T": torch.rand(1, 3, 512, 512, generator=g),
                "B": torch.rand(1, 3, 512, 512, generator=g)}


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(pfp, "build_sample_init", lambda *a, **k: _FakeSampleInit())
    real_build_prior = pfp.build_prior

    def fake_build_prior(cfg, device):
        if cfg.prior_init == "slurpp":
            return _FakeSLURPP()
        return real_build_prior(cfg, device)
    monkeypatch.setattr(pfp, "build_prior", fake_build_prior)


def build(variant, seed=3407):
    torch.manual_seed(seed)
    return pfp.PhysicsFormationDataset(_FakeTrainSet(), make_cfg(variant))


def test_runs_on_cpu():
    assert str(pfp.DeviceSingleton.get()) == "cpu"


def test_build_prior_dispatch():
    cfg = make_cfg("affine")
    assert isinstance(build_prior(cfg, "cpu"), AffinePriorInitializer)
    assert isinstance(build_prior(make_cfg("none"), "cpu"), NonePriorInitializer)


def test_affine_compose_matches_none_at_init_standalone():
    cfg = make_cfg("affine")
    a, n = AffinePriorInitializer(cfg), NonePriorInitializer(cfg)
    pa, pn = a.get_priors(None), n.get_priors(None)
    assert pa["T"].shape == pn["T"].shape == (1, 1, RES, RES)
    assert pa["B"].shape == pn["B"].shape == (1, 3, 1, 1)
    J = torch.rand(4, 3, RES, RES)
    Ia, In = a.compose(J, pa["T"], pa["B"]), n.compose(J, pn["T"], pn["B"])
    # (1 - 0.5) * 0.5 == 0.25 exactly in float32 -> bitwise identical
    assert torch.equal(Ia, In)
    assert torch.equal(Ia, 0.5 * J + 0.25)


def test_affine_same_initial_image_as_none_full_pipeline():
    """Same seed -> same J; through logit/sigmoid storage, I must coincide."""
    ds_none, ds_aff = build("none"), build("affine")
    for p, q in zip(ds_none.syn_J.parameters(), ds_aff.syn_J.parameters()):
        assert torch.equal(p, q)
    I_none, y_none = ds_none.get_data()
    I_aff, y_aff = ds_aff.get_data()
    assert torch.equal(y_none, y_aff)
    assert (I_none - I_aff).abs().max().item() < 1e-6


def test_affine_gradients_differ_from_none():
    ds_none, ds_aff = build("none"), build("affine")
    gs = []
    for ds in (ds_none, ds_aff):
        I, _ = ds.get_data()
        (I * torch.linspace(0, 1, I.numel()).view_as(I)).sum().backward()
        gs.append((ds.syn_T.grad.clone(), ds.syn_B.grad.clone()))
    assert not torch.allclose(gs[0][0], gs[1][0])
    assert not torch.allclose(gs[0][1], gs[1][1])


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_shapes_and_optimizer_groups(variant):
    ds = build(variant)
    t_shape, b_shape, t_train, b_train = EXPECTED[variant]
    assert tuple(ds.syn_T.shape) == t_shape
    assert tuple(ds.syn_B.shape) == b_shape
    assert ds.syn_T.requires_grad is t_train
    assert ds.syn_B.requires_grad is b_train

    opt_params = {id(p) for g in ds.optimizer.param_groups for p in g["params"]}
    assert (id(ds.syn_T) in opt_params) is t_train
    assert (id(ds.syn_B) in opt_params) is b_train
    assert len(ds.optimizer.param_groups) == 1 + int(t_train) + int(b_train)
    for p in ds.syn_J.parameters():
        assert id(p) in opt_params

    I, y = ds.get_data()
    assert tuple(I.shape) == (N, 3, RES, RES) and len(y) == N


def test_none_frozen_is_fixed_compression():
    """I = 0.5 J + 0.25 at init, and T/B unchanged after an optimizer step."""
    ds = build("none_frozen")
    T0, B0 = ds.syn_T.detach().clone(), ds.syn_B.detach().clone()
    J = ds.syn_J.decode()
    I, _ = ds.get_data()
    assert (I - (0.5 * J + 0.25).clamp(0, 1)).abs().max().item() < 1e-6
    ds.optimizer.zero_grad()
    I.sum().backward()
    ds.optimizer.step()
    assert torch.equal(ds.syn_T, T0) and torch.equal(ds.syn_B, B0)
    assert ds.syn_T.grad is None and ds.syn_B.grad is None


def test_checkpoint_roundtrip_frozen():
    """Resume path used by distill_controls.slurm: save/load with frozen T/B."""
    ds = build("none_frozen")
    sd = ds.get_save_dict()
    ds2 = build("none_frozen", seed=1)
    ds2.load_from_dict(sd)
    assert torch.allclose(ds.get_data()[0], ds2.get_data()[0])
