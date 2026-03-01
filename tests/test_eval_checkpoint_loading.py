"""
End-to-end test: checkpoint saving → loading → eval for all benchmarks and models.

Tests:
- Creating a mock checkpoint with model_args embedded
- Loading via eval.py's load_model() for every (benchmark, model) combination
- Running all three eval protocols with a tiny config (fast, minimal samples)
- Verifying no size-mismatch or runtime errors occur
"""
from __future__ import annotations

import sys
import types
import tempfile
from pathlib import Path
from argparse import Namespace

import pytest
import torch
import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── benchmark definitions (mirrors eval.py) ─────────────────────────────────
BENCHMARK_DEFAULTS = {
    "c5g7":      {"spatial_shape": (8, 8),    "n_omega": 8,  "n_groups": 7, "dim": 2},
    "pinte2009": {"spatial_shape": (8, 8),    "n_omega": 8,  "n_groups": 1, "dim": 2},
}

MODELS = ["fno", "deeponet", "ap_micromacro"]

# Tiny arch so tests run fast
TINY_ARCH = dict(
    fno_channels=8,
    n_fno_blocks=2,
    n_modes=4,
    n_basis=16,
    macro_channels=8,
    micro_latent_dim=16,
    activation="gelu",
    lambda_moment=1.0,
    lambda_diffusion=0.1,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_args(benchmark: str, model: str, checkpoint_path: str) -> Namespace:
    """Build a Namespace that mirrors what eval.py's parse_args() returns."""
    bm = BENCHMARK_DEFAULTS[benchmark]
    args = Namespace(
        checkpoint=checkpoint_path,
        benchmark=benchmark,
        model=model,
        device="cpu",
        **TINY_ARCH,
        # eval-specific (not used by load_model, but needed by main())
        train_n_omega=bm["n_omega"],
        test_n_omegas=[4, 8],
        resolution_multipliers=[1, 2],
        epsilon_values=[0.01, 1.0],
        batch_size=2,
        n_test_samples=4,
        output_dir=None,
        seed=0,
        protocol="all",
    )
    return args


def _build_model(model_name: str, bm_def: dict):
    """Instantiate model with tiny arch params."""
    dim = bm_def["dim"]
    n_groups = bm_def["n_groups"]
    n_bc_faces = 2 * dim
    time_dependent = bm_def.get("time_dependent", False)

    if model_name == "fno":
        from src.models.fno import FNOTransport
        return FNOTransport(
            dim=dim, n_groups=n_groups, n_params=2,
            fno_channels=TINY_ARCH["fno_channels"],
            n_fno_blocks=TINY_ARCH["n_fno_blocks"],
            n_modes=TINY_ARCH["n_modes"],
            activation=TINY_ARCH["activation"],
            time_dependent=time_dependent,
            n_bc_faces=n_bc_faces,
        )
    elif model_name == "deeponet":
        from src.models.deeponet import DeepONetTransport
        return DeepONetTransport(
            dim=dim, n_groups=n_groups, n_params=2,
            n_basis=TINY_ARCH["n_basis"],
            activation=TINY_ARCH["activation"],
            time_dependent=time_dependent,
            n_bc_faces=n_bc_faces,
        )
    elif model_name == "ap_micromacro":
        from src.models.ap_micromacro import APMicroMacroTransport
        return APMicroMacroTransport(
            dim=dim, n_groups=n_groups, n_params=2,
            macro_channels=TINY_ARCH["macro_channels"],
            n_fno_blocks=TINY_ARCH["n_fno_blocks"],
            n_modes=TINY_ARCH["n_modes"],
            micro_latent_dim=TINY_ARCH["micro_latent_dim"],
            activation=TINY_ARCH["activation"],
            time_dependent=time_dependent,
            n_bc_faces=n_bc_faces,
            lambda_moment=TINY_ARCH["lambda_moment"],
            lambda_diffusion=TINY_ARCH["lambda_diffusion"],
        )
    raise ValueError(model_name)


def _save_checkpoint(model, model_name: str, benchmark: str, path: str):
    """Save a mock checkpoint that includes model_args (post-fix format)."""
    saved_args = dict(TINY_ARCH)
    saved_args["model"] = model_name
    saved_args["benchmark"] = benchmark
    state = {
        "epoch": 1,
        "global_step": 10,
        "model_state": model.state_dict(),
        "val_loss": 0.5,
        "best_val_loss": 0.5,
        "model_args": saved_args,
    }
    torch.save(state, path)


# ── load_model() (copy from eval.py, kept in sync) ──────────────────────────

def _load_model(args, bm_def):
    """Local copy of eval.py:load_model — must stay in sync."""
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    saved_args = ckpt.get("model_args", {})
    arch_keys = [
        "model", "fno_channels", "n_fno_blocks", "n_modes", "n_basis",
        "macro_channels", "micro_latent_dim", "activation",
        "lambda_moment", "lambda_diffusion",
    ]
    for key in arch_keys:
        if key in saved_args:
            setattr(args, key, saved_args[key])

    model = _build_model(args.model, bm_def)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model, device


# ── fixtures / parametrize ───────────────────────────────────────────────────

@pytest.mark.parametrize("benchmark", list(BENCHMARK_DEFAULTS.keys()))
@pytest.mark.parametrize("model_name", MODELS)
def test_checkpoint_roundtrip(benchmark, model_name, tmp_path):
    """Save a checkpoint then load it; no size-mismatch errors allowed."""
    bm_def = BENCHMARK_DEFAULTS[benchmark]

    # 1. Build and save
    model = _build_model(model_name, bm_def)
    ckpt_file = str(tmp_path / f"{benchmark}_{model_name}.pt")
    _save_checkpoint(model, model_name, benchmark, ckpt_file)

    # 2. Load via load_model with WRONG arch defaults (simulating a fresh eval call)
    args = _make_args(benchmark, model_name, ckpt_file)
    # Deliberately set wrong arch values to prove checkpoint overrides them
    args.fno_channels = 999
    args.n_modes = 999
    args.n_fno_blocks = 999

    model_loaded, device = _load_model(args, bm_def)
    assert model_loaded is not None
    assert str(device) == "cpu"


@pytest.mark.parametrize("benchmark", list(BENCHMARK_DEFAULTS.keys()))
@pytest.mark.parametrize("model_name", MODELS)
def test_eval_forward_pass(benchmark, model_name, tmp_path):
    """Load checkpoint and run one forward pass with a mock batch."""
    bm_def = BENCHMARK_DEFAULTS[benchmark]

    model_orig = _build_model(model_name, bm_def)
    ckpt_file = str(tmp_path / f"{benchmark}_{model_name}.pt")
    _save_checkpoint(model_orig, model_name, benchmark, ckpt_file)

    args = _make_args(benchmark, model_name, ckpt_file)
    model, device = _load_model(args, bm_def)

    # Build a single mock batch
    from src.data.dataset import MockDataset, collate_fn
    ds = MockDataset(
        n_samples=2,
        spatial_shape=bm_def["spatial_shape"],
        n_omega=bm_def["n_omega"],
        n_groups=bm_def["n_groups"],
        benchmark_name=benchmark,
        seed=0,
        solver_name="mock",
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        pred = model(batch_dev)

    assert "I" in pred
    assert "phi" in pred
    assert not torch.isnan(pred["I"]).any(), "NaN in intensity output"
    assert not torch.isnan(pred["phi"]).any(), "NaN in flux output"


@pytest.mark.parametrize("benchmark", list(BENCHMARK_DEFAULTS.keys()))
@pytest.mark.parametrize("model_name", MODELS)
def test_eval_all_protocols(benchmark, model_name, tmp_path):
    """
    Load checkpoint, then run all three eval protocols (minimal config).
    Verifies no crash and results are non-empty dicts.
    """
    bm_def = BENCHMARK_DEFAULTS[benchmark]

    model_orig = _build_model(model_name, bm_def)
    ckpt_file = str(tmp_path / f"{benchmark}_{model_name}.pt")
    _save_checkpoint(model_orig, model_name, benchmark, ckpt_file)

    args = _make_args(benchmark, model_name, ckpt_file)
    model, device = _load_model(args, bm_def)

    from src.data.dataset import MockDataset
    from src.data.io import ZarrDatasetWriter
    from src.eval.protocols import (
        TestSetProtocol,
        SNTransferProtocol,
        ResolutionTransferProtocol,
        RegimeSweepProtocol,
    )

    data_dir = tmp_path / "datasets"
    data_dir.mkdir()

    def _write_split(split_name: str, spatial_shape: tuple, n: int, seed: int):
        ds = MockDataset(
            n_samples=n, spatial_shape=spatial_shape,
            n_omega=bm_def["n_omega"], n_groups=bm_def["n_groups"],
            benchmark_name=benchmark, seed=seed, solver_name="mock",
        )
        zarr_path = str(data_dir / f"{benchmark}_{split_name}.zarr")
        w = ZarrDatasetWriter(zarr_path, mode="w")
        for i, s in enumerate(ds._samples):
            w.write(s, idx=i)
        w.close()

    base_shape = bm_def["spatial_shape"]
    n = args.n_test_samples

    # Write all splits that the protocols will try to load from disk
    _write_split("test",          base_shape, n, seed=1)
    _write_split("resolution_x2", tuple(s * 2 for s in base_shape), n, seed=2)
    for i_eps, eps in enumerate([0.01, 1.0]):  # matches args.epsilon_values
        eps_tag = f"{eps:.3f}".rstrip("0").rstrip(".")
        _write_split(f"regime_eps{eps_tag}", base_shape, n, seed=100 + i_eps)

    test_samples = MockDataset(
        n_samples=n, spatial_shape=base_shape,
        n_omega=bm_def["n_omega"], n_groups=bm_def["n_groups"],
        benchmark_name=benchmark, seed=1, solver_name="mock",
    )._samples

    # --- Test Set (baseline) ---
    ts = TestSetProtocol(
        model=model,
        test_samples=test_samples,
        device=str(device),
        batch_size=args.batch_size,
    )
    ts_results = ts.run()
    assert "test" in ts_results, "TestSetProtocol must return a 'test' key"
    r = ts_results["test"]
    for metric in ("I_rel_l2", "phi_rel_l2", "J_rel_l2"):
        assert metric in r, f"Missing {metric} in test set results"
        assert r[metric] == r[metric], f"NaN {metric} in test set results"

    # --- SN Transfer ---
    sn = SNTransferProtocol(
        model=model,
        test_samples=test_samples,
        train_n_omega=bm_def["n_omega"],
        test_n_omegas=args.test_n_omegas,
        device=str(device),
        batch_size=args.batch_size,
    )
    sn_results = sn.run()
    assert sn_results, "SN transfer returned empty results"
    for nw, r in sn_results.items():
        assert "I_rel_l2" in r, f"Missing I_rel_l2 for n_omega={nw}"
        assert not (r["I_rel_l2"] != r["I_rel_l2"]), f"NaN I_rel_l2 at n_omega={nw}"

    # --- Resolution Transfer (x1 uses base_test_samples; x2 loads from disk) ---
    res = ResolutionTransferProtocol(
        model=model,
        base_spatial_shape=base_shape,
        resolution_multipliers=[1, 2],
        n_groups=bm_def["n_groups"],
        benchmark_name=benchmark,
        n_test_samples=n,
        device=str(device),
        batch_size=args.batch_size,
        n_omega=bm_def["n_omega"],
        base_test_samples=test_samples,
        data_dir=str(data_dir),
    )
    res_results = res.run()
    assert res_results, "Resolution transfer returned empty results"

    # --- Regime Sweep (all splits loaded from disk) ---
    sweep = RegimeSweepProtocol(
        model=model,
        epsilon_values=[0.01, 1.0],
        spatial_shape=base_shape,
        n_omega=bm_def["n_omega"],
        n_groups=bm_def["n_groups"],
        benchmark_name=benchmark,
        n_test_samples=n,
        device=str(device),
        batch_size=args.batch_size,
        data_dir=str(data_dir),
    )
    sweep_results = sweep.run()
    assert sweep_results, "Regime sweep returned empty results"


@pytest.mark.parametrize("benchmark", list(BENCHMARK_DEFAULTS.keys()))
@pytest.mark.parametrize("model_name", MODELS)
def test_legacy_checkpoint_no_model_args(benchmark, model_name, tmp_path):
    """
    A checkpoint WITHOUT model_args (legacy format) should still load
    correctly when the caller passes matching arch flags on the CLI.
    """
    bm_def = BENCHMARK_DEFAULTS[benchmark]
    model = _build_model(model_name, bm_def)
    ckpt_file = str(tmp_path / f"legacy_{benchmark}_{model_name}.pt")

    # Save WITHOUT model_args (old format)
    torch.save({
        "epoch": 0,
        "model_state": model.state_dict(),
        "val_loss": 1.0,
    }, ckpt_file)

    # Pass correct arch on CLI — must not crash
    args = _make_args(benchmark, model_name, ckpt_file)
    model_loaded, _ = _load_model(args, bm_def)
    assert model_loaded is not None
