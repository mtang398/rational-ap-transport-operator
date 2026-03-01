"""
Dataset generation script.

Generates transport samples using solver (mock or real) and saves to Zarr/HDF5.

Standard splits (train / val / test) are used for training and SN-transfer eval.
Evaluation-specific splits are also generated here so that eval.py never needs
to run the solver again:

  resolution_x2 / resolution_x4
    Produced by bilinear interpolation of the test split to a 2× or 4× finer
    spatial grid.  NO new solver runs — the test samples are loaded from disk
    and their inputs (sigma_a, sigma_s, q) and targets (phi, I, J) are upsampled
    with scipy.ndimage.zoom.  The test split MUST be generated first.
    Used by ResolutionTransferProtocol.

  regime_eps<value>
    Fixed-epsilon samples at each value in the regime-sweep grid, e.g.
    regime_eps0.001, regime_eps0.010, …, regime_eps1.000.
    Used by RegimeSweepProtocol.

Usage:
  python scripts/generate_dataset.py --benchmark c5g7 --split train --n_samples 200
  python scripts/generate_dataset.py --benchmark c5g7 --split test  --n_samples 50
  python scripts/generate_dataset.py --benchmark c5g7 --split all_eval --n_samples 50
  # resolution_x2/x4 are generated from test — no separate n_samples needed:
  python scripts/generate_dataset.py --benchmark c5g7 --split resolution_x2
  python scripts/generate_dataset.py --benchmark c5g7 --split regime_eps0.010 --n_samples 50
"""
from __future__ import annotations
import argparse
import logging
import sys
import os
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed
from src.data.io import ZarrDatasetWriter
from src.data.schema import SCHEMA_VERSION

logger = logging.getLogger(__name__)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Generate transport dataset")
    p.add_argument("--benchmark", default="c5g7",
                   choices=["c5g7", "pinte2009"])
    p.add_argument("--solver", default="auto", choices=["auto", "mock", "openmc"],
                   help="'auto' picks the best available real solver for each benchmark, "
                        "falling back to mock if none is installed.")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument(
        "--split", default="train",
        help=(
            "Dataset split to generate. Standard: train, val, test. "
            "Eval splits (no solver re-runs during eval.py): "
            "resolution_x2, resolution_x4 (finer spatial grids); "
            "regime_eps<value> e.g. regime_eps0.010 (fixed epsilon); "
            "all_eval (generates test + all resolution + all regime splits at once)."
        ),
    )
    p.add_argument("--spatial_shape", type=int, nargs="+", default=None)
    p.add_argument("--n_omega", type=int, default=None)
    p.add_argument("--n_groups", type=int, default=None)
    p.add_argument("--output_dir", default="runs/datasets")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epsilon_min", type=float, default=0.01)
    p.add_argument("--epsilon_max", type=float, default=1.0)
    p.add_argument(
        "--xs_perturb", type=float, default=0.10,
        help="Half-width of uniform per-material XS perturbation (default 0.10 = ±10%%). "
             "Applied independently per material per energy group to sigma_a, sigma_s, "
             "and nu_sigma_f.  Set to 0 to disable (all samples share canonical XS).",
    )
    # ── OpenMC particle counts ──────────────────────────────────────────────
    # Particle count is kept at publication level (100 000/batch) so the
    # per-batch variance matches published C5G7 benchmarks exactly.
    # n_batches is reduced to 100 (50 inactive + 50 active) from the
    # publication standard of 300 (50 inactive + 250 active).  This gives
    # 5 M total active histories vs 25 M at publication quality — ~2.24×
    # larger flux uncertainty (~0.7 % vs ~0.3 %), which is negligible noise
    # for operator-learning targets while cutting wall time by ~5×.
    # n_inactive is kept at 50 (same as OECD/NEA C5G7 reference) so the
    # fission source is fully converged before tallying begins.
    p.add_argument("--n_particles", type=int, default=100_000,
                   help="MC particles per batch (default 100 000, same as publication).")
    p.add_argument("--n_batches", type=int, default=100,
                   help="Total MC batches: 50 inactive + 50 active (default 100). "
                        "Use 300 for full publication quality (50 inactive + 250 active).")
    p.add_argument("--n_inactive", type=int, default=50,
                   help="Inactive (source-convergence) batches — must match publication "
                        "value of 50 to ensure fission source convergence (default 50).")
    return p


# Epsilon values used by RegimeSweepProtocol — must match eval.py defaults.
REGIME_SWEEP_EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

# Resolution multipliers used by ResolutionTransferProtocol — must match eval.py defaults.
RESOLUTION_MULTIPLIERS = [2, 4]  # x1 is the test split itself

# Per-split seed offsets — large gaps so no overlap even for very large datasets.
# The base seed (--seed) anchors train; all other splits are offset from it.
# resolution_x2 / resolution_x4 are NOT listed here — they are produced by
# interpolation of the test split and require no separate seed.
SPLIT_SEED_OFFSETS = {
    "train":     0,
    "val":   10_000,
    "test":  20_000,
    # regime sweep: base offset + index (one per epsilon value)
    "regime": 50_000,
}

BENCHMARK_DEFAULTS = {
    "c5g7": {
        "spatial_shape": (51, 51), "n_omega": 8, "n_groups": 7,
        "converter_cls": "C5G7Converter",
    },
    "pinte2009": {
        "spatial_shape": (32, 32), "n_omega": 16, "n_groups": 1,
        "converter_cls": "Pinte2009Converter",
    },
}


def _upsample_sample(sample, multiplier: int):
    """
    Upsample a TransportSample to a finer spatial grid by bilinear interpolation.

    All spatially-gridded arrays (sigma_a, sigma_s, q, phi, I, J) are upsampled
    with scipy.ndimage.zoom using the given integer multiplier.  Scalar quantities
    (params, bc, metadata, sample_id) are carried over unchanged.

    This is the correct approach for resolution transfer: the physics problem is
    identical, only the query grid is finer.  Running the solver again at the
    finer grid would change the sample seed and produce a different problem.
    """
    from scipy.ndimage import zoom as _zoom
    from src.data.schema import TransportSample, InputFields, QueryPoints, TargetFields

    m = multiplier

    def _zoom_spatial(arr):
        """Zoom a [...spatial..., C] array along all but the last axis."""
        if arr is None:
            return None
        arr = np.array(arr, dtype=np.float32)
        ndim = arr.ndim
        # last axis is channel (groups / directions) — do not zoom it
        zoom_factors = (m,) * (ndim - 1) + (1,)
        return _zoom(arr, zoom_factors, order=1).astype(np.float32)

    inp = sample.inputs
    new_inputs = InputFields(
        sigma_a=_zoom_spatial(inp.sigma_a),
        sigma_s=_zoom_spatial(inp.sigma_s),
        q=_zoom_spatial(inp.q),
        extra_fields={k: _zoom_spatial(v) for k, v in (inp.extra_fields or {}).items()},
        bc=inp.bc,
        params=inp.params,
        metadata=dict(inp.metadata or {}),
    )

    # Rebuild query grid at the new resolution
    old_q = sample.query
    # infer spatial shape from the upsampled sigma_a
    new_spatial = new_inputs.sigma_a.shape[:-1]  # drop groups axis
    dim = len(new_spatial)
    # build a uniform grid over [0,1]^dim at the new resolution
    axes = [np.linspace(0.0, 1.0, s, dtype=np.float32) for s in new_spatial]
    grids = np.meshgrid(*axes, indexing="ij")
    x_new = np.stack([g.ravel() for g in grids], axis=-1).astype(np.float32)
    new_query = QueryPoints(
        x=x_new,
        omega=old_q.omega,
        w_omega=old_q.w_omega,
        t=old_q.t,
    )

    tgt = sample.targets
    old_spatial = sample.inputs.sigma_a.shape[:-1]  # e.g. (51, 51) or (20, 20, 20)

    # phi: stored as [Nx, G] (flat) — reshape to [*spatial, G], zoom, re-flatten
    new_phi = None
    if tgt.phi is not None:
        phi_arr = np.array(tgt.phi, dtype=np.float32)  # [Nx, G]
        G = phi_arr.shape[-1]
        phi_grid = phi_arr.reshape(old_spatial + (G,))   # [*spatial, G]
        phi_zoomed = _zoom_spatial(phi_grid)              # [*spatial*m, G]
        new_phi = phi_zoomed.reshape(-1, G)               # [Nx_new, G]

    # I: stored as [Nx, Nw, G] — reshape to [*spatial, Nw*G], zoom, re-flatten
    new_I = None
    if tgt.I is not None:
        I_arr = np.array(tgt.I, dtype=np.float32)  # [Nx, Nw, G]
        Nw = I_arr.shape[-2]
        G  = I_arr.shape[-1]
        I_grid = I_arr.reshape(old_spatial + (Nw * G,))
        I_zoomed = _zoom(I_grid, (m,) * dim + (1,), order=1).astype(np.float32)
        new_I = I_zoomed.reshape(-1, Nw, G)

    # J: stored as [Nx, d, G] — same pattern
    new_J = None
    if tgt.J is not None:
        J_arr = np.array(tgt.J, dtype=np.float32)  # [Nx, d, G]
        d = J_arr.shape[-2]
        G = J_arr.shape[-1]
        J_grid = J_arr.reshape(old_spatial + (d * G,))
        J_zoomed = _zoom(J_grid, (m,) * dim + (1,), order=1).astype(np.float32)
        new_J = J_zoomed.reshape(-1, d, G)

    new_targets = TargetFields(
        I=new_I,
        phi=new_phi,
        J=new_J,
        qois=tgt.qois,
    )

    new_metadata = dict(new_inputs.metadata or {})
    new_metadata["resolution_multiplier"] = m
    new_metadata["upsampled_from"] = sample.sample_id
    new_inputs.metadata = new_metadata

    return TransportSample(
        inputs=new_inputs,
        query=new_query,
        targets=new_targets,
        sample_id=f"{sample.sample_id}_x{m}",
    )


def _generate_resolution_split(
    benchmark: str,
    multiplier: int,
    output_dir: Path,
    n_samples: int,
):
    """
    Build a resolution_xN split by interpolating the test split.

    Loads <benchmark>_test.zarr from output_dir, upsamples every sample
    by `multiplier`, and writes <benchmark>_resolution_xN.zarr.

    The test split must already exist.
    """
    import numpy as np
    from src.data.io import ZarrDatasetWriter
    from src.data.schema import SCHEMA_VERSION

    test_path = output_dir / f"{benchmark}_test.zarr"
    if not test_path.exists():
        # also try .zarr.h5
        test_path_h5 = output_dir / f"{benchmark}_test.zarr.h5"
        if test_path_h5.exists():
            test_path = test_path_h5
        else:
            raise RuntimeError(
                f"Test split not found at {output_dir / f'{benchmark}_test.zarr'}.\n"
                f"Generate the test split first:\n"
                f"  python scripts/generate_dataset.py "
                f"--benchmark {benchmark} --split test --n_samples <N>"
            )

    from src.data.dataset import TransportDataset as _DS
    ds = _DS(source=test_path)
    n = min(n_samples, len(ds)) if n_samples > 0 else len(ds)
    logger.info(
        f"  [resolution_x{multiplier}] Interpolating {n} test samples "
        f"from {test_path.name} (x{multiplier} zoom)…"
    )

    out_path = output_dir / f"{benchmark}_resolution_x{multiplier}.zarr"
    writer = ZarrDatasetWriter(str(out_path), mode="w")
    try:
        from tqdm import tqdm
        indices = tqdm(range(n), desc=f"Upsampling x{multiplier}")
    except ImportError:
        indices = range(n)

    for i in indices:
        sample = ds.reader.read(i)
        up = _upsample_sample(sample, multiplier)
        writer.write(up, idx=i)

    writer.close()
    logger.info(f"  Done → {out_path}  ({n} samples, schema_version={SCHEMA_VERSION})")


def _generate_and_save(
    benchmark: str,
    split: str,
    n_samples: int,
    spatial_shape: tuple,
    n_omega: int,
    n_groups: int,
    solver: str,          # already-resolved solver name
    solver_kwargs: dict,
    output_dir: Path,
    seed: int,
    xs_perturb: float,
    epsilon_range: tuple,
    epsilon_fixed: float = None,  # if set, override epsilon_range with a point value
):
    """Generate samples for one split and write to disk."""
    import numpy as np
    from src.solvers import get_solver as _get_solver
    from src.data.converters import C5G7Converter, Pinte2009Converter

    defaults = BENCHMARK_DEFAULTS.get(benchmark, BENCHMARK_DEFAULTS["c5g7"])
    cls_name = defaults["converter_cls"]
    converter_map = {
        "C5G7Converter":    C5G7Converter,
        "Pinte2009Converter": Pinte2009Converter,
    }
    converter = converter_map[cls_name]()
    rng = np.random.default_rng(seed)

    # Build epsilon kwargs
    if epsilon_fixed is not None:
        # Tight band around the fixed value so each sample still has a distinct
        # (but nearly identical) epsilon — keeps the rng state advancing consistently.
        eps_kwargs = {"epsilon": epsilon_fixed}
    else:
        eps_kwargs = {"epsilon_range": epsilon_range}

    logger.info(
        f"  [{benchmark}/{split}] spatial_shape={spatial_shape}, "
        f"n_omega={n_omega}, n_samples={n_samples}, "
        f"epsilon={'fixed=' + str(epsilon_fixed) if epsilon_fixed else str(epsilon_range)}"
    )

    samples = converter.convert(
        n_samples=n_samples, spatial_shape=spatial_shape,
        n_omega=n_omega, rng=rng,
        xs_perturb=xs_perturb, **eps_kwargs,
    )

    slv = _get_solver(solver, **solver_kwargs)
    logger.info(f"  Running {solver} on {len(samples)} samples…")
    samples = slv.batch_solve(samples)

    # Spot-check schema
    errs_found = sum(bool(s.validate()) for s in samples[:5])
    if errs_found == 0:
        logger.info("  Schema validation passed.")
    else:
        logger.warning(f"  {errs_found} samples failed schema validation (first 5 checked).")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{benchmark}_{split}.zarr"
    logger.info(f"  Writing {len(samples)} samples → {out_path}")
    writer = ZarrDatasetWriter(str(out_path), mode="w")
    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc="Writing")
    except ImportError:
        iterator = samples
    for i, sample in enumerate(iterator):
        writer.write(sample, idx=i)
    writer.close()
    logger.info(f"  Done. schema_version={SCHEMA_VERSION}")


def main():
    setup_logging("INFO")
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    from src.solvers import detect_best_solver

    defaults = BENCHMARK_DEFAULTS.get(args.benchmark, BENCHMARK_DEFAULTS["c5g7"])
    base_shape = tuple(args.spatial_shape) if args.spatial_shape else defaults["spatial_shape"]
    n_omega    = args.n_omega   or defaults["n_omega"]
    n_groups   = args.n_groups  or defaults["n_groups"]
    eps_range  = (args.epsilon_min, args.epsilon_max)
    output_dir = Path(args.output_dir)

    # Resolve solver once
    resolved = detect_best_solver(args.benchmark) if args.solver == "auto" else args.solver
    logger.info(
        f"Benchmark={args.benchmark}  solver={resolved}"
        + ("  (auto-selected)" if args.solver == "auto" else "")
    )
    if resolved == "mock":
        logger.warning(
            f"No real solver for '{args.benchmark}' — using MockSolver. "
            "Targets are APPROXIMATE. Install OpenMC or OpenSn for real solutions."
        )

    solver_kwargs: dict = {"benchmark": args.benchmark, "fallback": True}
    if resolved == "openmc":
        solver_kwargs.update({
            "n_particles": args.n_particles,
            "n_batches":   args.n_batches,
            "n_inactive":  args.n_inactive,
        })
        logger.info(
            f"OpenMC: n_particles={args.n_particles}, "
            f"n_batches={args.n_batches}, n_inactive={args.n_inactive}"
        )

    common = dict(
        benchmark=args.benchmark, solver=resolved, solver_kwargs=solver_kwargs,
        output_dir=output_dir, xs_perturb=args.xs_perturb,
        epsilon_range=eps_range,
        n_omega=n_omega, n_groups=n_groups,
    )

    def _gen(split_name, n, seed, spatial_shape, epsilon_fixed=None):
        _generate_and_save(
            split=split_name, n_samples=n, spatial_shape=spatial_shape,
            seed=seed, epsilon_fixed=epsilon_fixed,
            **common,
        )

    split = args.split

    # ── all_eval: generate test + every resolution + every regime split ──────
    if split == "all_eval":
        logger.info("=== all_eval: generating all evaluation splits ===")

        # 1. test split — fresh solver run with independent seed
        _gen("test", args.n_samples,
             seed=args.seed + SPLIT_SEED_OFFSETS["test"],
             spatial_shape=base_shape)

        # 2. resolution splits — interpolated from test (no new solver runs)
        for mult in RESOLUTION_MULTIPLIERS:
            _generate_resolution_split(
                benchmark=args.benchmark,
                multiplier=mult,
                output_dir=output_dir,
                n_samples=args.n_samples,
            )

        # 3. regime sweep splits — fresh solver runs at fixed epsilon values
        for i_eps, eps in enumerate(REGIME_SWEEP_EPSILONS):
            eps_tag = f"{eps:.3f}".rstrip("0").rstrip(".")
            _gen(f"regime_eps{eps_tag}", args.n_samples,
                 seed=args.seed + SPLIT_SEED_OFFSETS["regime"] + i_eps,
                 spatial_shape=base_shape, epsilon_fixed=eps)

        logger.info("=== all_eval complete ===")
        return

    # ── resolution_xN split — interpolated from test, no solver ─────────────
    if split.startswith("resolution_x"):
        mult = int(split.split("resolution_x")[1])
        _generate_resolution_split(
            benchmark=args.benchmark,
            multiplier=mult,
            output_dir=output_dir,
            n_samples=args.n_samples,
        )
        return

    # ── regime_eps<value> split ──────────────────────────────────────────────
    if split.startswith("regime_eps"):
        eps_val = float(split.split("regime_eps")[1])
        i_eps = REGIME_SWEEP_EPSILONS.index(eps_val) if eps_val in REGIME_SWEEP_EPSILONS else 99
        _gen(split, args.n_samples,
             seed=args.seed + SPLIT_SEED_OFFSETS["regime"] + i_eps,
             spatial_shape=base_shape, epsilon_fixed=eps_val)
        return

    # ── standard train / val / test split ───────────────────────────────────
    if split not in SPLIT_SEED_OFFSETS:
        logger.warning(
            f"Unknown split '{split}' — not in SPLIT_SEED_OFFSETS.  "
            f"Using seed_offset=0 (same as 'train').  "
            f"Known splits: {list(SPLIT_SEED_OFFSETS.keys())}"
        )
    seed_offset = SPLIT_SEED_OFFSETS.get(split, 0)
    _gen(split, args.n_samples,
         seed=args.seed + seed_offset,
         spatial_shape=base_shape)


if __name__ == "__main__":
    main()
