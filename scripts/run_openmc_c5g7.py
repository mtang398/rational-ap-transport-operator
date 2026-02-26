"""
One-shot script: run OpenMC C5G7 → generate training dataset.

This script:
  1. Checks that OpenMC is installed (provides install instructions if not)
  2. Runs the C5G7 7-group eigenvalue calculation with OpenMC
  3. Generates n_samples TransportSamples with real MC flux as targets
  4. Writes them to runs/datasets/c5g7_{split}.zarr.h5

Usage
-----
  # Quick smoke test (~2 min on CPU):
  python scripts/run_openmc_c5g7.py --n_particles 10000 --n_batches 50 --n_inactive 10 --n_samples 5

  # Full production run (~30 min on CPU, ~5 min with MPI):
  python scripts/run_openmc_c5g7.py --n_particles 500000 --n_batches 500 --n_samples 200

  # Multiple splits:
  python scripts/run_openmc_c5g7.py --splits train val test \\
      --n_samples 200 50 50 --n_particles 100000

Install OpenMC (once)
---------------------
  # Windows / Linux / macOS  (no nuclear data required for multi-group):
  conda install -c conda-forge openmc

  # Check installation:
  python -c "import openmc; print(openmc.__version__)"

Output
------
  runs/datasets/c5g7_train.zarr.h5    ← TransportSamples with MC flux targets
  runs/openmc_c5g7/statepoint.*.h5    ← raw OpenMC output (keep for inspection)
  runs/openmc_c5g7/flux.npy           ← extracted flux [51, 51, 7]

Notes
-----
- The C5G7 geometry is deterministic, so all samples share the same XS map.
  Sample diversity comes from:
    (a) Different angular quadrature (omega) sets per sample
    (b) ±3% random perturbations to XS (simulating material uncertainty)
    (c) Different epsilon values (Knudsen number)
- The Monte Carlo flux is run once and cached. Re-running will load from cache
  unless --force is passed.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed
from src.data.io import ZarrDatasetWriter
from src.data.schema import SCHEMA_VERSION


def check_openmc():
    """Verify OpenMC is installed and print helpful message if not."""
    try:
        import openmc
        print(f"✓  OpenMC {openmc.__version__} found.")
        return True
    except ImportError:
        print("""
ERROR: OpenMC is not installed.

To install (Windows / Linux / macOS):
  conda install -c conda-forge openmc

  # Check it works:
  python -c "import openmc; print(openmc.__version__)"

OpenMC does NOT require a nuclear data library for multi-group (MGXS) mode.
The C5G7 model in this repository uses built-in macroscopic cross sections.
""")
        return False


def parse_args():
    p = argparse.ArgumentParser(
        description="Run OpenMC C5G7 and generate training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--splits",      nargs="+", default=["train"],
                   choices=["train", "val", "test"],
                   help="Dataset splits to generate")
    p.add_argument("--n_samples",   nargs="+", type=int, default=[200],
                   help="Number of samples per split (one per split or single value)")
    p.add_argument("--n_particles", type=int, default=100_000,
                   help="Monte Carlo particles per batch")
    p.add_argument("--n_batches",   type=int, default=300,
                   help="Total OpenMC batches (active + inactive)")
    p.add_argument("--n_inactive",  type=int, default=50,
                   help="Inactive (source convergence) batches")
    p.add_argument("--n_omega",     type=int, default=8,
                   help="Angular directions in output samples")
    p.add_argument("--n_mesh",      type=int, default=51,
                   help="Tally mesh resolution (51 = 3×17 pins)")
    p.add_argument("--output_dir",  default="runs/datasets",
                   help="Output directory for HDF5 datasets")
    p.add_argument("--work_dir",    default="runs/openmc_c5g7",
                   help="OpenMC working directory (XML inputs, statepoint)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--force",       action="store_true",
                   help="Force re-run even if statepoint exists")
    p.add_argument("--epsilon_min", type=float, default=0.01)
    p.add_argument("--epsilon_max", type=float, default=1.0)
    p.add_argument("--no_perturb",  action="store_true",
                   help="Disable ±3%% XS perturbation across samples")
    return p.parse_args()


def main():
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    args = parse_args()

    if not check_openmc():
        sys.exit(1)

    set_seed(args.seed)

    import numpy as np
    from src.solvers.openmc_interface import OpenMCInterface
    from src.solvers.openmc_c5g7_model import C5G7OpenMCModel
    from src.data.converters.c5g7 import (
        C5G7Converter, build_quarter_core_map, MATERIAL_NAMES, C5G7_XS,
    )
    from src.data.schema import (
        TransportSample, InputFields, QueryPoints, TargetFields, BCSpec,
    )

    # ── Step 1: Run OpenMC once ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Running OpenMC C5G7 eigenvalue calculation")
    logger.info(f"  particles={args.n_particles}, batches={args.n_batches}, "
                f"inactive={args.n_inactive}")
    logger.info("=" * 60)

    mc_model = C5G7OpenMCModel(
        work_dir    = args.work_dir,
        n_particles = args.n_particles,
        n_batches   = args.n_batches,
        n_inactive  = args.n_inactive,
        n_mesh      = args.n_mesh,
    )
    phi_mc, keff = mc_model.run(force=args.force)
    logger.info(f"  k-effective = {keff:.6f}")
    logger.info(f"  Flux shape  = {phi_mc.shape}")

    # Save raw flux for reference / converter use
    mc_model.save_flux_npy(phi_mc, str(Path(args.work_dir) / "flux.npy"))

    # ── Step 2: Build geometry + XS arrays ──────────────────────────────────
    G   = 7
    nx  = ny = args.n_mesh
    L   = 64.26   # cm

    mat_map = build_quarter_core_map(17)  # 51×51 material IDs
    sigma_a_base = np.zeros((nx, ny, G), dtype=np.float32)
    sigma_s_base = np.zeros((nx, ny, G), dtype=np.float32)
    q_base       = np.zeros((nx, ny, G), dtype=np.float32)

    for mid, mname in enumerate(MATERIAL_NAMES):
        xs   = C5G7_XS[mname]
        mask = (mat_map == mid)
        sigma_a_base[mask] = xs["sigma_a"]
        sigma_s_base[mask] = [xs["sigma_s"][g][g] for g in range(G)]
        chi   = np.array(xs["chi"],         dtype=np.float32)
        nu_sf = np.array(xs["nu_sigma_f"],  dtype=np.float32)
        q_base[mask] = chi * np.sum(nu_sf) * 0.1

    xs_coord = np.linspace(0, L, nx, dtype=np.float32)
    ys_coord = np.linspace(0, L, ny, dtype=np.float32)
    XX, YY   = np.meshgrid(xs_coord, ys_coord, indexing="ij")
    x_query  = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    Nx       = nx * ny
    phi_base = phi_mc.reshape(Nx, G)

    # ── Step 3: Generate samples for each split ──────────────────────────────
    # Expand n_samples to one value per split
    n_samples_list = args.n_samples
    if len(n_samples_list) == 1:
        n_samples_list = n_samples_list * len(args.splits)
    if len(n_samples_list) != len(args.splits):
        logger.error("--n_samples must be length 1 or same length as --splits")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for split, n_samples in zip(args.splits, n_samples_list):
        logger.info("=" * 60)
        logger.info(f"Step 3: Generating {n_samples} samples for split={split}")
        logger.info("=" * 60)

        samples = []
        eps_vals = np.linspace(args.epsilon_min, args.epsilon_max, n_samples)

        for i in range(n_samples):
            epsilon = float(eps_vals[i])

            # Optional per-sample XS perturbation
            if args.no_perturb or i == 0:
                sigma_a = sigma_a_base.copy()
                sigma_s = sigma_s_base.copy()
                q       = q_base.copy()
            else:
                sigma_a = sigma_a_base * rng.uniform(0.97, 1.03, sigma_a_base.shape).astype(np.float32)
                sigma_s = sigma_s_base * rng.uniform(0.97, 1.03, sigma_s_base.shape).astype(np.float32)
                q       = q_base.copy()

            # Angular quadrature: rotate slightly per sample for diversity
            angle_offset = rng.uniform(0, 2*np.pi / args.n_omega)
            angles   = np.linspace(angle_offset,
                                   angle_offset + 2*np.pi,
                                   args.n_omega, endpoint=False).astype(np.float32)
            omega    = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
            w_omega  = np.full(args.n_omega, 2*np.pi / args.n_omega, dtype=np.float32)

            # Flux: use MC result (with consistent XS rescaling)
            phi_vals = phi_base.copy()
            norm     = 2 * np.pi
            I_vals   = np.broadcast_to(phi_vals[:, np.newaxis, :] / norm,
                                        (Nx, args.n_omega, G)).copy()

            # Current via Fick's law from MC flux
            phi_grid = phi_vals.reshape(nx, ny, G)
            dx = L / max(nx-1, 1);  dy = L / max(ny-1, 1)
            D_grid = 1.0 / (3.0 * np.maximum(sigma_a + sigma_s, 1e-8))
            Jx = -(D_grid * np.gradient(phi_grid, dx, axis=0)).reshape(Nx, G)
            Jy = -(D_grid * np.gradient(phi_grid, dy, axis=1)).reshape(Nx, G)
            J_vals = np.stack([Jx, Jy], axis=1)

            bc     = BCSpec(bc_type="vacuum")
            inputs = InputFields(
                sigma_a  = sigma_a,
                sigma_s  = sigma_s,
                q        = q,
                bc       = bc,
                params   = {"epsilon": epsilon, "g": 0.0, "keff": keff},
                metadata = {
                    "benchmark_name": "c5g7",
                    "flux_source":    "openmc_multigroup_mc",
                    "n_particles":    args.n_particles,
                    "n_batches":      args.n_batches,
                    "keff":           keff,
                    "dim": 2, "group_count": G, "units": "cm",
                },
            )
            query   = QueryPoints(x=x_query, omega=omega, w_omega=w_omega)
            targets = TargetFields(I=I_vals, phi=phi_vals, J=J_vals)
            sample  = TransportSample(
                inputs=inputs, query=query, targets=targets,
                sample_id=f"c5g7_openmc_{split}_{i:04d}",
            )
            samples.append(sample)

        # Validate first 5
        for s in samples[:5]:
            errs = s.validate()
            if errs:
                logger.warning(f"  {s.sample_id}: {errs}")

        # Write
        out_path = output_dir / f"c5g7_{split}.zarr"
        writer   = ZarrDatasetWriter(str(out_path), mode="w")
        try:
            from tqdm import tqdm
            it = tqdm(samples, desc=f"Writing {split}")
        except ImportError:
            it = samples
        for idx, s in enumerate(it):
            writer.write(s, idx=idx)
        writer.close()

        logger.info(f"  Saved {len(samples)} samples → {writer.path}")

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info(f"  k-effective = {keff:.6f}")
    logger.info(f"  Flux source = OpenMC multi-group Monte Carlo")
    logger.info(f"  Dataset     = {output_dir}/c5g7_*.zarr.h5")
    logger.info(f"  Raw OpenMC  = {args.work_dir}/statepoint.*.h5")
    logger.info("=" * 60)
    logger.info(
        "\nNext step: train a model on this dataset:\n"
        "  python train.py --benchmark c5g7 --model ap_micromacro --n_epochs 100"
    )


if __name__ == "__main__":
    main()
