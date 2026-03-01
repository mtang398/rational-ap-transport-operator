"""
C5G7 MOX benchmark converter script.

Converts raw C5G7 data (if present) or generates samples from published geometry.

Expected raw data location:
  data/raw/c5g7/geometry.json
  data/raw/c5g7/xs_7group.json
  data/raw/c5g7/solution_ref.npy (optional)

See src/data/converters/c5g7.py for expected file formats.

Usage:
  python scripts/convert_c5g7.py --raw_dir data/raw/c5g7 --output runs/datasets/c5g7_train.zarr
  python scripts/convert_c5g7.py --output runs/datasets/c5g7_train.zarr
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed
from src.data.converters.c5g7 import C5G7Converter
from src.data.io import ZarrDatasetWriter

logger = logging.getLogger(__name__)


def main():
    setup_logging("INFO")
    p = argparse.ArgumentParser(description="Convert C5G7 MOX benchmark data")
    p.add_argument("--raw_dir", default=None, help="Path to raw C5G7 data dir")
    p.add_argument("--output", default="runs/datasets/c5g7_train.zarr")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--spatial_shape", type=int, nargs="+", default=[17, 17])
    p.add_argument("--n_omega", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver", default="auto", choices=["auto", "mock", "openmc"],
                   help="'auto' uses the best available real solver, falling back to mock")
    args = p.parse_args()

    set_seed(args.seed)
    import numpy as np
    from src.solvers import get_solver, detect_best_solver

    logger.info("C5G7 MOX Converter")
    logger.info(f"  raw_dir: {args.raw_dir}")

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
        if not raw_dir.exists():
            logger.warning(f"raw_dir {raw_dir} does not exist. Proceeding without raw dir.")
            raw_dir = None
    else:
        raw_dir = None

    converter = C5G7Converter(raw_dir=raw_dir)

    if not converter._has_raw:
        logger.info("Raw data not found. Expected format:")
        print(C5G7Converter.expected_raw_format())

    rng = np.random.default_rng(args.seed)
    samples = converter.convert(
        n_samples=args.n_samples,
        spatial_shape=tuple(args.spatial_shape),
        n_omega=args.n_omega,
        rng=rng,
    )

    # Run real solver if available
    resolved_solver = detect_best_solver("c5g7") if args.solver == "auto" else args.solver
    logger.info(
        f"Solver: {resolved_solver}"
        + ("  (auto-selected)" if args.solver == "auto" else "")
    )
    if resolved_solver != "mock":
        solver = get_solver(resolved_solver, benchmark="c5g7", fallback=True)
        logger.info(f"Running {resolved_solver} solver on {len(samples)} samples…")
        samples = solver.batch_solve(samples)
    else:
        logger.warning("No real solver available for c5g7 — targets are diffusion approximation.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ZarrDatasetWriter(str(out_path), mode="w")
    for i, s in enumerate(samples):
        writer.write(s, idx=i)
    writer.flush(benchmark_name="c5g7_mox")
    logger.info(f"Saved {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
