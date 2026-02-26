"""
Dataset generation script.

Generates transport samples using solver (mock or real) and saves to Zarr/HDF5.

Usage:
  python scripts/generate_dataset.py --benchmark c5g7 --solver mock --n_samples 200
  python scripts/generate_dataset.py --benchmark kobayashi --solver mock --n_samples 100
  python scripts/generate_dataset.py --benchmark pinte2009 --solver mock --n_samples 150
"""
from __future__ import annotations
import argparse
import logging
import sys
import os
from pathlib import Path

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
                   choices=["c5g7", "c5g7_td", "kobayashi", "pinte2009"])
    p.add_argument("--solver", default="mock", choices=["mock", "opensn", "openmc"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--spatial_shape", type=int, nargs="+", default=None)
    p.add_argument("--n_omega", type=int, default=None)
    p.add_argument("--n_groups", type=int, default=None)
    p.add_argument("--output_dir", default="runs/datasets")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epsilon_min", type=float, default=0.01)
    p.add_argument("--epsilon_max", type=float, default=1.0)
    p.add_argument("--n_time", type=int, default=10)
    return p


BENCHMARK_DEFAULTS = {
    "c5g7": {
        "spatial_shape": (51, 51), "n_omega": 8, "n_groups": 7,
        "converter_cls": "C5G7Converter",
    },
    "c5g7_td": {
        "spatial_shape": (51, 51), "n_omega": 8, "n_groups": 7,
        "converter_cls": "C5G7TDConverter",
    },
    "kobayashi": {
        "spatial_shape": (20, 20, 20), "n_omega": 24, "n_groups": 1,
        "converter_cls": "KobayashiConverter",
    },
    "pinte2009": {
        "spatial_shape": (32, 32), "n_omega": 16, "n_groups": 1,
        "converter_cls": "Pinte2009Converter",
    },
}


def main():
    setup_logging("INFO")
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    import numpy as np
    from src.solvers import get_solver
    from src.data.converters import C5G7Converter, C5G7TDConverter, KobayashiConverter, Pinte2009Converter

    defaults = BENCHMARK_DEFAULTS.get(args.benchmark, BENCHMARK_DEFAULTS["c5g7"])
    spatial_shape = tuple(args.spatial_shape) if args.spatial_shape else defaults["spatial_shape"]
    n_omega = args.n_omega or defaults["n_omega"]
    n_groups = args.n_groups or defaults["n_groups"]

    logger.info(f"Generating {args.n_samples} samples for benchmark={args.benchmark}")
    logger.info(f"  spatial_shape={spatial_shape}, n_omega={n_omega}, n_groups={n_groups}")

    rng = np.random.default_rng(args.seed)

    # Pick converter
    cls_name = defaults["converter_cls"]
    converter_map = {
        "C5G7Converter": C5G7Converter,
        "C5G7TDConverter": C5G7TDConverter,
        "KobayashiConverter": KobayashiConverter,
        "Pinte2009Converter": Pinte2009Converter,
    }
    ConverterCls = converter_map[cls_name]
    converter = ConverterCls()

    # Generate samples
    if cls_name == "C5G7TDConverter":
        samples = converter.convert(
            n_samples=args.n_samples,
            spatial_shape=spatial_shape,
            n_omega=n_omega,
            n_time=args.n_time,
            rng=rng,
        )
    else:
        samples = converter.convert(
            n_samples=args.n_samples,
            spatial_shape=spatial_shape,
            n_omega=n_omega,
            rng=rng,
        )

    # Optionally run through solver
    if args.solver != "mock":
        solver = get_solver(args.solver, fallback=True)
        logger.info(f"Running solver: {args.solver}")
        samples = solver.batch_solve(samples)

    # Validate samples
    errors_found = 0
    for s in samples[:5]:  # check first 5
        errs = s.validate()
        if errs:
            logger.warning(f"Sample {s.sample_id} validation errors: {errs}")
            errors_found += 1
    if errors_found == 0:
        logger.info("Schema validation passed for spot-checked samples.")

    # Save to Zarr
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.benchmark}_{args.split}.zarr"

    logger.info(f"Writing {len(samples)} samples to {out_path}")
    writer = ZarrDatasetWriter(str(out_path), mode="w")
    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc="Writing")
    except ImportError:
        iterator = samples
    for i, sample in enumerate(iterator):
        writer.write(sample, idx=i)
    writer.close()   # flush + close (required for h5py)

    logger.info(f"Done. Dataset saved to {writer.path}")
    logger.info(f"  n_samples={len(samples)}, schema_version={SCHEMA_VERSION}")


if __name__ == "__main__":
    main()
