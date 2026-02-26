"""
Pinte et al. 2009 continuum radiative transfer benchmark converter script.

See src/data/converters/pinte2009.py for expected raw file format.

Usage:
  python scripts/convert_pinte2009.py --output runs/datasets/pinte2009_train.zarr
"""
import argparse, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed
from src.data.converters.pinte2009 import Pinte2009Converter
from src.data.io import ZarrDatasetWriter

logger = logging.getLogger(__name__)


def main():
    setup_logging("INFO")
    p = argparse.ArgumentParser(description="Convert Pinte 2009 RT benchmark data")
    p.add_argument("--raw_dir", default=None)
    p.add_argument("--output", default="runs/datasets/pinte2009_train.zarr")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--spatial_shape", type=int, nargs="+", default=[32, 32])
    p.add_argument("--n_omega", type=int, default=16)
    p.add_argument("--wavelength_um", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()

    set_seed(args.seed)
    import numpy as np
    rng = np.random.default_rng(args.seed)

    converter = Pinte2009Converter(raw_dir=args.raw_dir, wavelength_um=args.wavelength_um)
    samples = converter.convert(
        n_samples=args.n_samples,
        spatial_shape=tuple(args.spatial_shape),
        n_omega=args.n_omega,
        rng=rng,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ZarrDatasetWriter(str(out_path), mode="w")
    for i, s in enumerate(samples):
        writer.write(s, idx=i)
    writer.flush(benchmark_name="pinte2009_rt")
    logger.info(f"Saved {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
