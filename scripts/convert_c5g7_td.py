"""
C5G7-TD time-dependent benchmark converter script.

See src/data/converters/c5g7_td.py for expected raw file format.

Usage:
  python scripts/convert_c5g7_td.py --output runs/datasets/c5g7_td_train.zarr
"""
import argparse, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed
from src.data.converters.c5g7_td import C5G7TDConverter
from src.data.io import ZarrDatasetWriter

logger = logging.getLogger(__name__)


def main():
    setup_logging("INFO")
    p = argparse.ArgumentParser(description="Convert C5G7-TD benchmark data")
    p.add_argument("--raw_dir", default=None)
    p.add_argument("--output", default="runs/datasets/c5g7_td_train.zarr")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--spatial_shape", type=int, nargs="+", default=[17, 17])
    p.add_argument("--n_omega", type=int, default=16)
    p.add_argument("--n_time", type=int, default=10)
    p.add_argument("--t_end", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    set_seed(args.seed)
    import numpy as np
    rng = np.random.default_rng(args.seed)

    converter = C5G7TDConverter(raw_dir=args.raw_dir)
    samples = converter.convert(
        n_samples=args.n_samples,
        spatial_shape=tuple(args.spatial_shape),
        n_omega=args.n_omega,
        n_time=args.n_time,
        t_end=args.t_end,
        rng=rng,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ZarrDatasetWriter(str(out_path), mode="w")
    for i, s in enumerate(samples):
        writer.write(s, idx=i)
    writer.flush(benchmark_name="c5g7_td")
    logger.info(f"Saved {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
