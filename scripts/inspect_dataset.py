"""
Dataset inspection script.

Prints schema version, stats/ranges, verifies masks/shapes, plots slices.

Usage:
  python scripts/inspect_dataset.py runs/datasets/c5g7_train.zarr
  python scripts/inspect_dataset.py runs/datasets/c5g7_train.zarr --plot
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logging
from src.data.io import ZarrDatasetReader

logger = logging.getLogger(__name__)


def print_array_stats(name: str, arr):
    import numpy as np
    a = np.array(arr)
    print(f"  {name:20s}: shape={str(a.shape):20s}  "
          f"min={a.min():.4e}  max={a.max():.4e}  mean={a.mean():.4e}  "
          f"dtype={a.dtype}")


def main():
    setup_logging("INFO")
    p = argparse.ArgumentParser(description="Inspect a transport dataset")
    p.add_argument("dataset_path", help="Path to Zarr or HDF5 dataset")
    p.add_argument("--n_show", type=int, default=3, help="Number of samples to inspect")
    p.add_argument("--plot", action="store_true", help="Plot sample slices")
    p.add_argument("--validate", action="store_true", default=True, help="Validate sample schemas")
    args = p.parse_args()

    path = Path(args.dataset_path)
    # On Windows, zarr v3 writes .zarr.zip; accept both spellings
    if not path.exists():
        zip_path = Path(str(path) + ".zip")
        if zip_path.exists():
            path = zip_path
        else:
            logger.error(f"Dataset not found: {path} (also tried {zip_path})")
            sys.exit(1)

    logger.info(f"Inspecting dataset: {path}")

    reader = ZarrDatasetReader(path)
    print("\n" + "=" * 60)
    print(f"Dataset: {path}")
    print(f"Schema version: {reader.metadata.get('schema_version', 'unknown')}")
    print(f"N samples: {len(reader)}")
    print(f"Benchmark: {reader.metadata.get('benchmark_name', 'unknown')}")
    print("=" * 60)

    n_show = min(args.n_show, len(reader))
    errors_total = []
    for i in range(n_show):
        sample = reader.read(i)
        print(f"\n--- Sample {i}: {sample.sample_id} ---")
        print(f"  Spatial shape:  {sample.inputs.spatial_shape}")
        print(f"  n_groups:       {sample.inputs.n_groups}")
        print(f"  n_spatial:      {sample.query.n_spatial}")
        print(f"  n_omega:        {sample.query.n_omega}")
        print(f"  time_dep:       {sample.query.is_time_dependent}")
        print(f"  params:         {sample.inputs.params}")
        print(f"  BC type:        {sample.inputs.bc.bc_type}")
        print(f"  metadata:       {sample.inputs.metadata}")
        print("  Arrays:")
        print_array_stats("sigma_a", sample.inputs.sigma_a)
        print_array_stats("sigma_s", sample.inputs.sigma_s)
        print_array_stats("q", sample.inputs.q)
        print_array_stats("x", sample.query.x)
        print_array_stats("omega", sample.query.omega)
        if sample.query.w_omega is not None:
            print_array_stats("w_omega", sample.query.w_omega)
            import numpy as np
            w_sum = sample.query.w_omega.sum()
            dim = sample.inputs.dim
            expected = 4 * 3.14159 if dim == 3 else 2 * 3.14159
            print(f"    -> sum(w_omega) = {w_sum:.4f} (expected ~{expected:.4f} for isotropic)")
        print_array_stats("I", sample.targets.I)
        print_array_stats("phi", sample.targets.phi)
        print_array_stats("J", sample.targets.J)
        if sample.targets.qois:
            for qk, qv in sample.targets.qois.items():
                print_array_stats(f"qoi_{qk}", qv)

        # Validate
        if args.validate:
            errs = sample.validate()
            if errs:
                print(f"  !! VALIDATION ERRORS: {errs}")
                errors_total.extend(errs)
            else:
                print(f"  [OK] Schema validation passed")

    print("\n" + "=" * 60)
    if errors_total:
        print(f"Total validation errors: {len(errors_total)}")
        for e in errors_total:
            print(f"  - {e}")
    else:
        print("All spot-checked samples passed validation.")

    # Summary statistics over all samples
    if len(reader) > 0:
        print("\nSummary statistics (first 20 samples):")
        import numpy as np
        n_check = min(20, len(reader))
        phi_means, eps_vals = [], []
        for i in range(n_check):
            s = reader.read(i)
            phi_means.append(s.targets.phi.mean())
            eps_vals.append(s.inputs.params.get("epsilon", float("nan")))
        print(f"  phi mean range: [{min(phi_means):.4e}, {max(phi_means):.4e}]")
        print(f"  epsilon range:  [{min(eps_vals):.4g}, {max(eps_vals):.4g}]")

    # Optional plotting
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            sample = reader.read(0)
            dim = sample.inputs.dim
            if dim == 2:
                H, W = sample.inputs.spatial_shape
                phi_grid = sample.targets.phi.reshape(H, W, -1)[:, :, 0]
                sigma_a_grid = sample.inputs.sigma_a.reshape(H, W, -1)[:, :, 0]

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                im0 = axes[0].imshow(sigma_a_grid, origin="lower", cmap="viridis")
                axes[0].set_title("sigma_a (group 0)")
                plt.colorbar(im0, ax=axes[0])

                im1 = axes[1].imshow(phi_grid, origin="lower", cmap="hot")
                axes[1].set_title("phi (scalar flux, group 0)")
                plt.colorbar(im1, ax=axes[1])

                # Angular distribution at center point
                cx = H // 2 * W + W // 2
                I_center = sample.targets.I[cx, :, 0]  # [Nw]
                omega_angles = np.arctan2(sample.query.omega[:, 1], sample.query.omega[:, 0])
                axes[2].plot(np.degrees(omega_angles), I_center, "o-")
                axes[2].set_xlabel("Direction angle (deg)")
                axes[2].set_ylabel("I(x_center, omega)")
                axes[2].set_title("Angular distribution at domain center")

                plt.tight_layout()
                plot_path = Path(args.dataset_path).parent / "inspect_plot.png"
                plt.savefig(plot_path, dpi=100)
                print(f"\nPlot saved to {plot_path}")
                plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
