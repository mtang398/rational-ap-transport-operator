"""
Evaluation entry point.

Loads a checkpoint and runs evaluation protocols:
  - test_set:            baseline metrics on the held-out test split (I, phi, J)
  - sn_transfer:         evaluate at unseen angular discretizations
  - resolution_transfer: evaluate at finer spatial grids (bilinear-interpolated targets)
  - regime_sweep:        evaluate across epsilon values (NOT supported for c5g7)
  - all:                 run all supported protocols for the benchmark

Protocol support by benchmark
------------------------------
  c5g7      : test_set, sn_transfer, resolution_transfer
              (regime_sweep is skipped — C5G7 is a fixed k-eigenvalue problem;
               epsilon is not a physical parameter of the OpenMC simulation,
               so a regime sweep would not test genuine transport-regime variation)
  pinte2009 : test_set, sn_transfer, resolution_transfer, regime_sweep

Saves per-protocol CSVs, metrics.json, and summary.csv.

Usage:
  python eval.py --checkpoint runs/c5g7_ap_micromacro_seed42/checkpoints/best.pt \\
      --benchmark c5g7 --model ap_micromacro --data_dir runs/datasets
  python eval.py --checkpoint runs/.../best.pt --protocol test_set --benchmark c5g7 \\
      --data_dir runs/datasets
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_utils import setup_logging, log_environment_info
from src.utils.seed import set_seed
from src.utils.io_utils import save_json, save_csv
from src.eval.protocols import (
    TestSetProtocol,
    SNTransferProtocol,
    ResolutionTransferProtocol,
    RegimeSweepProtocol,
)
from src.data.dataset import TransportDataset
from train import load_dataset, _find_dataset_path

logger = logging.getLogger(__name__)

BENCHMARK_DEFAULTS = {
    "c5g7":      {"spatial_shape": (51, 51),       "n_omega": 8,  "n_groups": 7, "dim": 2},
    "pinte2009": {"spatial_shape": (32, 32),       "n_omega": 16, "n_groups": 1, "dim": 2},
}

# Benchmarks for which the regime_sweep protocol is not physically meaningful
# and must be skipped.
#
# c5g7: eigenvalue (k-eff) problem solved by OpenMC. Epsilon is stored as a
#   model-input parameter but does NOT affect the OpenMC simulation — the
#   physics is identical across all epsilon values.  Running a regime sweep
#   would only test whether the model ignores an irrelevant scalar, not whether
#   it generalises across transport regimes.
NO_REGIME_SWEEP = {"c5g7"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate transport neural operator")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    p.add_argument("--benchmark", default="c5g7", choices=list(BENCHMARK_DEFAULTS.keys()))
    p.add_argument("--model", default="ap_micromacro", choices=["fno", "deeponet", "ap_micromacro"])
    p.add_argument("--protocol", default="all",
                   choices=["test_set", "sn_transfer", "resolution_transfer", "regime_sweep", "all"])
    # SN transfer
    p.add_argument("--train_n_omega", type=int, default=8)
    p.add_argument("--test_n_omegas", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32, 64])
    # Resolution transfer
    p.add_argument("--resolution_multipliers", type=int, nargs="+", default=[1, 2, 4])
    # Regime sweep
    p.add_argument("--epsilon_values", type=float, nargs="+",
                   default=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
    # Common
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_test_samples", type=int, default=50)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--data_dir", required=True,
                   help="Directory containing pre-generated <benchmark>_test.zarr and "
                        "all evaluation splits.  Run generate_dataset.py --split all_eval first.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    # Model arch params (to recreate model)
    p.add_argument("--fno_channels", type=int, default=32)
    p.add_argument("--n_fno_blocks", type=int, default=4)
    p.add_argument("--n_modes", type=int, default=12)
    p.add_argument("--n_basis", type=int, default=128)
    p.add_argument("--macro_channels", type=int, default=32)
    p.add_argument("--micro_latent_dim", type=int, default=64)
    p.add_argument("--activation", default="gelu")
    p.add_argument("--lambda_moment", type=float, default=1.0)
    p.add_argument("--lambda_diffusion", type=float, default=0.1)
    return p.parse_args()


def load_model(args, bm_def):
    """Recreate and load model from checkpoint."""
    import torch
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Override arch args with values stored in the checkpoint so the model
    # architecture exactly matches the saved weights.
    saved_args = ckpt.get("model_args", {})
    arch_keys = [
        "model", "fno_channels", "n_fno_blocks", "n_modes", "n_basis",
        "macro_channels", "micro_latent_dim", "activation",
        "lambda_moment", "lambda_diffusion",
    ]
    for key in arch_keys:
        if key in saved_args:
            setattr(args, key, saved_args[key])

    dim = bm_def["dim"]
    n_groups = bm_def["n_groups"]
    n_bc_faces = 2 * dim
    time_dependent = bm_def.get("time_dependent", False)

    if args.model == "fno":
        from src.models.fno import FNOTransport
        model = FNOTransport(
            dim=dim, n_groups=n_groups, n_params=2,
            fno_channels=args.fno_channels, n_fno_blocks=args.n_fno_blocks,
            n_modes=args.n_modes, activation=args.activation,
            time_dependent=time_dependent, n_bc_faces=n_bc_faces,
        )
    elif args.model == "deeponet":
        from src.models.deeponet import DeepONetTransport
        model = DeepONetTransport(
            dim=dim, n_groups=n_groups, n_params=2, n_basis=args.n_basis,
            activation=args.activation, time_dependent=time_dependent, n_bc_faces=n_bc_faces,
        )
    elif args.model == "ap_micromacro":
        from src.models.ap_micromacro import APMicroMacroTransport
        model = APMicroMacroTransport(
            dim=dim, n_groups=n_groups, n_params=2,
            macro_channels=args.macro_channels, n_fno_blocks=args.n_fno_blocks,
            n_modes=args.n_modes, micro_latent_dim=args.micro_latent_dim,
            activation=args.activation, time_dependent=time_dependent,
            n_bc_faces=n_bc_faces, lambda_moment=args.lambda_moment,
            lambda_diffusion=args.lambda_diffusion,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    return model, device


def main():
    setup_logging("INFO")
    args = parse_args()
    set_seed(args.seed)

    bm_def = BENCHMARK_DEFAULTS.get(args.benchmark, BENCHMARK_DEFAULTS["c5g7"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"runs/eval/{args.benchmark}_{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from: {args.checkpoint}")
    model, device = load_model(args, bm_def)

    test_ds, test_source = load_dataset(
        data_dir=args.data_dir,
        benchmark=args.benchmark,
        split="test",
        bm_def=bm_def,
        max_samples=args.n_test_samples,
    )

    log_environment_info(
        benchmark_name=args.benchmark,
        model_name=args.model,
        source_override=f"Test: {test_source}",
    )

    # Protocols expect a flat list of TransportSample objects
    if hasattr(test_ds, "_samples") and test_ds._samples is not None:
        test_samples = test_ds._samples
    else:
        test_samples = [test_ds.reader.read(i) for i in range(len(test_ds))]

    # Load any results already written by a previous eval.py invocation on the
    # same output_dir (run_all.py calls eval.py once per protocol).  This
    # ensures that when --protocol sn_transfer runs first and then
    # --protocol resolution_transfer runs second, the final metrics.json and
    # summary.csv contain ALL protocols rather than only the last one.
    metrics_path = output_dir / "metrics.json"
    all_results: dict = {}
    if metrics_path.exists():
        from src.utils.io_utils import load_json
        try:
            all_results = load_json(str(metrics_path))
            logger.info(f"Loaded existing metrics from {metrics_path} — will merge new results.")
        except Exception as e:
            # Preserve the corrupt file for diagnosis; crash loudly rather than
            # silently discarding previously written protocol results.
            corrupt_path = metrics_path.with_suffix(".corrupt.json")
            import shutil
            shutil.copy(str(metrics_path), str(corrupt_path))
            raise RuntimeError(
                f"metrics.json is corrupt or unreadable ({e}).\n"
                f"A copy has been saved to {corrupt_path}.\n"
                f"Delete {metrics_path} to start fresh, or fix the file manually."
            ) from e

    # --- Test Set (baseline) ---
    if args.protocol in ("test_set", "all"):
        logger.info("Running test set evaluation...")
        ts_proto = TestSetProtocol(
            model=model,
            test_samples=test_samples,
            device=str(device),
            batch_size=args.batch_size,
        )
        ts_results = ts_proto.run()
        ts_csv = str(output_dir / "test_set.csv")
        ts_proto.to_csv(ts_results, ts_csv)
        all_results["test_set"] = ts_results
        logger.info(f"Test set results saved to: {ts_csv}")

    # --- SN Transfer ---
    if args.protocol in ("sn_transfer", "all"):
        logger.info("Running SN transfer protocol...")
        sn_proto = SNTransferProtocol(
            model=model,
            test_samples=test_samples,
            train_n_omega=args.train_n_omega,
            test_n_omegas=args.test_n_omegas,
            device=str(device),
            batch_size=args.batch_size,
        )
        sn_results = sn_proto.run()
        sn_csv = str(output_dir / "sn_transfer.csv")
        sn_proto.to_csv(sn_results, sn_csv)
        all_results["sn_transfer"] = sn_results
        logger.info(f"SN transfer results saved to: {sn_csv}")

    # --- Resolution Transfer ---
    if args.protocol in ("resolution_transfer", "all"):
        logger.info("Running resolution transfer protocol...")
        res_proto = ResolutionTransferProtocol(
            model=model,
            base_spatial_shape=bm_def["spatial_shape"],
            resolution_multipliers=args.resolution_multipliers,
            n_groups=bm_def["n_groups"],
            benchmark_name=args.benchmark,
            n_test_samples=args.n_test_samples,
            device=str(device),
            batch_size=args.batch_size,
            n_omega=args.train_n_omega,
            base_test_samples=test_samples,
            data_dir=args.data_dir,
        )
        res_results = res_proto.run()
        res_csv = str(output_dir / "resolution_transfer.csv")
        res_proto.to_csv(res_results, res_csv)
        all_results["resolution_transfer"] = res_results
        logger.info(f"Resolution transfer results saved to: {res_csv}")

    # --- Regime Sweep ---
    if args.protocol in ("regime_sweep", "all"):
        if args.benchmark in NO_REGIME_SWEEP:
            logger.warning(
                f"Skipping regime_sweep for benchmark '{args.benchmark}': "
                "epsilon is not a physical parameter of the underlying solver "
                "(OpenMC k-eigenvalue problem), so a regime sweep would not "
                "test genuine transport-regime generalisation. "
                "See module docstring for details."
            )
        else:
            logger.info("Running regime sweep protocol...")
            sweep_proto = RegimeSweepProtocol(
                model=model,
                epsilon_values=args.epsilon_values,
                spatial_shape=bm_def["spatial_shape"],
                n_omega=args.train_n_omega,
                n_groups=bm_def["n_groups"],
                benchmark_name=args.benchmark,
                n_test_samples=args.n_test_samples,
                device=str(device),
                batch_size=args.batch_size,
                data_dir=args.data_dir,
            )
            sweep_results = sweep_proto.run()
            sweep_csv = str(output_dir / "regime_sweep.csv")
            sweep_proto.to_csv(sweep_results, sweep_csv)
            all_results["regime_sweep"] = sweep_results
            logger.info(f"Regime sweep results saved to: {sweep_csv}")

    # Save merged metrics.json (contains all protocols run so far)
    save_json(all_results, str(metrics_path))
    logger.info(f"Full metrics saved to: {metrics_path}")

    # Save summary CSV — rebuild entirely from all_results so it always
    # reflects every protocol that has been run, including previous invocations.
    summary_rows = []
    for proto_name, proto_results in all_results.items():
        for key, result in proto_results.items():
            row = {"protocol": proto_name, "key": str(key)}
            row.update({k: v for k, v in result.items() if isinstance(v, (int, float, str))})
            summary_rows.append(row)

    summary_csv = str(output_dir / "summary.csv")
    save_csv(summary_rows, summary_csv)
    logger.info(f"Summary CSV saved to: {summary_csv}")

    # Print key metrics
    logger.info("=" * 50)
    logger.info("KEY RESULTS SUMMARY")
    for proto_name, proto_results in all_results.items():
        logger.info(f"  {proto_name}:")
        for key, result in proto_results.items():
            I_rel   = result.get("I_rel_l2",   float("nan"))
            phi_rel = result.get("phi_rel_l2", float("nan"))
            J_rel   = result.get("J_rel_l2",   float("nan"))
            logger.info(
                f"    {key}: I_rel_l2={I_rel:.4e}  "
                f"phi_rel_l2={phi_rel:.4e}  J_rel_l2={J_rel:.4e}"
            )


if __name__ == "__main__":
    main()
