"""
Hyperparameter sweep script.

Runs grid sweeps over seeds, resolutions, SN orders, and epsilon values.
Aggregates all results into a single CSV.

Usage:
  python sweep.py --benchmark c5g7 --model ap_micromacro \\
                  --seeds 1 2 3 --n_omega 4 8 16 --n_epochs 20
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Grid sweep over hyperparameters")
    p.add_argument("--benchmark", default="c5g7")
    p.add_argument("--model", default="ap_micromacro")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--n_omega", type=int, nargs="+", default=[8, 16])
    p.add_argument("--resolution_mult", type=int, nargs="+", default=[1, 2])
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_samples_train", type=int, default=100)
    p.add_argument("--output_dir", default="runs/sweep")
    p.add_argument("--data_dir", default="runs/datasets",
                   help="Directory with pre-generated datasets. On-the-fly solver "
                        "generation is used when a split file is not found.")
    p.add_argument("--solver", default="auto",
                   choices=["auto", "mock", "opensn", "openmc"],
                   help="Solver for on-the-fly generation. 'auto' uses the best "
                        "available real solver, falling back to mock.")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running")
    p.add_argument("--device", default=None)
    return p.parse_args()


BENCHMARK_DEFAULTS = {
    "c5g7":      {"spatial_shape": (51, 51), "n_groups": 7,  "dim": 2},
    "pinte2009": {"spatial_shape": (32, 32),  "n_groups": 1, "dim": 2},
}


def run_training(args, seed: int, n_omega: int, resolution_mult: int) -> Dict[str, Any]:
    """Run one training configuration."""
    bm_def = BENCHMARK_DEFAULTS.get(args.benchmark, BENCHMARK_DEFAULTS["c5g7"])
    base_shape = bm_def["spatial_shape"]
    shape = tuple(s * resolution_mult for s in base_shape)

    run_name = f"{args.benchmark}_{args.model}_seed{seed}_nw{n_omega}_res{resolution_mult}"

    cmd = [
        sys.executable, "train.py",
        "--benchmark", args.benchmark,
        "--model", args.model,
        "--n_epochs", str(args.n_epochs),
        "--batch_size", str(args.batch_size),
        "--n_samples_train", str(args.n_samples_train),
        "--n_samples_val", "25",
        "--run_name", run_name,
        "--seed", str(seed),
        "--log_dir", args.output_dir,
        "--data_dir", args.data_dir,
        "--solver", args.solver,
    ]
    if args.device:
        cmd += ["--device", args.device]

    logger.info(f"Running: {' '.join(cmd)}")

    if args.dry_run:
        return {"run_name": run_name, "status": "dry_run"}

    result = subprocess.run(cmd, capture_output=False, text=True)
    status = "success" if result.returncode == 0 else "failed"

    ckpt_path = Path(args.output_dir) / run_name / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(args.output_dir) / run_name / "checkpoints" / "latest.pt"

    return {
        "run_name": run_name,
        "status": status,
        "checkpoint": str(ckpt_path),
        "seed": seed,
        "n_omega": n_omega,
        "resolution_mult": resolution_mult,
        "benchmark": args.benchmark,
        "model": args.model,
    }


def run_eval(args, run_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run evaluation on a trained model."""
    ckpt = run_info["checkpoint"]
    if not Path(ckpt).exists():
        logger.warning(f"Checkpoint not found: {ckpt}")
        return {"status": "no_checkpoint"}

    eval_out = Path(args.output_dir) / "eval" / run_info["run_name"]

    cmd = [
        sys.executable, "eval.py",
        "--checkpoint", ckpt,
        "--benchmark", run_info["benchmark"],
        "--model", run_info["model"],
        "--protocol", "all",
        "--output_dir", str(eval_out),
        "--n_test_samples", "30",
        "--batch_size", "4",
        "--seed", str(run_info["seed"]),
        "--data_dir", args.data_dir,
        "--solver", args.solver,
    ]

    logger.info(f"Eval: {' '.join(cmd)}")

    if args.dry_run:
        return {"eval_status": "dry_run"}

    result = subprocess.run(cmd, capture_output=False, text=True)

    # Try to read summary CSV
    summary_csv = eval_out / "summary.csv"
    metrics = {}
    if summary_csv.exists():
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row.get('protocol', 'unk')}_{row.get('key', 'unk')}"
                for k, v in row.items():
                    if k not in ("protocol", "key"):
                        metrics[f"{key}_{k}"] = v

    return {
        "eval_status": "success" if result.returncode == 0 else "failed",
        "eval_dir": str(eval_out),
        **metrics,
    }


def main():
    setup_logging("INFO")
    args = parse_args()

    from src.utils.logging_utils import log_environment_info
    log_environment_info(
        benchmark_name=args.benchmark,
        model_name=args.model,
        data_dir="runs/datasets",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grid over all combinations
    configs = list(product(args.seeds, args.n_omega, args.resolution_mult))
    logger.info(f"Sweep: {len(configs)} configurations")
    logger.info(f"  seeds={args.seeds}, n_omega={args.n_omega}, resolution_mult={args.resolution_mult}")

    all_results = []

    for seed, n_omega, res_mult in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Config: seed={seed}, n_omega={n_omega}, resolution_mult={res_mult}")

        # Train
        run_info = run_training(args, seed, n_omega, res_mult)
        if run_info.get("status") == "failed":
            logger.error(f"Training failed for {run_info['run_name']}")

        # Eval
        eval_info = run_eval(args, run_info)

        combined = {**run_info, **eval_info}
        all_results.append(combined)

    # Save aggregate CSV
    agg_csv = output_dir / "sweep_results.csv"
    if all_results:
        fieldnames = sorted(set(k for r in all_results for k in r.keys()))
        with open(agg_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        logger.info(f"\nSweep results saved: {agg_csv}")

    # Save JSON
    agg_json = output_dir / "sweep_results.json"
    with open(agg_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"Sweep complete. {len(all_results)} runs.")
    logger.info(f"Results: {agg_csv}")


if __name__ == "__main__":
    main()
