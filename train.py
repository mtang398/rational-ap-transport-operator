"""
Training entry point for transport neural operators.

Usage:
  # Using argparse:
  python train.py --benchmark c5g7 --model ap_micromacro --n_epochs 100
  python train.py --benchmark c5g7 --model fno --seed 42

  # Using Hydra:
  python train.py benchmark=c5g7 model=ap_micromacro seed=42
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed, deterministic_mode
from src.data.dataset import MockDataset
from src.models import get_model
from src.trainers import Trainer

logger = logging.getLogger(__name__)

BENCHMARK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "c5g7":      {"spatial_shape": (51, 51), "n_omega": 8,  "n_groups": 7,  "dim": 2},
    "c5g7_td":   {"spatial_shape": (51, 51), "n_omega": 8,  "n_groups": 7,  "dim": 2, "time_dependent": True},
    "kobayashi": {"spatial_shape": (20, 20, 20), "n_omega": 24, "n_groups": 1, "dim": 3},
    "pinte2009": {"spatial_shape": (32, 32), "n_omega": 16, "n_groups": 1,  "dim": 2},
}


def parse_args():
    p = argparse.ArgumentParser(description="Train transport neural operator")
    # Benchmark
    p.add_argument("--benchmark", default="c5g7", choices=list(BENCHMARK_DEFAULTS.keys()))
    p.add_argument("--n_samples_train", type=int, default=200)
    p.add_argument("--n_samples_val", type=int, default=50)
    p.add_argument("--epsilon_min", type=float, default=0.01)
    p.add_argument("--epsilon_max", type=float, default=1.0)
    p.add_argument("--resample_omega_range", type=int, nargs=2, default=None,
                   help="E.g. --resample_omega_range 4 32 for variable-Nw training")
    # Model
    p.add_argument("--model", default="ap_micromacro", choices=["fno", "deeponet", "ap_micromacro"])
    p.add_argument("--fno_channels", type=int, default=32)
    p.add_argument("--n_fno_blocks", type=int, default=4)
    p.add_argument("--n_modes", type=int, default=12)
    p.add_argument("--n_basis", type=int, default=128)
    p.add_argument("--macro_channels", type=int, default=32)
    p.add_argument("--micro_latent_dim", type=int, default=64)
    p.add_argument("--activation", default="gelu", choices=["gelu", "silu", "relu"])
    p.add_argument("--lambda_moment", type=float, default=1.0)
    p.add_argument("--lambda_diffusion", type=float, default=0.1)
    # Training
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "onecycle", "none"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    # Logging
    p.add_argument("--log_dir", default="runs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--resume_from", default=None)
    # System
    p.add_argument("--device", default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_model(args, bm_defaults: Dict[str, Any]):
    """Build model from args."""
    dim = bm_defaults["dim"]
    n_groups = bm_defaults["n_groups"]
    n_bc_faces = 2 * dim
    time_dependent = bm_defaults.get("time_dependent", False)

    if args.model == "fno":
        from src.models.fno import FNOTransport
        return FNOTransport(
            dim=dim, n_groups=n_groups, n_params=2,
            fno_channels=args.fno_channels,
            n_fno_blocks=args.n_fno_blocks,
            n_modes=args.n_modes,
            activation=args.activation,
            time_dependent=time_dependent,
            n_bc_faces=n_bc_faces,
        )
    elif args.model == "deeponet":
        from src.models.deeponet import DeepONetTransport
        return DeepONetTransport(
            dim=dim, n_groups=n_groups, n_params=2,
            n_basis=args.n_basis,
            activation=args.activation,
            time_dependent=time_dependent,
            n_bc_faces=n_bc_faces,
        )
    elif args.model == "ap_micromacro":
        from src.models.ap_micromacro import APMicroMacroTransport
        return APMicroMacroTransport(
            dim=dim, n_groups=n_groups, n_params=2,
            macro_channels=args.macro_channels,
            n_fno_blocks=args.n_fno_blocks,
            n_modes=args.n_modes,
            micro_latent_dim=args.micro_latent_dim,
            activation=args.activation,
            time_dependent=time_dependent,
            n_bc_faces=n_bc_faces,
            lambda_moment=args.lambda_moment,
            lambda_diffusion=args.lambda_diffusion,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


def main():
    setup_logging("INFO")
    args = parse_args()
    set_seed(args.seed)
    deterministic_mode(True)

    bm_def = BENCHMARK_DEFAULTS.get(args.benchmark, BENCHMARK_DEFAULTS["c5g7"])

    run_name = args.run_name or f"{args.benchmark}_{args.model}_seed{args.seed}"
    logger.info(f"Run: {run_name}")
    logger.info(f"Benchmark: {args.benchmark}, Model: {args.model}")

    # Build datasets
    import numpy as np
    epsilon_range = (args.epsilon_min, args.epsilon_max)
    resample_omega_range = tuple(args.resample_omega_range) if args.resample_omega_range else None

    train_ds = MockDataset(
        n_samples=args.n_samples_train,
        spatial_shape=bm_def["spatial_shape"],
        n_omega=bm_def["n_omega"],
        n_groups=bm_def["n_groups"],
        benchmark_name=args.benchmark,
        epsilon_range=epsilon_range,
        seed=args.seed,
        resample_omega_range=resample_omega_range,
    )
    val_ds = MockDataset(
        n_samples=args.n_samples_val,
        spatial_shape=bm_def["spatial_shape"],
        n_omega=bm_def["n_omega"],
        n_groups=bm_def["n_groups"],
        benchmark_name=args.benchmark,
        epsilon_range=epsilon_range,
        seed=args.seed + 1000,
    )

    logger.info(f"Train dataset: {len(train_ds)} samples")
    logger.info(f"Val dataset: {len(val_ds)} samples")

    # Build model
    model = build_model(args, bm_def)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model} | Parameters: {n_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        n_epochs=args.n_epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        log_dir=args.log_dir,
        run_name=run_name,
        log_every=args.log_every,
        val_every=args.val_every,
        resume_from=args.resume_from,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        model_args=vars(args),
    )

    trainer.train()
    logger.info(f"Training complete. Best val loss: {trainer.best_val_loss:.4e}")
    logger.info(f"Checkpoints saved in: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    main()
