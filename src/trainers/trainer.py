"""
Main training loop for transport neural operators.

Supports:
- AMP (automatic mixed precision)
- Gradient clipping
- EMA (exponential moving average) of model weights
- Tensorboard logging
- Checkpointing with resume
- Deterministic seeds
- Configurable optimizers and schedulers
"""

from __future__ import annotations
import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader

from ..data.dataset import collate_fn

logger = logging.getLogger(__name__)


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (self.decay * self.shadow[name]
                                     + (1 - self.decay) * param.data)

    def apply(self):
        """Apply EMA weights to model (for eval)."""
        self._orig = {}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self._orig[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights (after eval)."""
        for name, param in self.model.named_parameters():
            if name in self._orig:
                param.data.copy_(self._orig[name])
        self._orig = {}


class Trainer:
    """
    Trainer for transport neural operators.

    Handles:
    - Training loop with loss computation
    - Validation
    - Checkpointing and resume
    - AMP, EMA, grad clipping
    - Tensorboard logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        # Optimizer
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # Schedule
        scheduler: str = "cosine",
        n_epochs: int = 100,
        warmup_epochs: int = 5,
        # Training
        batch_size: int = 8,
        grad_clip: float = 1.0,
        use_amp: bool = False,
        # EMA
        use_ema: bool = False,
        ema_decay: float = 0.999,
        # Logging
        log_dir: str = "runs",
        run_name: str = "run",
        log_every: int = 10,
        val_every: int = 1,
        # Resume
        checkpoint_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        # Device
        device: Optional[str] = None,
        # Workers
        num_workers: int = 0,
        seed: int = 42,
        # Model hyperparameters (stored in checkpoint for reproducible eval)
        model_args: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.log_every = log_every
        self.val_every = val_every
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.seed = seed
        self.model_args = model_args or {}

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler
        total_steps = n_epochs * len(self.train_loader)
        warmup_steps = warmup_epochs * len(self.train_loader)
        if scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=lr * 0.01,
            )
        elif scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                total_steps=total_steps,
            )
        else:
            self.scheduler = None

        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None

        # EMA
        self.ema = EMA(model, ema_decay) if use_ema else None

        # Logging
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
        except ImportError:
            self.writer = None
            logger.warning("tensorboard not available; skipping TB logging.")

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        if resume_from:
            self._load_checkpoint(resume_from)

    def _compute_loss(self, pred: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Dispatch to model-specific loss if available, else use default L2 loss."""
        if hasattr(self.model, "compute_loss"):
            return self.model.compute_loss(pred, batch)

        # Default: L2 loss on I
        I_pred = pred["I"]
        I_true = batch["I"].to(self.device)
        omega_mask = batch.get("omega_mask")
        if omega_mask is not None:
            omega_mask = omega_mask.to(self.device)
            mask = omega_mask.unsqueeze(1).unsqueeze(-1).float()
            n_valid = mask.sum().clamp(min=1)
            loss_I = ((I_pred - I_true)**2 * mask).sum() / n_valid
        else:
            loss_I = torch.nn.functional.mse_loss(I_pred, I_true)

        phi_pred = pred.get("phi")
        phi_true = batch["phi"].to(self.device)
        loss_phi = torch.nn.functional.mse_loss(phi_pred, phi_true) if phi_pred is not None else torch.tensor(0.0)

        loss_total = loss_I + 0.1 * loss_phi
        return {
            "loss_total": loss_total,
            "loss_intensity": loss_I,
            "loss_phi": loss_phi,
        }

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensor values in batch to device."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_losses: Dict[str, float] = {}
        n_batches = 0

        for batch in self.train_loader:
            batch = self._move_batch(batch)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = self.model(batch)
                    losses = self._compute_loss(pred, batch)
                self.scaler.scale(losses["loss_total"]).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(batch)
                losses = self._compute_loss(pred, batch)
                losses["loss_total"].backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.ema is not None:
                self.ema.update()

            # Accumulate losses
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                step_losses = {k: v / n_batches for k, v in total_losses.items()}
                if self.writer:
                    for k, v in step_losses.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                if self.writer:
                    self.writer.add_scalar("train/lr", lr, self.global_step)

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        if self.ema is not None:
            self.ema.apply()

        self.model.eval()
        total_losses: Dict[str, float] = {}
        n_batches = 0

        for batch in self.val_loader:
            batch = self._move_batch(batch)
            pred = self.model(batch)
            losses = self._compute_loss(pred, batch)
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            n_batches += 1

        if self.ema is not None:
            self.ema.restore()

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    def train(self):
        """Full training loop."""
        logger.info(f"Starting training: {self.n_epochs} epochs, device={self.device}")
        for epoch in range(self.epoch, self.n_epochs):
            self.epoch = epoch
            t0 = time.time()
            train_losses = self.train_epoch()
            t1 = time.time()

            log_str = f"Epoch {epoch+1}/{self.n_epochs} | time={t1-t0:.1f}s"
            for k, v in train_losses.items():
                log_str += f" | train_{k}={v:.4e}"

            val_losses = {}
            if (epoch + 1) % self.val_every == 0:
                val_losses = self.validate()
                for k, v in val_losses.items():
                    log_str += f" | val_{k}={v:.4e}"
                if self.writer:
                    for k, v in val_losses.items():
                        self.writer.add_scalar(f"val/{k}", v, epoch)

            logger.info(log_str)

            # Save checkpoint
            val_loss = val_losses.get("loss_total", train_losses.get("loss_total", float("inf")))
            self._save_checkpoint(epoch, val_loss)

        if self.writer:
            self.writer.close()
        logger.info("Training complete.")

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save training checkpoint."""
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "model_args": self.model_args,
        }
        if self.scheduler is not None:
            state["scheduler_state"] = self.scheduler.state_dict()
        if self.ema is not None:
            state["ema_shadow"] = self.ema.shadow

        # Always save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)

        # Save best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            logger.info(f"New best model saved: val_loss={val_loss:.4e}")

    def _load_checkpoint(self, path: str):
        """Load training state from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if "ema_shadow" in ckpt and self.ema is not None:
            self.ema.shadow = ckpt["ema_shadow"]
        self.epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from {path}: epoch={self.epoch}")

    def save_model_only(self, path: str):
        """Save only model weights (for deployment)."""
        torch.save(self.model.state_dict(), path)
