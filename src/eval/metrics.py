"""
Evaluation metrics for transport neural operators.

Metrics:
- L2 error and relative L2 error for I (masked over omega)
- Moment errors: phi, J
- QoI errors (benchmark-specific)
- Energy balance residual (best-effort, reports N/A if fields missing)

Relative L2 convention
----------------------
All relative L2 errors are computed **per sample** and then averaged across
the batch.  This is the standard convention in operator-learning papers
(e.g. FNO, DeepONet) and avoids the denominator-collapse problem that occurs
when fields such as J have near-zero global norm (common in diffusive regimes
where the net current is small relative to the scalar flux).

Formally, for a batch of B samples:
    rel_L2 = (1/B) Σ_b ||pred_b - true_b|| / (||true_b|| + ε)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MetricBundle:
    """Container for all evaluation metrics for one batch/sample."""
    # Intensity metrics
    I_l2: float = float("nan")
    I_rel_l2: float = float("nan")
    # Moment metrics
    phi_l2: float = float("nan")
    phi_rel_l2: float = float("nan")
    J_l2: float = float("nan")
    J_rel_l2: float = float("nan")
    # QoI metrics (benchmark-specific, keyed by QoI name)
    qoi_errors: Dict[str, float] = field(default_factory=dict)
    # Conservation / energy balance residual
    energy_balance_residual: float = float("nan")  # N/A if not computable
    # Metadata
    n_omega: int = 0
    n_spatial: int = 0
    epsilon: float = float("nan")
    benchmark: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def l2_error(pred: Tensor, true: Tensor, mask: Optional[Tensor] = None) -> float:
    """Compute RMS L2 error averaged over all elements (or masked elements)."""
    diff = pred - true
    if mask is not None:
        diff = diff * mask.float()
        n = mask.float().sum().clamp(min=1)
        return math.sqrt((diff**2).sum().item() / n.item())
    return math.sqrt(F.mse_loss(pred, true).item())


def relative_l2_error(pred: Tensor, true: Tensor, mask: Optional[Tensor] = None,
                      eps: float = 1e-8) -> float:
    """
    Compute per-sample relative L2 error, averaged over the batch.

    For each sample b:
        r_b = ||pred_b - true_b||_F / (||true_b||_F + eps)
    Returns mean(r_b) over all B samples in the leading batch dimension.

    This per-sample averaging is robust to near-zero global norms (e.g. J in
    diffusive regimes) and matches the convention in FNO / DeepONet papers.
    """
    B = pred.shape[0]
    # Flatten everything except the batch dimension
    pred_flat = pred.reshape(B, -1)
    true_flat = true.reshape(B, -1)

    if mask is not None:
        # mask broadcasts: flatten to [B, -1] matching pred_flat
        mask_flat = mask.reshape(B, -1).float()
        diff_flat = (pred_flat - true_flat) * mask_flat
        true_flat = true_flat * mask_flat
    else:
        diff_flat = pred_flat - true_flat

    num = (diff_flat ** 2).sum(dim=1)          # [B]
    denom = (true_flat ** 2).sum(dim=1) + eps  # [B]
    per_sample = torch.sqrt(num / denom)        # [B]
    return per_sample.mean().item()


def compute_metrics(
    pred: Dict[str, Tensor],
    batch: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> MetricBundle:
    """
    Compute all metrics for one batch.

    Args:
        pred: model output dict (I, phi, J, ...)
        batch: ground truth batch dict
        device: torch device

    Returns:
        MetricBundle with all metrics filled in
    """
    def _to(t):
        if device is not None:
            return t.to(device)
        return t

    metrics = MetricBundle()

    # Extract predictions and targets
    I_pred = pred.get("I")
    phi_pred = pred.get("phi") if pred.get("phi") is not None else pred.get("phi_I")
    J_pred = pred.get("J") if pred.get("J") is not None else pred.get("J_I")

    I_true = _to(batch["I"]) if "I" in batch else None
    phi_true = _to(batch["phi"]) if "phi" in batch else None
    J_true = _to(batch["J"]) if "J" in batch else None

    omega_mask = _to(batch["omega_mask"]) if "omega_mask" in batch else None

    # --- Intensity metrics ---
    if I_pred is not None and I_true is not None:
        # Build mask [B, Nx, Nw, G] for l2_error (element-wise masking)
        # and a broadcastable version for relative_l2_error (reshape-based).
        if omega_mask is not None:
            # [B, Nw] -> [B, Nx, Nw, G]  (contiguous, not just a view)
            I_mask = omega_mask.unsqueeze(1).unsqueeze(-1).expand_as(I_pred).contiguous()
        else:
            I_mask = None

        metrics.I_l2 = l2_error(I_pred, I_true, I_mask)
        metrics.I_rel_l2 = relative_l2_error(I_pred, I_true, I_mask)

    # --- Moment metrics ---
    if phi_pred is not None and phi_true is not None:
        metrics.phi_l2 = l2_error(phi_pred, phi_true)
        metrics.phi_rel_l2 = relative_l2_error(phi_pred, phi_true)

    if J_pred is not None and J_true is not None:
        metrics.J_l2 = l2_error(J_pred, J_true)
        metrics.J_rel_l2 = relative_l2_error(J_pred, J_true)

    # --- QoI metrics ---
    for key in batch:
        if key.startswith("qoi_"):
            qoi_name = key[4:]
            pred_key = key
            if pred_key in pred:
                qoi_pred = pred[pred_key]
                qoi_true = _to(batch[key])
                metrics.qoi_errors[qoi_name] = relative_l2_error(qoi_pred, qoi_true)

    # --- Energy balance residual (best-effort) ---
    # Check: int phi * sigma_a dx ~ int q dx (fixed source balance)
    if phi_pred is not None and "sigma_a" in batch and "q" in batch:
        try:
            sigma_a = _to(batch["sigma_a"])  # [B, Nx, G]
            q_src = _to(batch["q"])          # [B, Nx, G]
            absorption = (phi_pred * sigma_a).sum(dim=(1, 2))  # [B]
            source = q_src.sum(dim=(1, 2))                      # [B]
            residual = ((absorption - source).abs() / (source.abs() + 1e-8)).mean().item()
            metrics.energy_balance_residual = residual
        except Exception:
            pass  # Report as NaN if computation fails

    # --- Metadata ---
    if omega_mask is not None:
        metrics.n_omega = int(omega_mask[0].sum().item())
    elif "omega" in batch:
        metrics.n_omega = batch["omega"].shape[1]
    if "x" in batch:
        metrics.n_spatial = batch["x"].shape[1]
    if "params" in batch and "param_keys" in batch:
        keys = batch["param_keys"]
        if "epsilon" in keys:
            eps_idx = keys.index("epsilon")
            metrics.epsilon = float(batch["params"][0, eps_idx].item())
    if "metadata" in batch:
        meta = batch.get("metadata", {})
        if isinstance(meta, dict):
            metrics.benchmark = meta.get("benchmark_name", "")
        elif isinstance(meta, list) and len(meta) > 0:
            metrics.benchmark = meta[0].get("benchmark_name", "")

    return metrics


def aggregate_metrics(metric_list: list) -> Dict[str, float]:
    """Average a list of MetricBundles into a summary dict."""
    if not metric_list:
        return {}

    keys = ["I_l2", "I_rel_l2", "phi_l2", "phi_rel_l2", "J_l2", "J_rel_l2", "energy_balance_residual"]
    summary = {}
    for k in keys:
        vals = [getattr(m, k) for m in metric_list if not math.isnan(getattr(m, k))]
        summary[k] = sum(vals) / len(vals) if vals else float("nan")
        summary[f"{k}_std"] = float(torch.tensor(vals).std().item()) if len(vals) > 1 else (0.0 if vals else float("nan"))

    # QoI aggregation
    all_qoi_keys = set()
    for m in metric_list:
        all_qoi_keys.update(m.qoi_errors.keys())
    for qk in all_qoi_keys:
        vals = [m.qoi_errors[qk] for m in metric_list if qk in m.qoi_errors]
        summary[f"qoi_{qk}_rel_l2"] = sum(vals) / len(vals) if vals else float("nan")

    return summary
