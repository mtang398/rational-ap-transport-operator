"""
Evaluation protocols for transport neural operators.

1. SN Transfer: evaluate at unseen angular discretizations
2. Resolution Transfer: evaluate at unseen spatial resolutions
3. Regime Sweep: evaluate across epsilon values
"""

from __future__ import annotations
import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import MockDataset, collate_fn, resample_omega_directions
from ..data.schema import TransportSample
from .metrics import compute_metrics, aggregate_metrics, MetricBundle

logger = logging.getLogger(__name__)


class SNTransferProtocol:
    """
    SN Transfer Evaluation Protocol.

    Tests discretization-agnosticism: model trained on one set of angular
    quadratures, evaluated on different SN orders / direction counts.

    Reports: I_rel_l2, phi_rel_l2 vs n_omega
    """

    def __init__(
        self,
        model: nn.Module,
        test_samples: List[TransportSample],
        train_n_omega: int = 8,
        test_n_omegas: List[int] = None,
        device: Optional[str] = None,
        batch_size: int = 4,
    ):
        self.model = model
        self.test_samples = test_samples
        self.train_n_omega = train_n_omega
        self.test_n_omegas = test_n_omegas or [4, 8, 16, 32, 64]
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def run(self) -> Dict[str, Any]:
        """
        Run SN transfer evaluation.

        Returns:
            dict mapping n_omega -> aggregate metrics
        """
        self.model.eval()
        results = {}

        for n_omega in self.test_n_omegas:
            logger.info(f"SN transfer: evaluating at n_omega={n_omega}")
            # Resample directions
            resampled = [
                resample_omega_directions(s, n_omega, np.random.default_rng(42))
                for s in self.test_samples
            ]

            from ..data.dataset import TransportDataset
            ds = TransportDataset(resampled)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                                collate_fn=collate_fn)

            metric_list = []
            t0 = time.time()
            with torch.no_grad():
                for batch in loader:
                    batch_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                    pred = self.model(batch_dev)
                    m = compute_metrics(pred, batch_dev, self.device)
                    m.n_omega = n_omega
                    metric_list.append(m)
            t1 = time.time()

            summary = aggregate_metrics(metric_list)
            summary["n_omega"] = n_omega
            summary["runtime_s"] = t1 - t0
            summary["is_train_nw"] = (n_omega == self.train_n_omega)
            results[n_omega] = summary
            logger.info(f"  n_omega={n_omega}: I_rel_l2={summary.get('I_rel_l2', float('nan')):.4e}")

        return results

    def to_csv(self, results: Dict[str, Any], path: str):
        rows = list(results.values())
        if not rows:
            return
        fieldnames = sorted(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class ResolutionTransferProtocol:
    """
    Resolution Transfer Evaluation Protocol.

    Trains on coarse spatial grid, evaluates at higher resolutions.
    The model handles this naturally since x is a continuous query.

    Reports: I_rel_l2, phi_rel_l2 vs spatial_resolution and runtime
    """

    def __init__(
        self,
        model: nn.Module,
        base_spatial_shape: tuple,
        resolution_multipliers: List[int] = None,
        n_groups: int = 1,
        benchmark_name: str = "mock",
        n_test_samples: int = 50,
        device: Optional[str] = None,
        batch_size: int = 4,
        n_omega: int = 8,
    ):
        self.model = model
        self.base_shape = base_spatial_shape
        self.resolution_multipliers = resolution_multipliers or [1, 2, 4]
        self.n_groups = n_groups
        self.benchmark_name = benchmark_name
        self.n_test = n_test_samples
        self.batch_size = batch_size
        self.n_omega = n_omega

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def _make_dataset_at_resolution(self, multiplier: int) -> MockDataset:
        new_shape = tuple(s * multiplier for s in self.base_shape)
        return MockDataset(
            n_samples=self.n_test,
            spatial_shape=new_shape,
            n_omega=self.n_omega,
            n_groups=self.n_groups,
            benchmark_name=self.benchmark_name,
            seed=999,
        )

    def run(self) -> Dict[str, Any]:
        self.model.eval()
        results = {}

        for mult in self.resolution_multipliers:
            shape = tuple(s * mult for s in self.base_shape)
            logger.info(f"Resolution transfer: evaluating at shape={shape} (x{mult})")

            ds = self._make_dataset_at_resolution(mult)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                                collate_fn=collate_fn)

            metric_list = []
            t0 = time.time()
            with torch.no_grad():
                for batch in loader:
                    batch_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                    try:
                        pred = self.model(batch_dev)
                        m = compute_metrics(pred, batch_dev, self.device)
                        m.n_spatial = int(np.prod(shape))
                        metric_list.append(m)
                    except Exception as e:
                        logger.warning(f"Forward failed at resolution x{mult}: {e}")
            t1 = time.time()

            summary = aggregate_metrics(metric_list)
            summary["resolution_multiplier"] = mult
            summary["spatial_shape"] = str(shape)
            summary["n_spatial"] = int(np.prod(shape))
            summary["runtime_s"] = t1 - t0
            results[mult] = summary
            logger.info(f"  x{mult}: I_rel_l2={summary.get('I_rel_l2', float('nan')):.4e}, runtime={t1-t0:.2f}s")

        return results

    def to_csv(self, results: Dict[str, Any], path: str):
        rows = list(results.values())
        if not rows:
            return
        fieldnames = sorted(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class RegimeSweepProtocol:
    """
    Regime Sweep Evaluation Protocol.

    Evaluates model across epsilon values spanning transport to diffusion limits.
    Reports I_rel_l2 and phi_rel_l2 vs epsilon.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon_values: List[float] = None,
        spatial_shape: tuple = (16, 16),
        n_omega: int = 8,
        n_groups: int = 1,
        benchmark_name: str = "mock",
        n_test_samples: int = 50,
        device: Optional[str] = None,
        batch_size: int = 4,
    ):
        self.model = model
        self.epsilon_values = epsilon_values or [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
        self.spatial_shape = spatial_shape
        self.n_omega = n_omega
        self.n_groups = n_groups
        self.benchmark_name = benchmark_name
        self.n_test = n_test_samples
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def run(self) -> Dict[str, Any]:
        self.model.eval()
        results = {}

        for eps in self.epsilon_values:
            logger.info(f"Regime sweep: evaluating at epsilon={eps:.4g}")

            ds = MockDataset(
                n_samples=self.n_test,
                spatial_shape=self.spatial_shape,
                n_omega=self.n_omega,
                n_groups=self.n_groups,
                benchmark_name=self.benchmark_name,
                epsilon_range=(eps * 0.99, eps * 1.01),  # narrow range around target
                seed=777,
            )
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                                collate_fn=collate_fn)

            metric_list = []
            with torch.no_grad():
                for batch in loader:
                    batch_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                    try:
                        pred = self.model(batch_dev)
                        m = compute_metrics(pred, batch_dev, self.device)
                        m.epsilon = eps
                        metric_list.append(m)
                    except Exception as e:
                        logger.warning(f"Forward failed at epsilon={eps}: {e}")

            summary = aggregate_metrics(metric_list)
            summary["epsilon"] = eps
            summary["log_epsilon"] = float(np.log10(eps))
            results[eps] = summary
            logger.info(f"  eps={eps:.4g}: I_rel_l2={summary.get('I_rel_l2', float('nan')):.4e}, "
                        f"phi_rel_l2={summary.get('phi_rel_l2', float('nan')):.4e}")

        return results

    def to_csv(self, results: Dict[str, Any], path: str):
        rows = list(results.values())
        if not rows:
            return
        fieldnames = sorted(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
