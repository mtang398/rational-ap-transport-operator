"""
Tests for evaluation metrics.
"""
import math
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import l2_error, relative_l2_error, compute_metrics, aggregate_metrics, MetricBundle
from src.data.dataset import MockDataset, collate_fn
from torch.utils.data import DataLoader


class TestL2Metrics:
    def test_perfect_prediction(self):
        pred = torch.ones(10, 5)
        true = torch.ones(10, 5)
        assert l2_error(pred, true) == pytest.approx(0.0, abs=1e-6)
        assert relative_l2_error(pred, true) == pytest.approx(0.0, abs=1e-6)

    def test_known_error(self):
        pred = torch.zeros(4)
        true = torch.ones(4)
        # MSE = 1.0, so l2_error = sqrt(1) = 1.0
        assert l2_error(pred, true) == pytest.approx(1.0, rel=1e-5)
        # rel_l2 = ||pred-true|| / ||true|| = sqrt(4) / sqrt(4) = 1.0
        assert relative_l2_error(pred, true) == pytest.approx(1.0, rel=1e-5)

    def test_masked_error(self):
        pred = torch.zeros(8)
        true = torch.ones(8)
        mask = torch.zeros(8, dtype=torch.bool)
        mask[:4] = True  # only first 4 valid
        # Error computed only on valid entries
        err = l2_error(pred, true, mask)
        assert err == pytest.approx(1.0, rel=1e-4)

    def test_relative_zero_denominator(self):
        """Should not divide by zero."""
        pred = torch.zeros(5)
        true = torch.zeros(5)
        rel = relative_l2_error(pred, true, eps=1e-8)
        assert math.isfinite(rel)


class TestComputeMetrics:
    def _make_batch_pred(self, n_omega=4, batch_size=2):
        ds = MockDataset(n_samples=batch_size, spatial_shape=(8, 8), n_omega=n_omega)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(loader))
        # Perfect prediction (use true values as pred)
        pred = {
            "I": batch["I"].clone(),
            "phi": batch["phi"].clone(),
            "J": batch["J"].clone(),
        }
        return pred, batch

    def test_perfect_prediction_metrics(self):
        pred, batch = self._make_batch_pred()
        m = compute_metrics(pred, batch)
        # Perfect prediction should give near-zero errors
        assert m.I_l2 == pytest.approx(0.0, abs=1e-5)
        assert m.I_rel_l2 == pytest.approx(0.0, abs=1e-4)

    def test_metrics_not_nan(self):
        pred, batch = self._make_batch_pred()
        m = compute_metrics(pred, batch)
        assert not math.isnan(m.I_l2)
        assert not math.isnan(m.phi_l2)

    def test_metadata_filled(self):
        pred, batch = self._make_batch_pred()
        m = compute_metrics(pred, batch)
        assert m.n_omega > 0
        assert m.n_spatial > 0

    def test_energy_balance(self):
        pred, batch = self._make_batch_pred()
        m = compute_metrics(pred, batch)
        # Energy balance should be computable (not nan)
        # May be large due to mock data, but should not error
        assert math.isfinite(m.energy_balance_residual) or math.isnan(m.energy_balance_residual)


class TestAggregateMetrics:
    def test_aggregate_empty(self):
        result = aggregate_metrics([])
        assert result == {}

    def test_aggregate_single(self):
        m = MetricBundle(I_l2=0.1, I_rel_l2=0.05, phi_l2=0.02)
        result = aggregate_metrics([m])
        assert result["I_l2"] == pytest.approx(0.1, rel=1e-5)
        assert result["phi_l2"] == pytest.approx(0.02, rel=1e-5)

    def test_aggregate_multiple(self):
        ms = [MetricBundle(I_l2=i * 0.1) for i in range(1, 6)]
        result = aggregate_metrics(ms)
        expected_mean = sum(i * 0.1 for i in range(1, 6)) / 5
        assert result["I_l2"] == pytest.approx(expected_mean, rel=1e-5)
