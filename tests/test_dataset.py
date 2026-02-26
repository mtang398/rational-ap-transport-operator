"""
Tests for dataset loading, collation, and Zarr roundtrip.
"""
import pytest
import tempfile
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schema import make_mock_sample
from src.data.dataset import MockDataset, TransportDataset, collate_fn, sample_to_tensor_dict
from src.data.io import ZarrDatasetWriter, ZarrDatasetReader


class TestMockDataset:
    def test_len(self):
        ds = MockDataset(n_samples=10, spatial_shape=(8, 8), n_omega=4, n_groups=1)
        assert len(ds) == 10

    def test_getitem_keys(self):
        ds = MockDataset(n_samples=5, spatial_shape=(8, 8), n_omega=4, n_groups=1)
        item = ds[0]
        for key in ["sigma_a", "sigma_s", "q", "x", "omega", "w_omega", "I", "phi", "J", "bc", "params"]:
            assert key in item, f"Missing key: {key}"

    def test_getitem_shapes(self):
        ds = MockDataset(n_samples=5, spatial_shape=(8, 8), n_omega=4, n_groups=1)
        item = ds[0]
        Nx = 64
        assert item["x"].shape == (Nx, 2)
        assert item["omega"].shape == (4, 2)
        assert item["I"].shape == (Nx, 4, 1)
        assert item["phi"].shape == (Nx, 1)
        assert item["J"].shape == (Nx, 2, 1)

    def test_multigroup_shapes(self):
        ds = MockDataset(n_samples=3, spatial_shape=(8, 8), n_omega=4, n_groups=7)
        item = ds[0]
        assert item["sigma_a"].shape == (64, 7)
        assert item["I"].shape == (64, 4, 7)

    def test_3d_shapes(self):
        ds = MockDataset(n_samples=3, spatial_shape=(4, 4, 4), n_omega=4, n_groups=1)
        item = ds[0]
        Nx = 64
        assert item["x"].shape == (Nx, 3)
        assert item["omega"].shape == (4, 3)
        assert item["I"].shape == (Nx, 4, 1)
        assert item["J"].shape == (Nx, 3, 1)


class TestCollateFn:
    def test_same_omega_size(self):
        """Batching with same Nw: no padding needed."""
        ds = MockDataset(n_samples=4, spatial_shape=(8, 8), n_omega=8)
        items = [ds[i] for i in range(4)]
        batch = collate_fn(items)
        B = 4
        Nx = 8 * 8  # 64
        assert batch["omega"].shape == (B, 8, 2)
        assert batch["I"].shape[1] == Nx

    def test_padding_different_omega(self):
        """Test that collate pads variable-Nw correctly."""
        # Create samples with different Nw
        s4 = make_mock_sample(n_omega=4)
        s8 = make_mock_sample(n_omega=8)
        items = [sample_to_tensor_dict(s4), sample_to_tensor_dict(s8)]
        batch = collate_fn(items)

        assert batch["omega"].shape[1] == 8  # padded to max
        assert batch["omega_mask"].shape == (2, 8)
        # First sample: 4 valid, 4 padded
        assert batch["omega_mask"][0, :4].all()
        assert not batch["omega_mask"][0, 4:].any()
        # Second sample: all 8 valid
        assert batch["omega_mask"][1, :].all()

    def test_mask_on_I(self):
        """Padded omega directions should have zero I."""
        s4 = make_mock_sample(n_omega=4)
        s8 = make_mock_sample(n_omega=8)
        items = [sample_to_tensor_dict(s4), sample_to_tensor_dict(s8)]
        batch = collate_fn(items)
        # I[0, :, 4:, :] should be 0 (padded directions)
        assert (batch["I"][0, :, 4:, :] == 0).all()


class TestZarrRoundtrip:
    @pytest.fixture(autouse=True)
    def require_zarr(self):
        pytest.importorskip("zarr", reason="zarr not installed")

    def test_write_read(self):
        """Write and read samples from Zarr store."""
        samples = [make_mock_sample((8, 8), n_omega=4, n_groups=1) for _ in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.zarr"
            writer = ZarrDatasetWriter(str(path), mode="w")
            for i, s in enumerate(samples):
                writer.write(s, idx=i)
            writer.flush(benchmark_name="test")

            reader = ZarrDatasetReader(str(path))
            assert len(reader) == 5

            for i in range(5):
                s_orig = samples[i]
                s_read = reader.read(i)
                np.testing.assert_allclose(
                    s_orig.targets.I, s_read.targets.I, rtol=1e-5,
                    err_msg=f"I mismatch for sample {i}"
                )
                np.testing.assert_allclose(
                    s_orig.inputs.sigma_a, s_read.inputs.sigma_a, rtol=1e-5
                )
                assert s_orig.sample_id == s_read.sample_id

    def test_read_metadata(self):
        samples = [make_mock_sample() for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta_test.zarr"
            writer = ZarrDatasetWriter(str(path), mode="w")
            for i, s in enumerate(samples):
                writer.write(s, idx=i)
            writer.flush(benchmark_name="my_benchmark")

            reader = ZarrDatasetReader(str(path))
            assert reader.metadata.get("benchmark_name") == "my_benchmark"
            assert reader.n_samples == 3

    def test_zarr_as_dataset(self):
        """Test loading from Zarr path through TransportDataset."""
        samples = [make_mock_sample((8, 8), n_omega=4) for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ds.zarr"
            writer = ZarrDatasetWriter(str(path), mode="w")
            for i, s in enumerate(samples):
                writer.write(s, idx=i)
            writer.flush()

            ds = TransportDataset(path)
            assert len(ds) == 4
            item = ds[0]
            assert "I" in item


class TestOmegaResampling:
    def test_resample_dataset(self):
        ds = MockDataset(
            n_samples=5, n_omega=8,
            resample_omega_range=(4, 12)
        )
        # Each item might have different Nw; check shapes are consistent
        item = ds[0]
        Nw = item["omega"].shape[0]
        assert item["I"].shape[1] == Nw  # I Nw matches omega Nw
