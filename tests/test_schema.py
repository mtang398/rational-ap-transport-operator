"""
Tests for canonical data schema.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schema import (
    TransportSample, InputFields, QueryPoints, TargetFields,
    BCSpec, make_mock_sample, SCHEMA_VERSION
)


class TestBCSpec:
    def test_default(self):
        bc = BCSpec()
        assert bc.bc_type == "vacuum"
        assert bc.values == {}

    def test_encode_tensor(self):
        bc = BCSpec(bc_type="vacuum")
        t = bc.encode_tensor(n_faces=4)
        assert t.shape == (4, 2)

    def test_inflow_encoding(self):
        bc = BCSpec(bc_type="inflow", values={"face_0": np.array([1.0], dtype=np.float32)})
        t = bc.encode_tensor(n_faces=2)
        assert t.shape == (2, 2)
        # inflow type = 1.0 / 3.0
        assert abs(t[0, 0].item() - 1.0 / 3.0) < 1e-5

    def test_roundtrip(self):
        bc = BCSpec(bc_type="mixed",
                    values={"face_0": np.array([0.5])},
                    type_per_face={"face_0": "inflow", "face_1": "vacuum"})
        d = bc.to_dict()
        bc2 = BCSpec.from_dict(d)
        assert bc2.bc_type == "mixed"
        assert "face_0" in bc2.values
        assert bc2.type_per_face["face_0"] == "inflow"


class TestMockSample:
    @pytest.mark.parametrize("spatial_shape,n_omega,n_groups", [
        ((8, 8), 4, 1),
        ((8, 8), 8, 3),
        ((4, 4, 4), 8, 1),
    ])
    def test_make_mock_sample(self, spatial_shape, n_omega, n_groups):
        sample = make_mock_sample(spatial_shape=spatial_shape, n_omega=n_omega, n_groups=n_groups)
        assert sample.inputs.spatial_shape == spatial_shape
        assert sample.query.n_omega == n_omega
        assert sample.inputs.n_groups == n_groups

    def test_validate_passes(self):
        sample = make_mock_sample((8, 8), 4, 1)
        errors = sample.validate()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_shapes(self):
        sample = make_mock_sample((8, 8), 4, 2)
        Nx = 64
        assert sample.query.x.shape == (Nx, 2)
        assert sample.query.omega.shape == (4, 2)
        assert sample.targets.I.shape == (Nx, 4, 2)
        assert sample.targets.phi.shape == (Nx, 2)
        assert sample.targets.J.shape == (Nx, 2, 2)

    def test_3d_shapes(self):
        sample = make_mock_sample((4, 4, 4), 6, 1)
        Nx = 64
        assert sample.query.x.shape == (Nx, 3)
        assert sample.query.omega.shape == (6, 3)
        assert sample.targets.I.shape == (Nx, 6, 1)
        assert sample.targets.J.shape == (Nx, 3, 1)

    def test_positivity(self):
        """Intensity should be positive."""
        sample = make_mock_sample((8, 8), 8, 1)
        assert (sample.targets.I >= 0).all(), "I contains negative values"
        assert (sample.targets.phi >= 0).all(), "phi contains negative values"

    def test_serialization_roundtrip(self):
        sample = make_mock_sample((8, 8), 4, 1)
        d = sample.to_dict()
        sample2 = TransportSample.from_dict(d)
        np.testing.assert_allclose(sample.targets.I, sample2.targets.I, rtol=1e-5)
        np.testing.assert_allclose(sample.inputs.sigma_a, sample2.inputs.sigma_a, rtol=1e-5)
        assert sample2.sample_id == sample.sample_id

    def test_schema_version(self):
        sample = make_mock_sample()
        assert sample.schema_version == SCHEMA_VERSION


class TestTransportSampleValidation:
    def test_bad_I_shape(self):
        sample = make_mock_sample((8, 8), 4, 1)
        # Corrupt I shape
        sample.targets.I = np.zeros((64, 5, 1))  # wrong Nw
        errors = sample.validate()
        assert any("I shape" in e for e in errors)

    def test_bad_phi_shape(self):
        sample = make_mock_sample((8, 8), 4, 1)
        sample.targets.phi = np.zeros((32, 1))  # wrong Nx
        errors = sample.validate()
        assert any("phi" in e for e in errors)
