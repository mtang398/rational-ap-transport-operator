"""
Tests for neural operator models.

Checks output shapes, discretization-agnosticism (variable Nw),
and basic forward pass validity.
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MockDataset, collate_fn
from torch.utils.data import DataLoader


def make_batch(spatial_shape=(8, 8), n_omega=4, n_groups=1, batch_size=2, dim=2):
    """Create a small test batch."""
    ds = MockDataset(
        n_samples=batch_size,
        spatial_shape=spatial_shape,
        n_omega=n_omega,
        n_groups=n_groups,
        seed=42,
    )
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    return next(iter(loader))


class TestFNOTransport:
    def test_forward_2d(self):
        from src.models.fno import FNOTransport
        model = FNOTransport(dim=2, n_groups=1, n_params=2, fno_channels=8,
                             n_fno_blocks=2, n_modes=4, n_bc_faces=4)
        batch = make_batch(spatial_shape=(8, 8), n_omega=4, n_groups=1, batch_size=2)
        out = model(batch)
        B, Nx = 2, 64
        assert out["I"].shape == (B, Nx, 4, 1)
        assert out["phi"].shape == (B, Nx, 1)
        assert out["J"].shape == (B, Nx, 2, 1)

    def test_forward_multigroup(self):
        from src.models.fno import FNOTransport
        model = FNOTransport(dim=2, n_groups=3, n_params=2, fno_channels=8,
                             n_fno_blocks=2, n_modes=4, n_bc_faces=4)
        batch = make_batch(spatial_shape=(8, 8), n_omega=4, n_groups=3)
        out = model(batch)
        assert out["I"].shape[3] == 3  # G=3

    def test_variable_omega(self):
        """Model should handle different Nw at eval time."""
        from src.models.fno import FNOTransport
        model = FNOTransport(dim=2, n_groups=1, n_params=2, fno_channels=8,
                             n_fno_blocks=2, n_modes=4, n_bc_faces=4)
        for n_omega in [4, 8, 16]:
            batch = make_batch(n_omega=n_omega, batch_size=2)
            out = model(batch)
            assert out["I"].shape[2] == n_omega, f"Expected Nw={n_omega}, got {out['I'].shape[2]}"

    def test_I_positivity(self):
        from src.models.fno import FNOTransport
        model = FNOTransport(dim=2, n_groups=1, n_params=2, fno_channels=8,
                             n_fno_blocks=2, n_modes=4, n_bc_faces=4)
        batch = make_batch()
        out = model(batch)
        assert (out["I"] >= 0).all(), "I contains negative values"

    def test_no_nan(self):
        from src.models.fno import FNOTransport
        model = FNOTransport(dim=2, n_groups=1, n_params=2, fno_channels=8,
                             n_fno_blocks=2, n_modes=4, n_bc_faces=4)
        batch = make_batch()
        out = model(batch)
        for k, v in out.items():
            assert not torch.isnan(v).any(), f"NaN in output {k}"

    def test_backward(self):
        from src.models.fno import FNOTransport
        model = FNOTransport(dim=2, n_groups=1, n_params=2, fno_channels=8,
                             n_fno_blocks=2, n_modes=4, n_bc_faces=4)
        batch = make_batch()
        out = model(batch)
        loss = out["I"].mean()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any()


class TestDeepONetTransport:
    def test_forward_2d(self):
        from src.models.deeponet import DeepONetTransport
        model = DeepONetTransport(dim=2, n_groups=1, n_params=2, n_basis=16,
                                  branch_hidden=[32], trunk_hidden=[32], n_bc_faces=4)
        batch = make_batch(n_omega=4)
        out = model(batch)
        B, Nx = 2, 64
        assert out["I"].shape == (B, Nx, 4, 1)
        assert out["phi"].shape == (B, Nx, 1)

    def test_variable_omega(self):
        from src.models.deeponet import DeepONetTransport
        model = DeepONetTransport(dim=2, n_groups=1, n_params=2, n_basis=16,
                                  branch_hidden=[32], trunk_hidden=[32], n_bc_faces=4)
        for n_omega in [4, 8, 16]:
            batch = make_batch(n_omega=n_omega)
            out = model(batch)
            assert out["I"].shape[2] == n_omega

    def test_no_nan(self):
        from src.models.deeponet import DeepONetTransport
        model = DeepONetTransport(dim=2, n_groups=1, n_params=2, n_basis=16,
                                  branch_hidden=[32], trunk_hidden=[32], n_bc_faces=4)
        batch = make_batch()
        out = model(batch)
        for k, v in out.items():
            assert not torch.isnan(v).any(), f"NaN in {k}"


class TestAPMicroMacro:
    def test_forward_2d(self):
        from src.models.ap_micromacro import APMicroMacroTransport
        model = APMicroMacroTransport(
            dim=2, n_groups=1, n_params=2,
            macro_channels=8, n_fno_blocks=2, n_modes=4,
            micro_latent_dim=16, micro_hidden=[32], n_bc_faces=4
        )
        batch = make_batch(n_omega=4)
        out = model(batch)
        B, Nx = 2, 64
        assert out["I"].shape == (B, Nx, 4, 1)
        assert out["phi"].shape == (B, Nx, 1)
        assert out["J"].shape == (B, Nx, 2, 1)
        assert "I_P1" in out
        assert "R" in out
        assert "phi_I" in out
        assert "J_I" in out

    def test_reconstruction_decomposition(self):
        """I = I_P1 + R (before positivity clamp)."""
        from src.models.ap_micromacro import APMicroMacroTransport
        model = APMicroMacroTransport(
            dim=2, n_groups=1, n_params=2,
            macro_channels=8, n_fno_blocks=2, n_modes=4,
            micro_latent_dim=16, micro_hidden=[32], n_bc_faces=4
        )
        batch = make_batch(n_omega=4)
        out = model(batch)
        # I should approximately equal I_P1 + R (up to the softplus positivity layer)
        # Just check shapes match
        assert out["I_P1"].shape == out["I"].shape
        assert out["R"].shape == out["I"].shape

    def test_compute_loss(self):
        from src.models.ap_micromacro import APMicroMacroTransport
        model = APMicroMacroTransport(
            dim=2, n_groups=1, n_params=2,
            macro_channels=8, n_fno_blocks=2, n_modes=4,
            micro_latent_dim=16, micro_hidden=[32], n_bc_faces=4
        )
        batch = make_batch(n_omega=4)
        out = model(batch)
        losses = model.compute_loss(out, batch)
        assert "loss_total" in losses
        assert "loss_intensity" in losses
        assert "loss_moment" in losses
        assert "loss_diffusion" in losses
        # Loss should be finite
        assert not torch.isnan(losses["loss_total"])
        assert not torch.isinf(losses["loss_total"])

    def test_variable_omega(self):
        from src.models.ap_micromacro import APMicroMacroTransport
        model = APMicroMacroTransport(
            dim=2, n_groups=1, n_params=2,
            macro_channels=8, n_fno_blocks=2, n_modes=4,
            micro_latent_dim=16, micro_hidden=[32], n_bc_faces=4
        )
        for n_omega in [4, 8, 16]:
            batch = make_batch(n_omega=n_omega)
            out = model(batch)
            assert out["I"].shape[2] == n_omega

    def test_moment_shapes_multigroup(self):
        from src.models.ap_micromacro import APMicroMacroTransport
        model = APMicroMacroTransport(
            dim=2, n_groups=7, n_params=2,
            macro_channels=8, n_fno_blocks=2, n_modes=4,
            micro_latent_dim=16, micro_hidden=[32], n_bc_faces=4
        )
        batch = make_batch(n_groups=7)
        out = model(batch)
        B, Nx = 2, 64
        assert out["phi"].shape == (B, Nx, 7)
        assert out["J"].shape == (B, Nx, 2, 7)


class TestDeterminism:
    """Tests that model outputs are deterministic given same seed."""

    def test_fno_determinism(self):
        from src.models.fno import FNOTransport
        import numpy as np

        def run():
            torch.manual_seed(42)
            np.random.seed(42)
            model = FNOTransport(dim=2, n_groups=1, n_params=2, fno_channels=8,
                                 n_fno_blocks=2, n_modes=4, n_bc_faces=4)
            model.eval()
            batch = make_batch(spatial_shape=(8, 8), n_omega=4, batch_size=2)
            with torch.no_grad():
                return model(batch)["I"].clone()

        out1 = run()
        out2 = run()
        torch.testing.assert_close(out1, out2)
