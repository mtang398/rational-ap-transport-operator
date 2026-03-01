"""
Tests for trainer: smoke test for a few epochs end-to-end.
"""
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MockDataset
from src.trainers.trainer import Trainer
from src.utils.seed import set_seed


def make_tiny_model(model_name="ap_micromacro", dim=2, n_groups=1):
    if model_name == "fno":
        from src.models.fno import FNOTransport
        return FNOTransport(dim=dim, n_groups=n_groups, n_params=2,
                            fno_channels=4, n_fno_blocks=1, n_modes=4, n_bc_faces=4)
    elif model_name == "deeponet":
        from src.models.deeponet import DeepONetTransport
        return DeepONetTransport(dim=dim, n_groups=n_groups, n_params=2,
                                 n_basis=8, branch_hidden=[8], trunk_hidden=[8], n_bc_faces=4)
    elif model_name == "ap_micromacro":
        from src.models.ap_micromacro import APMicroMacroTransport
        return APMicroMacroTransport(dim=dim, n_groups=n_groups, n_params=2,
                                     macro_channels=4, n_fno_blocks=1, n_modes=4,
                                     micro_latent_dim=8, micro_hidden=[8], n_bc_faces=4)
    raise ValueError(model_name)


@pytest.mark.parametrize("model_name", ["fno", "deeponet", "ap_micromacro"])
def test_smoke_train(model_name):
    """Smoke test: 2 epochs of training should not crash."""
    set_seed(42)

    train_ds = MockDataset(n_samples=8, spatial_shape=(8, 8), n_omega=4, n_groups=1, seed=42, solver_name="mock")
    val_ds = MockDataset(n_samples=4, spatial_shape=(8, 8), n_omega=4, n_groups=1, seed=123, solver_name="mock")

    model = make_tiny_model(model_name, dim=2, n_groups=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            lr=1e-3,
            n_epochs=2,
            batch_size=4,
            log_dir=tmpdir,
            run_name=f"smoke_{model_name}",
            val_every=1,
            log_every=5,
            device="cpu",
            num_workers=0,
            seed=42,
        )
        trainer.train()

        # Check checkpoint was saved
        ckpt_path = Path(tmpdir) / f"smoke_{model_name}" / "checkpoints" / "latest.pt"
        assert ckpt_path.exists(), f"Checkpoint not saved: {ckpt_path}"


def test_checkpoint_resume():
    """Test that training can be resumed from a checkpoint."""
    set_seed(0)
    train_ds = MockDataset(n_samples=8, spatial_shape=(8, 8), n_omega=4, seed=0, solver_name="mock")

    model = make_tiny_model("fno")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train 1 epoch
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            lr=1e-3,
            n_epochs=1,
            batch_size=4,
            log_dir=tmpdir,
            run_name="resume_test",
            device="cpu",
            num_workers=0,
        )
        trainer.train()

        ckpt = str(Path(tmpdir) / "resume_test" / "checkpoints" / "latest.pt")
        assert Path(ckpt).exists()

        # Resume for 1 more epoch
        model2 = make_tiny_model("fno")
        trainer2 = Trainer(
            model=model2,
            train_dataset=train_ds,
            lr=1e-3,
            n_epochs=2,  # 1 more after resume
            batch_size=4,
            log_dir=tmpdir,
            run_name="resume_test",
            resume_from=ckpt,
            device="cpu",
            num_workers=0,
        )
        # epoch should start at 1 (not 0)
        assert trainer2.epoch == 1
        trainer2.train()


def test_ema():
    """Test EMA runs without crashing."""
    set_seed(1)
    train_ds = MockDataset(n_samples=8, spatial_shape=(8, 8), n_omega=4, seed=1, solver_name="mock")
    model = make_tiny_model("fno")

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            lr=1e-3,
            n_epochs=2,
            batch_size=4,
            use_ema=True,
            ema_decay=0.99,
            log_dir=tmpdir,
            run_name="ema_test",
            device="cpu",
            num_workers=0,
        )
        trainer.train()
