"""
Configuration dataclasses for training and evaluation.
These mirror the Hydra YAML configs and provide typed access.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class BenchmarkConfig:
    name: str = "c5g7"
    spatial_shape: List[int] = field(default_factory=lambda: [16, 16])
    n_omega_train: int = 8
    n_groups: int = 1
    n_samples_train: int = 200
    n_samples_val: int = 50
    n_samples_test: int = 50
    epsilon_range: List[float] = field(default_factory=lambda: [0.01, 1.0])
    solver: str = "auto"
    raw_dir: Optional[str] = None
    time_dependent: bool = False
    n_time: int = 10
    t_end: float = 1.0


@dataclass
class ModelConfig:
    name: str = "ap_micromacro"
    dim: int = 2
    n_groups: int = 1
    n_params: int = 2
    # FNO params
    fno_channels: int = 32
    n_fno_blocks: int = 4
    n_modes: int = 12
    # DeepONet params
    n_basis: int = 128
    # AP params
    macro_channels: int = 32
    micro_latent_dim: int = 64
    micro_hidden: List[int] = field(default_factory=lambda: [128, 128])
    # Common
    n_freq_x: int = 16
    n_freq_omega: int = 8
    activation: str = "gelu"
    time_dependent: bool = False
    n_bc_faces: int = 4
    lambda_moment: float = 1.0
    lambda_diffusion: float = 0.1


@dataclass
class TrainConfig:
    # Data
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    # Training
    n_epochs: int = 100
    warmup_epochs: int = 5
    batch_size: int = 8
    grad_clip: float = 1.0
    use_amp: bool = False
    use_ema: bool = False
    ema_decay: float = 0.999
    # Logging
    log_dir: str = "runs"
    run_name: str = "run"
    log_every: int = 10
    val_every: int = 1
    # Resume
    resume_from: Optional[str] = None
    # Device
    device: Optional[str] = None
    num_workers: int = 0
    seed: int = 42
    # Variable omega training
    resample_omega_range: Optional[List[int]] = None


@dataclass
class EvalConfig:
    checkpoint: str = "runs/latest.pt"
    protocol: str = "all"  # "omega_transfer", "sn_transfer" (alias), "resolution_transfer", "regime_sweep", "all"
    # SN transfer
    train_n_omega: int = 8
    test_n_omegas: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    # Resolution transfer
    resolution_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4])
    # Regime sweep
    epsilon_values: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 0.5, 1.0])
    # Common
    batch_size: int = 4
    n_test_samples: int = 50
    output_dir: str = "runs/eval"
    solver: str = "auto"
    device: Optional[str] = None
    seed: int = 42
