"""
PyTorch Dataset and collate_fn for transport samples.

Key features:
- Loads from Zarr or HDF5
- collate_fn pads variable Nw (angular directions) and provides omega_mask
- Supports in-memory caching
- Supports augmentation (random omega resampling for discretization-agnostic training)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor

from .schema import TransportSample, make_mock_sample
from .io import ZarrDatasetReader


class TransportDataset(Dataset):
    """
    PyTorch Dataset wrapping a collection of TransportSamples.

    Supports:
    - Loading from Zarr store (path)
    - Loading from list of in-memory TransportSamples
    - Optional transform/augmentation
    - Random omega resampling at load time (for SN-transfer training)
    """

    def __init__(
        self,
        source: Union[str, Path, List[TransportSample]],
        transform: Optional[Callable] = None,
        cache: bool = False,
        resample_omega: Optional[int] = None,
        resample_omega_range: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            source: path to Zarr store OR list of TransportSamples
            transform: optional callable applied to loaded sample dict
            cache: if True, load all samples into memory at init
            resample_omega: if set, resample omega to this fixed count at load time
            resample_omega_range: if set, sample a random count from this range per item
        """
        self.transform = transform
        self.resample_omega = resample_omega
        self.resample_omega_range = resample_omega_range

        if isinstance(source, (str, Path)):
            self.reader = ZarrDatasetReader(source)
            self.n_samples = len(self.reader)
            self._samples: Optional[List[TransportSample]] = None
            if cache:
                self._samples = [self.reader.read(i) for i in range(self.n_samples)]
        else:
            self.reader = None
            self._samples = list(source)
            self.n_samples = len(self._samples)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load sample
        if self._samples is not None:
            sample = self._samples[idx]
        else:
            sample = self.reader.read(idx)

        # Optional omega resampling (discretization-agnostic training)
        if self.resample_omega_range is not None:
            lo, hi = self.resample_omega_range
            n_omega = np.random.randint(lo, hi + 1)
            sample = resample_omega_directions(sample, n_omega)
        elif self.resample_omega is not None:
            sample = resample_omega_directions(sample, self.resample_omega)

        item = sample_to_tensor_dict(sample)

        if self.transform is not None:
            item = self.transform(item)

        return item


def sample_to_tensor_dict(sample: TransportSample) -> Dict[str, Any]:
    """Convert TransportSample to dict of tensors for batching."""
    inp = sample.inputs
    qry = sample.query
    tgt = sample.targets

    # Flatten spatial dims for fields -> [Nx, G]
    Nx = int(np.prod(inp.spatial_shape))
    G = inp.n_groups
    dim = inp.dim

    sigma_a_flat = torch.from_numpy(inp.sigma_a.reshape(Nx, G))
    sigma_s_flat = torch.from_numpy(inp.sigma_s.reshape(Nx, G))
    q_flat = torch.from_numpy(inp.q.reshape(Nx, G))

    # BC encoding: [n_faces, 2]
    bc_tensor = inp.bc.encode_tensor(n_faces=2 * dim)

    # Scalar params as tensor [num_params]
    param_keys = sorted(inp.params.keys())
    param_vals = torch.tensor([inp.params[k] for k in param_keys], dtype=torch.float32)

    # Query
    x = torch.from_numpy(qry.x)          # [Nx, dim]
    omega = torch.from_numpy(qry.omega)  # [Nw, dim_omega]
    Nw = qry.n_omega
    w_omega = torch.from_numpy(qry.w_omega) if qry.w_omega is not None else torch.ones(Nw) / Nw
    omega_mask = torch.ones(Nw, dtype=torch.bool)

    # Targets
    I = torch.from_numpy(tgt.I)        # [Nx, Nw, G]
    phi = torch.from_numpy(tgt.phi)    # [Nx, G]
    J = torch.from_numpy(tgt.J)        # [Nx, dim, G]

    item: Dict[str, Any] = {
        # Inputs
        "sigma_a": sigma_a_flat,   # [Nx, G]
        "sigma_s": sigma_s_flat,   # [Nx, G]
        "q": q_flat,               # [Nx, G]
        "bc": bc_tensor,           # [n_faces, 2]
        "params": param_vals,      # [P]
        "param_keys": param_keys,
        # Query
        "x": x,                    # [Nx, dim]
        "omega": omega,            # [Nw, d_omega]
        "w_omega": w_omega,        # [Nw]
        "omega_mask": omega_mask,  # [Nw]
        # Targets
        "I": I,                    # [Nx, Nw, G]
        "phi": phi,                # [Nx, G]
        "J": J,                    # [Nx, dim, G]
        # Metadata
        "sample_id": sample.sample_id,
        "metadata": sample.inputs.metadata,
        "spatial_shape": list(inp.spatial_shape),
        "n_groups": G,
        "dim": dim,
    }

    # Optional time
    if qry.t is not None:
        item["t"] = torch.from_numpy(qry.t)

    # Optional QoIs
    for k, v in tgt.qois.items():
        item[f"qoi_{k}"] = torch.from_numpy(v)

    return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a batch of sample dicts.

    Variable Nw is handled by padding omega, w_omega, I, omega_mask to max Nw.
    All spatial sizes are assumed equal within a batch (use same spatial_shape per batch).
    """
    if len(batch) == 0:
        return {}

    # Find max Nw
    max_Nw = max(item["omega"].shape[0] for item in batch)
    B = len(batch)

    def _stack(key: str) -> Tensor:
        return torch.stack([item[key] for item in batch], dim=0)

    def _pad_omega_dim(t: Tensor, max_Nw: int, pad_val: float = 0.0) -> Tensor:
        """Pad tensor along dim that corresponds to Nw."""
        Nw = t.shape[0]
        if Nw == max_Nw:
            return t
        pad_shape = list(t.shape)
        pad_shape[0] = max_Nw - Nw
        return torch.cat([t, torch.full(pad_shape, pad_val, dtype=t.dtype)], dim=0)

    # Pad omega-related tensors
    omega_padded = torch.stack([_pad_omega_dim(item["omega"], max_Nw) for item in batch])  # [B, max_Nw, d]
    w_omega_padded = torch.stack([_pad_omega_dim(item["w_omega"], max_Nw, 0.0) for item in batch])  # [B, max_Nw]
    omega_mask_padded = torch.stack([
        _pad_omega_dim(item["omega_mask"].float(), max_Nw, 0.0).bool()
        for item in batch
    ])  # [B, max_Nw]

    # Pad I: [Nx, Nw, G] -> pad Nw dim
    I_padded = torch.stack([
        _pad_omega_dim(item["I"].permute(1, 0, 2), max_Nw).permute(1, 0, 2)
        for item in batch
    ])  # [B, Nx, max_Nw, G]

    out: Dict[str, Any] = {
        "sigma_a": _stack("sigma_a"),    # [B, Nx, G]
        "sigma_s": _stack("sigma_s"),
        "q": _stack("q"),
        "bc": _stack("bc"),              # [B, n_faces, 2]
        "params": _stack("params"),      # [B, P]
        "x": _stack("x"),                # [B, Nx, dim]
        "omega": omega_padded,           # [B, max_Nw, d]
        "w_omega": w_omega_padded,       # [B, max_Nw]
        "omega_mask": omega_mask_padded, # [B, max_Nw]
        "I": I_padded,                   # [B, Nx, max_Nw, G]
        "phi": _stack("phi"),            # [B, Nx, G]
        "J": _stack("J"),                # [B, Nx, dim, G]
        "param_keys": batch[0]["param_keys"],
        "n_groups": batch[0]["n_groups"],
        "dim": batch[0]["dim"],
        "spatial_shape": batch[0]["spatial_shape"],
    }

    # Optional time
    if "t" in batch[0]:
        out["t"] = _stack("t")

    # QoIs
    for key in batch[0]:
        if key.startswith("qoi_"):
            out[key] = _stack(key)

    return out


def resample_omega_directions(
    sample: TransportSample,
    n_omega_new: int,
    rng: Optional[np.random.Generator] = None,
    random: bool = False,
) -> TransportSample:
    """
    Resample angular directions and recompute I at the new omega set.

    By default (random=False) uses the same regular-grid convention as the
    C5G7/Kobayashi converters (``np.linspace`` angles), so that SN-transfer
    evaluation tests generalisation across *different SN orders* rather than
    across random quadrature sets that the model was never trained on.

    When random=True a random quadrature is drawn from ``rng``.  Use this only
    for data-augmentation during training (so the model truly learns
    discretisation-agnosticism) — NOT for the SN-transfer evaluation protocol.

    Uses the P1 approximation with the sample's stored J (current) so that
    real-MC-derived angular structure is preserved:

        I(x, Ω) = φ/(norm) * max(0, 1 + (3/norm) * J·Ω / φ)

    where norm = 2π (2D) or 4π (3D).  If J is all-zero (e.g. pure isotropic
    mock data), this reduces to the isotropic formula φ/norm.
    """
    from .schema import QueryPoints, TargetFields
    import copy

    if rng is None:
        rng = np.random.default_rng()

    dim = sample.inputs.dim
    G = sample.inputs.n_groups
    Nx = sample.query.n_spatial

    if dim == 2:
        if random:
            # Random quadrature — use for training augmentation only
            offset = rng.uniform(0, 2 * np.pi / n_omega_new)
            angles = (np.linspace(0, 2 * np.pi, n_omega_new, endpoint=False) + offset).astype(np.float32)
        else:
            # Regular SN grid — matches converter convention (linspace)
            angles = np.linspace(0, 2 * np.pi, n_omega_new, endpoint=False, dtype=np.float32)
        omega_new = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        w_new = np.ones(n_omega_new, dtype=np.float32) * (2 * np.pi / n_omega_new)
    else:
        if random:
            phi_ang = rng.uniform(0, 2 * np.pi, n_omega_new).astype(np.float32)
            costh = rng.uniform(-1, 1, n_omega_new).astype(np.float32)
        else:
            # Lebedev-like regular azimuthal / Gauss-Legendre polar for 3D
            phi_ang = np.linspace(0, 2 * np.pi, n_omega_new, endpoint=False, dtype=np.float32)
            # Uniform in cos(theta) gives equal-area slices
            costh = np.linspace(-1 + 1.0 / n_omega_new, 1 - 1.0 / n_omega_new, n_omega_new, dtype=np.float32)
        sinth = np.sqrt(np.maximum(1 - costh**2, 0.0))
        omega_new = np.stack([sinth * np.cos(phi_ang), sinth * np.sin(phi_ang), costh], axis=-1)
        w_new = np.ones(n_omega_new, dtype=np.float32) * (4 * np.pi / n_omega_new)

    # P1 intensity: use stored J so real MC angular structure is retained.
    # J: [Nx, dim, G], omega_new: [Nw_new, dim]
    phi_vals = sample.targets.phi          # [Nx, G]
    J_vals   = sample.targets.J            # [Nx, dim, G]
    norm = 4 * np.pi if dim == 3 else 2 * np.pi

    phi_safe = np.maximum(phi_vals, 1e-30)                                # [Nx, G]
    JdotOmega = np.einsum('xdg,wd->xwg', J_vals, omega_new)              # [Nx, Nw_new, G]
    correction = 1.0 + (3.0 / norm) * JdotOmega / phi_safe[:, np.newaxis, :]
    correction = np.maximum(correction, 0.0)
    I_new = (phi_vals[:, np.newaxis, :] / norm * correction).astype(np.float32)  # [Nx, Nw_new, G]

    new_query = copy.copy(sample.query)
    new_query = QueryPoints(
        x=sample.query.x,
        omega=omega_new,
        w_omega=w_new,
        t=sample.query.t,
    )
    new_targets = TargetFields(
        I=I_new,
        phi=sample.targets.phi,
        J=sample.targets.J,
        qois=sample.targets.qois,
    )

    import copy as _copy
    new_sample = _copy.copy(sample)
    new_sample = TransportSample(
        inputs=sample.inputs,
        query=new_query,
        targets=new_targets,
        sample_id=sample.sample_id,
    )
    return new_sample


class SolverDataset(TransportDataset):
    """
    In-memory dataset generated on-the-fly using the converter + solver pipeline.

    Tries to use the best available real solver (OpenMC / OpenSn) for the
    given benchmark.  Falls back to the analytic MockSolver only when no real
    solver binary is detected, logging a WARNING in that case.

    This replaces the old MockDataset which always used analytic approximations
    regardless of what solvers were installed.
    """

    def __init__(
        self,
        n_samples: int = 100,
        spatial_shape: tuple = (16, 16),
        n_omega: int = 8,
        n_groups: int = 1,
        benchmark_name: str = "c5g7",
        epsilon_range: Tuple[float, float] = (0.01, 1.0),
        seed: int = 42,
        resample_omega_range: Optional[Tuple[int, int]] = None,
        solver_name: str = "auto",
        xs_perturb: float = 0.10,
    ):
        import logging as _logging
        _log = _logging.getLogger(__name__)

        from ..solvers import get_solver, detect_best_solver
        from .converters import (
            C5G7Converter, Pinte2009Converter,
        )

        converter_map = {
            "c5g7":      C5G7Converter,
            "pinte2009": Pinte2009Converter,
        }

        rng = np.random.default_rng(seed)
        resolved = detect_best_solver(benchmark_name) if solver_name == "auto" else solver_name
        ConverterCls = converter_map.get(benchmark_name)

        if ConverterCls is not None and resolved != "mock":
            # Use real converter + solver
            solver = get_solver(resolved, benchmark=benchmark_name, fallback=True)
            _log.info(
                f"SolverDataset [{benchmark_name}]: generating {n_samples} samples "
                f"with {resolved} solver (shape={spatial_shape})"
            )
            conv = ConverterCls()
            eps_vals = rng.uniform(*epsilon_range, size=n_samples).tolist()
            samples = []
            for i, eps in enumerate(eps_vals):
                s_list = conv.convert(
                    n_samples=1,
                    spatial_shape=spatial_shape,
                    n_omega=n_omega,
                    rng=rng,
                    epsilon=float(eps),
                    xs_perturb=xs_perturb,
                )
                samples.extend(solver.batch_solve(s_list, show_progress=False))
        else:
            # No real solver available — fall back to analytic MockSolver
            _log.warning(
                f"SolverDataset [{benchmark_name}]: no real solver available "
                f"(resolved='{resolved}') — using analytic MockSolver approximation. "
                "Install OpenMC or OpenSn for physically accurate labels."
            )
            solver = get_solver("mock")
            samples = []
            for i in range(n_samples):
                eps = float(rng.uniform(*epsilon_range))
                g = float(rng.uniform(-0.5, 0.9))
                s = make_mock_sample(
                    spatial_shape=spatial_shape,
                    n_omega=n_omega,
                    n_groups=n_groups,
                    benchmark_name=benchmark_name,
                    epsilon=eps,
                    g=g,
                    rng=rng,
                )
                samples.append(solver.solve(s))

        super().__init__(
            source=samples,
            resample_omega_range=resample_omega_range,
        )


# Backward-compatible alias — existing code that imports MockDataset continues to
# work, but now automatically prefers real solvers.
MockDataset = SolverDataset
