"""
Canonical data schema for transport neural operator experiments.

All samples follow this strict schema regardless of benchmark. The schema
supports multigroup cross sections, variable angular sets, time-dependent
queries, and structured boundary conditions.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import numpy as np
import torch

SCHEMA_VERSION = "1.0.0"


@dataclass
class BCSpec:
    """
    Structured boundary condition specification.

    bc_type: one of 'vacuum', 'inflow', 'reflective', 'mixed'
             for mixed BCs, per-face type is stored in type_per_face
    values:  dict mapping face label -> inflow intensity array (shape [Ngroups] or scalar)
    type_per_face: optional dict face_label -> bc_type string for mixed BCs
    """
    bc_type: str = "vacuum"
    values: Dict[str, np.ndarray] = field(default_factory=dict)
    type_per_face: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"bc_type": self.bc_type}
        d["values"] = {k: v.tolist() for k, v in self.values.items()}
        d["type_per_face"] = self.type_per_face
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BCSpec":
        values = {k: np.array(v, dtype=np.float32) for k, v in d.get("values", {}).items()}
        return cls(
            bc_type=d.get("bc_type", "vacuum"),
            values=values,
            type_per_face=d.get("type_per_face", {}),
        )

    def encode_tensor(self, n_faces: int = 6) -> torch.Tensor:
        """
        Encode BC spec as a fixed-size tensor for model input.
        Returns shape [n_faces, 2]: (bc_type_onehot_scalar, inflow_value).
        bc_type encoding: vacuum=0, inflow=1, reflective=2, mixed=3
        """
        type_map = {"vacuum": 0.0, "inflow": 1.0, "reflective": 2.0, "mixed": 3.0}
        face_labels = [f"face_{i}" for i in range(n_faces)]
        rows = []
        for face in face_labels:
            if self.type_per_face:
                t = type_map.get(self.type_per_face.get(face, self.bc_type), 0.0)
            else:
                t = type_map.get(self.bc_type, 0.0)
            v_arr = self.values.get(face, np.array([0.0], dtype=np.float32))
            v = float(v_arr.mean()) if len(v_arr) > 0 else 0.0
            rows.append([t / 3.0, v])  # normalize type to [0,1]
        return torch.tensor(rows, dtype=torch.float32)


@dataclass
class InputFields:
    """
    Physical input fields on a spatial grid.

    sigma_a: absorption cross section, shape [*spatial, G] where G=groups (G=1 for mono)
    sigma_s: scattering cross section, shape [*spatial, G] (or [*spatial, G, G] for full matrix)
    q:       internal source, shape [*spatial, G]
    extra_fields: additional named fields (e.g., fission source, density)
    bc:      boundary condition specification
    params:  scalar physics parameters: epsilon (Knudsen/mean-free-path), g (anisotropy), etc.
    metadata: benchmark name, dimensionality, group count, units
    """
    sigma_a: np.ndarray = field(default_factory=lambda: np.zeros((16, 16, 1), dtype=np.float32))
    sigma_s: np.ndarray = field(default_factory=lambda: np.zeros((16, 16, 1), dtype=np.float32))
    q: np.ndarray = field(default_factory=lambda: np.zeros((16, 16, 1), dtype=np.float32))
    extra_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    bc: BCSpec = field(default_factory=BCSpec)
    params: Dict[str, float] = field(default_factory=lambda: {"epsilon": 1.0, "g": 0.0})
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def spatial_shape(self) -> tuple:
        return self.sigma_a.shape[:-1]

    @property
    def n_groups(self) -> int:
        return self.sigma_a.shape[-1]

    @property
    def dim(self) -> int:
        return len(self.spatial_shape)


@dataclass
class QueryPoints:
    """
    Query locations for continuous operator evaluation.

    x: spatial query coordinates, shape [Nx, dim]
    omega: direction vectors (unit sphere), shape [Nw, dim]
    w_omega: quadrature weights for omega, shape [Nw], optional
    t: time values, shape [] (scalar) or [Nt] or [Nx] (per-spatial), optional
    """
    x: np.ndarray = field(default_factory=lambda: np.zeros((256, 2), dtype=np.float32))
    omega: np.ndarray = field(default_factory=lambda: np.zeros((8, 2), dtype=np.float32))
    w_omega: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None

    @property
    def n_spatial(self) -> int:
        return self.x.shape[0]

    @property
    def n_omega(self) -> int:
        return self.omega.shape[0]

    @property
    def is_time_dependent(self) -> bool:
        return self.t is not None


@dataclass
class TargetFields:
    """
    Ground-truth targets from reference solver.

    I:      specific intensity, shape [Nx, Nw, G] or [Nx, Nw, Nt, G]
    phi:    scalar flux (0th moment), shape [Nx, G]
    J:      current vector (1st moment), shape [Nx, dim, G]
    qois:   benchmark-specific quantities of interest (detector responses, etc.)
    """
    I: np.ndarray = field(default_factory=lambda: np.zeros((256, 8, 1), dtype=np.float32))
    phi: np.ndarray = field(default_factory=lambda: np.zeros((256, 1), dtype=np.float32))
    J: np.ndarray = field(default_factory=lambda: np.zeros((256, 2, 1), dtype=np.float32))
    qois: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class TransportSample:
    """
    Complete sample for the transport neural operator.
    This is the canonical unit for storage, dataloading, and model I/O.
    """
    inputs: InputFields
    query: QueryPoints
    targets: TargetFields
    sample_id: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to nested dict (JSON/Zarr-compatible, no tensors)."""
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "inputs": {
                "sigma_a": self.inputs.sigma_a,
                "sigma_s": self.inputs.sigma_s,
                "q": self.inputs.q,
                "extra_fields": {k: v for k, v in self.inputs.extra_fields.items()},
                "bc": self.inputs.bc.to_dict(),
                "params": self.inputs.params,
                "metadata": self.inputs.metadata,
            },
            "query": {
                "x": self.query.x,
                "omega": self.query.omega,
                "w_omega": self.query.w_omega,
                "t": self.query.t,
            },
            "targets": {
                "I": self.targets.I,
                "phi": self.targets.phi,
                "J": self.targets.J,
                "qois": {k: v for k, v in self.targets.qois.items()},
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransportSample":
        """Deserialize from nested dict."""
        inp = d["inputs"]
        qry = d["query"]
        tgt = d["targets"]

        inputs = InputFields(
            sigma_a=np.array(inp["sigma_a"], dtype=np.float32),
            sigma_s=np.array(inp["sigma_s"], dtype=np.float32),
            q=np.array(inp["q"], dtype=np.float32),
            extra_fields={k: np.array(v, dtype=np.float32) for k, v in inp.get("extra_fields", {}).items()},
            bc=BCSpec.from_dict(inp["bc"]),
            params={k: float(v) for k, v in inp.get("params", {}).items()},
            metadata=inp.get("metadata", {}),
        )
        query = QueryPoints(
            x=np.array(qry["x"], dtype=np.float32),
            omega=np.array(qry["omega"], dtype=np.float32),
            w_omega=np.array(qry["w_omega"], dtype=np.float32) if qry.get("w_omega") is not None else None,
            t=np.array(qry["t"], dtype=np.float32) if qry.get("t") is not None else None,
        )
        targets = TargetFields(
            I=np.array(tgt["I"], dtype=np.float32),
            phi=np.array(tgt["phi"], dtype=np.float32),
            J=np.array(tgt["J"], dtype=np.float32),
            qois={k: np.array(v, dtype=np.float32) for k, v in tgt.get("qois", {}).items()},
        )
        return cls(
            inputs=inputs,
            query=query,
            targets=targets,
            sample_id=d.get("sample_id", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def validate(self) -> List[str]:
        """Run shape/consistency checks. Returns list of error strings (empty = OK)."""
        errors = []
        Nx, dim = self.query.x.shape
        Nw = self.query.omega.shape[0]
        G = self.inputs.n_groups

        # Intensity shape
        expected_I = (Nx, Nw, G)
        if self.targets.I.shape != expected_I:
            errors.append(f"I shape {self.targets.I.shape} != expected {expected_I}")

        # phi shape
        if self.targets.phi.shape != (Nx, G):
            errors.append(f"phi shape {self.targets.phi.shape} != ({Nx},{G})")

        # J shape
        if self.targets.J.shape != (Nx, dim, G):
            errors.append(f"J shape {self.targets.J.shape} != ({Nx},{dim},{G})")

        # sigma shapes
        for name, arr in [("sigma_a", self.inputs.sigma_a), ("sigma_s", self.inputs.sigma_s), ("q", self.inputs.q)]:
            if arr.shape[-1] != G:
                errors.append(f"{name} last dim {arr.shape[-1]} != n_groups {G}")

        # quadrature weights
        if self.query.w_omega is not None and self.query.w_omega.shape != (Nw,):
            errors.append(f"w_omega shape {self.query.w_omega.shape} != ({Nw},)")

        return errors


def make_mock_sample(
    spatial_shape: tuple = (16, 16),
    n_omega: int = 8,
    n_groups: int = 1,
    benchmark_name: str = "mock",
    epsilon: float = 1.0,
    g: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> TransportSample:
    """
    Create a synthetic TransportSample with random but physically plausible fields.
    Used for testing and mock dataset generation.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    dim = len(spatial_shape)
    Nx = int(np.prod(spatial_shape))

    # Random cross sections (positive)
    sigma_a = rng.uniform(0.1, 1.0, (*spatial_shape, n_groups)).astype(np.float32)
    sigma_s = rng.uniform(0.05, 0.5, (*spatial_shape, n_groups)).astype(np.float32)
    # Ensure sigma_s < sigma_t for physical consistency
    sigma_s = np.minimum(sigma_s, sigma_a * 0.9)
    q = rng.uniform(0.0, 0.1, (*spatial_shape, n_groups)).astype(np.float32)

    # Isotropic direction samples on unit sphere
    if dim == 2:
        angles = rng.uniform(0, 2 * np.pi, n_omega).astype(np.float32)
        omega = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        w_omega = np.ones(n_omega, dtype=np.float32) * (2 * np.pi / n_omega)
    else:  # 3D
        phi = rng.uniform(0, 2 * np.pi, n_omega).astype(np.float32)
        costh = rng.uniform(-1, 1, n_omega).astype(np.float32)
        sinth = np.sqrt(1 - costh**2)
        omega = np.stack([sinth * np.cos(phi), sinth * np.sin(phi), costh], axis=-1)
        w_omega = np.ones(n_omega, dtype=np.float32) * (4 * np.pi / n_omega)

    # Spatial query: flattened grid
    axes = [np.linspace(0, 1, s, dtype=np.float32) for s in spatial_shape]
    grids = np.meshgrid(*axes, indexing="ij")
    x = np.stack([g_.ravel() for g_ in grids], axis=-1)  # [Nx, dim]

    # Simple analytic intensity: diffusion-like phi decays from source
    # phi ~ exp(-sigma_a_mean * |x - center|)
    sigma_a_flat = sigma_a.reshape(Nx, n_groups)  # [Nx, G]
    center = np.array([0.5] * dim, dtype=np.float32)
    dist = np.linalg.norm(x - center, axis=-1, keepdims=True)  # [Nx, 1]
    phi_vals = np.exp(-sigma_a_flat * dist) + 1e-6  # [Nx, G]

    # Anisotropic intensity: I(x,omega) ~ phi(x) * (1 + g * mu) / (4*pi or 2*pi)
    # mu = cos(angle) = omega . z_hat
    z_hat = np.zeros(dim, dtype=np.float32)
    z_hat[-1] = 1.0
    mu = (omega @ z_hat).reshape(1, n_omega)  # [1, Nw]
    phase = (1.0 + g * mu)  # [1, Nw]
    norm = 4 * np.pi if dim == 3 else 2 * np.pi
    I_vals = (phi_vals[:, np.newaxis, :] * phase[:, :, np.newaxis]) / norm  # [Nx, Nw, G]

    # Current: J = integral(omega * I) d_omega ~ -D * grad(phi)
    # Use simple finite-difference gradient approximation on the grid
    J_vals = np.zeros((Nx, dim, n_groups), dtype=np.float32)

    bc = BCSpec(bc_type="vacuum")

    inputs = InputFields(
        sigma_a=sigma_a,
        sigma_s=sigma_s,
        q=q,
        bc=bc,
        params={"epsilon": epsilon, "g": g},
        metadata={"benchmark_name": benchmark_name, "dim": dim, "group_count": n_groups},
    )
    query = QueryPoints(x=x, omega=omega, w_omega=w_omega)
    targets = TargetFields(I=I_vals, phi=phi_vals, J=J_vals)

    return TransportSample(inputs=inputs, query=query, targets=targets, sample_id=f"mock_{rng.integers(1000000)}")
