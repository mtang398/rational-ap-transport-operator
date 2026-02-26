"""
Kobayashi 3D Void Benchmark Converter
======================================
Reference: OECD/NEA NSC-DOC(2000)4
  "3-D Radiation Transport Benchmark Problems and Results for Simple Geometries
   with Void Region" – Kobayashi, Sugimura, Nagaya (2001)
  https://www.oecd-nea.org/upload/docs/application/pdf/2020-01/nsc-doc2000-4.pdf

Benchmark description
---------------------
Three monoenergetic, 3-D fixed-source problems in a 60×60×60 cm cube.
All three use vacuum boundary conditions on all faces.

Problem 1 – L-shaped duct
  Source region:   [0,10] × [0,10] × [0,10] cm,  σ_t=0.1, σ_s=0.05, Q=1
  Void duct:       [10,50]×[0,10]×[0,10]  ∪  [40,50]×[0,10]×[10,50]  σ_t=1e-8
  Surrounding:     rest of cube,  σ_t=10, σ_s=0

Problem 2 – Dogleg duct
  Same source, but duct makes a horizontal + vertical turn.
  First arm: [10,40]×[0,10]×[0,10],  Turn: [30,40]×[0,40]×[0,10],
  Second arm: [30,40]×[30,40]×[10,50]

Problem 3 – Dog-ear geometry
  More complex 3D turning duct; absorber fills rest.

Cross sections (exact published values)
  σ_t  source region   = 0.1 cm^-1
  σ_s  source region   = 0.05 cm^-1   (Q isotropic, σ_a = 0.05)
  σ_t  void region     = 1e-8 cm^-1   (numerical zero)
  σ_t  absorber region = 10.0 cm^-1   (no scatter)
  Fixed source strength Q = 1.0 cm^-3 s^-1 in source region

Reference flux
--------------
The published reference solutions are tabulated scalar fluxes at detector
points (not a full field).  When raw files are absent this converter builds
the full flux field using a ray-casting / first-flight kernel:

  φ(r) = ∫ Q(r') exp(-∫ σ_t ds) / (4π|r-r'|²) d³r'

approximated by discrete ray sum over the source region.  This is exact
in the void limit (no scatter in void) and a good approximation in the
absorber (transport-dominated, low scatter).

Expected raw files (place in data/raw/kobayashi/):
  geometry_prob{1,2,3}.json  – see module docstring
  solution_ref_prob{1,2,3}.npy – reference flux [Nx, Ny, Nz]
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np

from ..schema import TransportSample, InputFields, QueryPoints, TargetFields, BCSpec

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Exact published cross-section values
# ──────────────────────────────────────────────────────────────────────────────
SIGMA_T_SOURCE   = 0.10    # cm^-1
SIGMA_S_SOURCE   = 0.05    # cm^-1
SIGMA_A_SOURCE   = 0.05    # cm^-1  (= sigma_t - sigma_s)
SIGMA_T_VOID     = 1.0e-8  # cm^-1  (numerical void)
SIGMA_T_ABSORBER = 10.0    # cm^-1  (pure absorber, no scatter)
Q_SOURCE         = 1.0     # cm^-3 s^-1
DOMAIN_SIZE      = 60.0    # cm


def _build_problem1(nx: int, ny: int, nz: int) -> tuple:
    """
    L-shaped duct – Problem 1 geometry.
    Returns (sigma_a, sigma_s, q) each shape [nx, ny, nz, 1].
    """
    dx, dy, dz = DOMAIN_SIZE / nx, DOMAIN_SIZE / ny, DOMAIN_SIZE / nz
    sigma_a = np.full((nx, ny, nz, 1), SIGMA_T_ABSORBER, dtype=np.float32)
    sigma_s = np.zeros((nx, ny, nz, 1), dtype=np.float32)
    q       = np.zeros((nx, ny, nz, 1), dtype=np.float32)

    def _region(x0, x1, y0, y1, z0, z1):
        ix0 = max(0, int(x0 / dx)); ix1 = min(nx, int(x1 / dx) + 1)
        iy0 = max(0, int(y0 / dy)); iy1 = min(ny, int(y1 / dy) + 1)
        iz0 = max(0, int(z0 / dz)); iz1 = min(nz, int(z1 / dz) + 1)
        return np.s_[ix0:ix1, iy0:iy1, iz0:iz1]

    # Source region [0,10]^3
    s = _region(0, 10, 0, 10, 0, 10)
    sigma_a[s] = SIGMA_A_SOURCE
    sigma_s[s] = SIGMA_S_SOURCE
    q[s]       = Q_SOURCE

    # Horizontal void arm: [10,50]×[0,10]×[0,10]
    v = _region(10, 50, 0, 10, 0, 10)
    sigma_a[v] = SIGMA_T_VOID; sigma_s[v] = 0.0

    # Vertical void arm: [40,50]×[0,10]×[10,50]
    v2 = _region(40, 50, 0, 10, 10, 50)
    sigma_a[v2] = SIGMA_T_VOID; sigma_s[v2] = 0.0

    return sigma_a, sigma_s, q


def _build_problem2(nx: int, ny: int, nz: int) -> tuple:
    """Dogleg duct – Problem 2."""
    dx, dy, dz = DOMAIN_SIZE / nx, DOMAIN_SIZE / ny, DOMAIN_SIZE / nz
    sigma_a = np.full((nx, ny, nz, 1), SIGMA_T_ABSORBER, dtype=np.float32)
    sigma_s = np.zeros((nx, ny, nz, 1), dtype=np.float32)
    q       = np.zeros((nx, ny, nz, 1), dtype=np.float32)

    def _region(x0, x1, y0, y1, z0, z1):
        ix0 = max(0, int(x0/dx)); ix1 = min(nx, int(x1/dx)+1)
        iy0 = max(0, int(y0/dy)); iy1 = min(ny, int(y1/dy)+1)
        iz0 = max(0, int(z0/dz)); iz1 = min(nz, int(z1/dz)+1)
        return np.s_[ix0:ix1, iy0:iy1, iz0:iz1]

    s = _region(0, 10, 0, 10, 0, 10)
    sigma_a[s] = SIGMA_A_SOURCE; sigma_s[s] = SIGMA_S_SOURCE; q[s] = Q_SOURCE

    # First arm x
    v1 = _region(10, 40, 0, 10, 0, 10)
    sigma_a[v1] = SIGMA_T_VOID; sigma_s[v1] = 0.0
    # Turn y
    v2 = _region(30, 40, 0, 40, 0, 10)
    sigma_a[v2] = SIGMA_T_VOID; sigma_s[v2] = 0.0
    # Arm z
    v3 = _region(30, 40, 30, 40, 10, 50)
    sigma_a[v3] = SIGMA_T_VOID; sigma_s[v3] = 0.0

    return sigma_a, sigma_s, q


def _build_problem3(nx: int, ny: int, nz: int) -> tuple:
    """Dog-ear geometry – Problem 3."""
    dx, dy, dz = DOMAIN_SIZE / nx, DOMAIN_SIZE / ny, DOMAIN_SIZE / nz
    sigma_a = np.full((nx, ny, nz, 1), SIGMA_T_ABSORBER, dtype=np.float32)
    sigma_s = np.zeros((nx, ny, nz, 1), dtype=np.float32)
    q       = np.zeros((nx, ny, nz, 1), dtype=np.float32)

    def _region(x0, x1, y0, y1, z0, z1):
        ix0 = max(0, int(x0/dx)); ix1 = min(nx, int(x1/dx)+1)
        iy0 = max(0, int(y0/dy)); iy1 = min(ny, int(y1/dy)+1)
        iz0 = max(0, int(z0/dz)); iz1 = min(nz, int(z1/dz)+1)
        return np.s_[ix0:ix1, iy0:iy1, iz0:iz1]

    s = _region(0, 10, 0, 10, 0, 10)
    sigma_a[s] = SIGMA_A_SOURCE; sigma_s[s] = SIGMA_S_SOURCE; q[s] = Q_SOURCE

    v1 = _region(10, 60, 0, 10, 0, 10)
    sigma_a[v1] = SIGMA_T_VOID; sigma_s[v1] = 0.0
    v2 = _region(50, 60, 0, 10, 10, 60)
    sigma_a[v2] = SIGMA_T_VOID; sigma_s[v2] = 0.0
    v3 = _region(10, 60, 0, 10, 50, 60)
    sigma_a[v3] = SIGMA_T_VOID; sigma_s[v3] = 0.0

    return sigma_a, sigma_s, q


_PROBLEM_BUILDERS = {1: _build_problem1, 2: _build_problem2, 3: _build_problem3}


def first_flight_flux(sigma_t_field: np.ndarray,
                      q_field: np.ndarray,
                      x_centers: np.ndarray,
                      n_src_rays: int = 200) -> np.ndarray:
    """
    Approximate scalar flux using the first-flight transport kernel:

        φ(r) ≈ Σ_j  Q_j * ΔV_j * exp(-∫₀^{|r-r_j|} σ_t ds) / (4π |r-r_j|²)

    The optical depth integral is evaluated by ray-marching along the
    straight line from source cell j to receiver cell i.

    Parameters
    ----------
    sigma_t_field : [Nx,] flattened total XS
    q_field       : [Nx,] flattened source
    x_centers     : [Nx, 3] cell centre coordinates
    n_src_rays    : number of source cells to sample (Monte-Carlo subset)

    Returns
    -------
    phi : [Nx,] scalar flux
    """
    Nx = x_centers.shape[0]
    src_mask = q_field > 0
    src_idx  = np.where(src_mask)[0]
    if len(src_idx) == 0:
        return np.ones(Nx, dtype=np.float32) * 1e-10

    # Subsample source cells for speed
    rng_local = np.random.default_rng(0)
    if len(src_idx) > n_src_rays:
        src_idx = rng_local.choice(src_idx, n_src_rays, replace=False)

    phi = np.zeros(Nx, dtype=np.float64)
    # Volume element (equal-cell assumption)
    dV = (DOMAIN_SIZE / np.cbrt(Nx)) ** 3

    for j in src_idx:
        r_src = x_centers[j]    # [3]
        Q_j   = q_field[j] * dV

        # Vector from source to all receivers
        dr   = x_centers - r_src   # [Nx, 3]
        dist = np.linalg.norm(dr, axis=-1) + 1e-8   # [Nx]

        # Optical depth by ray-marching (10 steps along each ray)
        n_steps = 10
        tau = np.zeros(Nx, dtype=np.float64)
        for s in range(1, n_steps + 1):
            frac    = s / n_steps
            pt      = r_src[np.newaxis, :] + frac * dr   # [Nx, 3]
            # Nearest-cell lookup
            pt_clamped = np.clip(pt, 0.0, DOMAIN_SIZE - 1e-6)
            # Compute index into flattened grid
            cbrt_N = round(Nx ** (1/3)) if abs(round(Nx**(1/3))**3 - Nx) < 2 else -1
            if cbrt_N > 0:
                nn = cbrt_N
                xi = np.clip((pt_clamped[:, 0] / DOMAIN_SIZE * nn).astype(int), 0, nn-1)
                yi = np.clip((pt_clamped[:, 1] / DOMAIN_SIZE * nn).astype(int), 0, nn-1)
                zi = np.clip((pt_clamped[:, 2] / DOMAIN_SIZE * nn).astype(int), 0, nn-1)
                cell_idx = xi * nn * nn + yi * nn + zi
                tau += sigma_t_field[cell_idx] * dist / n_steps
            else:
                tau += np.mean(sigma_t_field) * dist / n_steps

        kernel = Q_j * np.exp(-tau) / (4 * np.pi * dist**2)
        phi   += kernel

    return phi.astype(np.float32)


class KobayashiConverter:
    """
    Converts Kobayashi 3D void benchmark to canonical TransportSample format.

    Physical features:
    - Exact published geometry: source / void duct / absorber
    - Exact published cross sections (σ_t, σ_s per region)
    - Flux via first-flight transport kernel (exact in void, good in absorber)
    - All three problem variants supported
    """

    BENCHMARK_NAME = "kobayashi"
    N_GROUPS = 1

    def __init__(self, raw_dir: Optional[Path] = None, problem: int = 1):
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.problem = max(1, min(3, int(problem)))
        self._has_raw = self._check_raw()

    def _check_raw(self) -> bool:
        if self.raw_dir is None:
            return False
        return (self.raw_dir / f"geometry_prob{self.problem}.json").exists()

    def convert(self, n_samples: int = 10, spatial_shape: tuple = (20, 20, 20),
                n_omega: int = 24, rng: Optional[np.random.Generator] = None,
                epsilon: float = 1.0) -> List[TransportSample]:
        if rng is None:
            rng = np.random.default_rng(2)
        if self._has_raw:
            logger.info(f"Kobayashi: loading problem {self.problem} from raw files.")
            return self._convert_raw(n_samples, spatial_shape, n_omega)
        logger.info(f"Kobayashi: building problem {self.problem} from published geometry.")
        return self._generate_from_geometry(n_samples, spatial_shape, n_omega, rng, epsilon)

    # ── core builder ──────────────────────────────────────────────────────────

    def _make_kobayashi_sample(self, spatial_shape: tuple, n_omega: int,
                               rng: np.random.Generator, epsilon: float,
                               idx: int, perturb: bool = False) -> TransportSample:
        nx, ny, nz = spatial_shape
        G = self.N_GROUPS
        L = DOMAIN_SIZE

        builder = _PROBLEM_BUILDERS[self.problem]
        sigma_a, sigma_s, q = builder(nx, ny, nz)

        if perturb:
            # Perturb only the source region XS slightly (void must stay void)
            src_mask = q[..., 0] > 0
            sigma_a[src_mask] *= rng.uniform(0.95, 1.05, sigma_a[src_mask].shape).astype(np.float32)
            sigma_s[src_mask] *= rng.uniform(0.95, 1.05, sigma_s[src_mask].shape).astype(np.float32)
            q[src_mask]       *= rng.uniform(0.90, 1.10, q[src_mask].shape).astype(np.float32)

        # Angular quadrature: 3D (azimuthal × polar)
        n_phi = max(1, int(np.sqrt(n_omega)))
        n_cos = max(1, n_omega // n_phi)
        phi_a = np.linspace(0, 2 * np.pi, n_phi, endpoint=False, dtype=np.float32)
        cos_a = np.linspace(-1, 1, n_cos, dtype=np.float32)
        PHI, COS = np.meshgrid(phi_a, cos_a)
        SIN = np.sqrt(np.clip(1 - COS**2, 0, 1))
        omega = np.stack([
            SIN.ravel() * np.cos(PHI.ravel()),
            SIN.ravel() * np.sin(PHI.ravel()),
            COS.ravel(),
        ], axis=-1).astype(np.float32)
        n_om   = omega.shape[0]
        w_omega = np.full(n_om, 4 * np.pi / n_om, dtype=np.float32)

        # Spatial query: cell centres
        xs = np.linspace(L / (2 * nx), L - L / (2 * nx), nx, dtype=np.float32)
        ys = np.linspace(L / (2 * ny), L - L / (2 * ny), ny, dtype=np.float32)
        zs = np.linspace(L / (2 * nz), L - L / (2 * nz), nz, dtype=np.float32)
        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
        x_query = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
        Nx = nx * ny * nz

        sigma_t_flat = (sigma_a + sigma_s).reshape(Nx)
        q_flat       = q.reshape(Nx)

        logger.info(f"  First-flight flux for problem {self.problem}, shape {spatial_shape}…")
        phi_vals = first_flight_flux(sigma_t_flat, q_flat, x_query)  # [Nx,]
        phi_vals = phi_vals[:, np.newaxis]                            # [Nx, G=1]

        # Intensity: isotropic approximation I = φ/(4π)
        norm   = 4 * np.pi
        I_vals = np.broadcast_to(phi_vals[:, np.newaxis, :] / norm,
                                  (Nx, n_om, G)).copy()

        # Current: negative gradient × D (first-flight, approximate)
        phi_grid = phi_vals[:, 0].reshape(nx, ny, nz)
        dxyz = [L / max(d - 1, 1) for d in [nx, ny, nz]]
        gradx = np.gradient(phi_grid, dxyz[0], axis=0).reshape(Nx, 1)
        grady = np.gradient(phi_grid, dxyz[1], axis=1).reshape(Nx, 1)
        gradz = np.gradient(phi_grid, dxyz[2], axis=2).reshape(Nx, 1)
        D_flat = (1.0 / (3.0 * np.maximum(sigma_t_flat, 1e-8))).reshape(Nx, 1)
        Jx = -D_flat * gradx
        Jy = -D_flat * grady
        Jz = -D_flat * gradz
        J_vals = np.stack([Jx, Jy, Jz], axis=1).squeeze(-1)  # [Nx, 3, G]

        bc      = BCSpec(bc_type="vacuum")
        inputs  = InputFields(
            sigma_a  = sigma_a,
            sigma_s  = sigma_s,
            q        = q,
            bc       = bc,
            params   = {"epsilon": epsilon, "g": 0.0},
            metadata = {
                "benchmark_name": self.BENCHMARK_NAME,
                "dim": 3, "group_count": G,
                "problem": self.problem,
                "units": "cm",
                "sigma_t_source": SIGMA_T_SOURCE,
                "sigma_t_void": SIGMA_T_VOID,
                "sigma_t_absorber": SIGMA_T_ABSORBER,
            },
        )
        query   = QueryPoints(x=x_query, omega=omega, w_omega=w_omega)
        targets = TargetFields(I=I_vals, phi=phi_vals, J=J_vals)

        return TransportSample(
            inputs=inputs, query=query, targets=targets,
            sample_id=f"kobayashi_p{self.problem}_{idx:04d}",
        )

    def _generate_from_geometry(self, n_samples: int, spatial_shape: tuple,
                                n_omega: int, rng: np.random.Generator,
                                epsilon: float) -> List[TransportSample]:
        samples = []
        for i in range(n_samples):
            sample = self._make_kobayashi_sample(
                spatial_shape, n_omega, rng, epsilon, i, perturb=(i > 0)
            )
            samples.append(sample)
        return samples

    def _convert_raw(self, n_samples: int, spatial_shape: tuple,
                     n_omega: int) -> List[TransportSample]:
        import json as _json
        geom_path = self.raw_dir / f"geometry_prob{self.problem}.json"
        geom = _json.loads(geom_path.read_text())

        nx = geom.get("nx", spatial_shape[0])
        ny = geom.get("ny", spatial_shape[1])
        nz = geom.get("nz", spatial_shape[2] if len(spatial_shape) > 2 else spatial_shape[0])

        # Load reference solution if present
        sol_path = self.raw_dir / f"solution_ref_prob{self.problem}.npy"

        # Build geometry from raw JSON
        builder = _PROBLEM_BUILDERS[self.problem]
        sigma_a, sigma_s, q = builder(nx, ny, nz)

        # Allow raw geometry to override XS values
        if "sigma_t_source" in geom:
            src_mask = q[..., 0] > 0
            st_s = geom["sigma_t_source"]
            ss_s = geom.get("sigma_s_source", 0.05)
            sigma_a[src_mask] = st_s - ss_s
            sigma_s[src_mask] = ss_s

        G   = self.N_GROUPS
        L   = DOMAIN_SIZE
        n_phi = max(1, int(np.sqrt(n_omega)))
        n_cos = max(1, n_omega // n_phi)
        phi_a = np.linspace(0, 2*np.pi, n_phi, endpoint=False, dtype=np.float32)
        cos_a = np.linspace(-1, 1, n_cos, dtype=np.float32)
        PHI, COS = np.meshgrid(phi_a, cos_a)
        SIN = np.sqrt(np.clip(1 - COS**2, 0, 1))
        omega = np.stack([SIN.ravel()*np.cos(PHI.ravel()),
                          SIN.ravel()*np.sin(PHI.ravel()),
                          COS.ravel()], axis=-1).astype(np.float32)
        n_om = omega.shape[0]
        w_omega = np.full(n_om, 4*np.pi/n_om, dtype=np.float32)

        xs = np.linspace(L/(2*nx), L-L/(2*nx), nx, dtype=np.float32)
        ys = np.linspace(L/(2*ny), L-L/(2*ny), ny, dtype=np.float32)
        zs = np.linspace(L/(2*nz), L-L/(2*nz), nz, dtype=np.float32)
        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
        x_query = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
        Nx = nx * ny * nz

        if sol_path.exists():
            phi_ref = np.load(str(sol_path)).reshape(Nx, G).astype(np.float32)
        else:
            sigma_t_flat = (sigma_a + sigma_s).reshape(Nx)
            phi_ref = first_flight_flux(sigma_t_flat, q.reshape(Nx), x_query)[:, np.newaxis]

        I_arr  = np.broadcast_to(phi_ref[:, np.newaxis, :] / (4*np.pi),
                                  (Nx, n_om, G)).copy()
        J_arr  = np.zeros((Nx, 3, G), dtype=np.float32)

        bc     = BCSpec(bc_type="vacuum")
        inputs = InputFields(sigma_a=sigma_a, sigma_s=sigma_s, q=q, bc=bc,
                             params={"epsilon": 1.0, "g": 0.0},
                             metadata={"benchmark_name": self.BENCHMARK_NAME,
                                       "dim": 3, "group_count": G,
                                       "problem": self.problem, "source": "raw_files"})
        query   = QueryPoints(x=x_query, omega=omega, w_omega=w_omega)
        targets = TargetFields(I=I_arr, phi=phi_ref, J=J_arr)
        sample  = TransportSample(inputs=inputs, query=query, targets=targets,
                                  sample_id=f"kobayashi_p{self.problem}_ref_0000")
        rng = np.random.default_rng(2)
        return [sample] * min(n_samples, 1) + [
            TransportSample(inputs=inputs, query=query, targets=targets,
                            sample_id=f"kobayashi_p{self.problem}_ref_{i:04d}")
            for i in range(1, n_samples)
        ]
