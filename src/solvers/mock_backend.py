"""
Mock solver backend for transport problems.

Produces physically consistent synthetic labels without external solver dependencies.
Uses analytic approximations:
  - Diffusion limit (small epsilon): phi ~ solution of diffusion equation via simple iterative scheme
  - Transport limit (large epsilon): direct source + scattering approximation
  - Void regions: ray tracing approximation

This backend ensures the training pipeline runs immediately without installing OpenSn/OpenMC.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np

from ..data.schema import TransportSample, InputFields, QueryPoints, TargetFields

logger = logging.getLogger(__name__)


class MockSolver:
    """
    Mock transport solver using analytic approximations.

    Provides consistent labels for training (not physically exact).
    The accuracy is sufficient for pipeline validation and training,
    but not for rigorous benchmarking against experimental data.
    """

    def __init__(self, n_iter: int = 5, diffusion_threshold: float = 0.1):
        """
        Args:
            n_iter: number of synthetic sweep iterations
            diffusion_threshold: epsilon below which to use diffusion approximation
        """
        self.n_iter = n_iter
        self.diffusion_threshold = diffusion_threshold

    def solve(self, sample: TransportSample) -> TransportSample:
        """
        Compute approximate transport solution for a given sample.
        Returns a new TransportSample with updated targets.
        """
        inp = sample.inputs
        qry = sample.query
        epsilon = inp.params.get("epsilon", 1.0)
        g = inp.params.get("g", 0.0)

        if epsilon < self.diffusion_threshold:
            phi, J = self._solve_diffusion(inp, qry)
        else:
            phi, J = self._solve_transport_approx(inp, qry, epsilon)

        I = self._reconstruct_intensity(phi, J, qry, inp, g)

        # J stored = angular quadrature moment of I, consistent with model's J head.
        # (J from _solve_* is the Fick law gradient, which differs from sum(w*omega*I)
        #  by the P1 normalization factor; use the quadrature definition throughout.)
        w_om = qry.w_omega   # [Nw]
        omega = qry.omega    # [Nw, dim]
        J_quad = np.einsum('w,wd,nwg->ndg', w_om, omega, I).astype(np.float32)

        new_targets = TargetFields(
            I=I.astype(np.float32),
            phi=phi.astype(np.float32),
            J=J_quad,
            qois=sample.targets.qois,
        )

        import copy
        new_sample = copy.copy(sample)
        new_sample = TransportSample(
            inputs=sample.inputs,
            query=sample.query,
            targets=new_targets,
            sample_id=sample.sample_id,
        )
        return new_sample

    def _solve_diffusion(self, inp: InputFields, qry: QueryPoints) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diffusion approximation: -D * laplacian(phi) + sigma_a * phi = q
        Solved approximately using the analytic Green's function for each cell.
        Returns phi [Nx, G] and J [Nx, dim, G].
        """
        spatial_shape = inp.spatial_shape
        G = inp.n_groups
        dim = inp.dim
        Nx = int(np.prod(spatial_shape))

        sigma_a_flat = inp.sigma_a.reshape(Nx, G)
        sigma_s_flat = inp.sigma_s.reshape(Nx, G)
        q_flat = inp.q.reshape(Nx, G)

        # Diffusion coefficient D = 1 / (3 * (sigma_a + sigma_s * (1-g)))
        # For mock: use simple sigma_t
        sigma_t = sigma_a_flat + sigma_s_flat
        D = 1.0 / (3.0 * np.maximum(sigma_t, 1e-8))

        # Approximate solution: convolution with Green's function
        x = qry.x  # [Nx, dim]
        phi = np.zeros((Nx, G), dtype=np.float64)

        for j in range(Nx):
            if q_flat[j].sum() < 1e-15:
                continue
            dx = x - x[j]  # [Nx, dim]
            r2 = np.sum(dx**2, axis=-1, keepdims=True) + 1e-8  # [Nx, 1]
            r = np.sqrt(r2)
            # 3D Green's function: exp(-kappa*r) / (4*pi*D*r) where kappa=sqrt(sigma_a/D)
            kappa = np.sqrt(np.maximum(sigma_a_flat[j] / np.maximum(D[j], 1e-8), 1e-8))
            if dim == 3:
                G_r = np.exp(-kappa[np.newaxis] * r) / (4 * np.pi * D[j][np.newaxis] * r + 1e-30)
            else:  # 2D
                G_r = np.exp(-kappa[np.newaxis] * r) / (2 * np.pi * D[j][np.newaxis] * r + 1e-30)
            phi += G_r * q_flat[j][np.newaxis]  # crude approximation (not exact Green's fn)

        phi = np.clip(phi, 1e-15, None)

        # Current: J = -D * grad(phi) (approximate via neighbor differences)
        J = self._compute_gradient(phi, x, D)

        return phi.astype(np.float32), J.astype(np.float32)

    def _solve_transport_approx(self, inp: InputFields, qry: QueryPoints,
                                 epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple transport approximation for moderate-to-large epsilon.
        Uses first-collision source method: phi_0 = q / sigma_t (uncollided),
        then adds scattered component iteratively.
        """
        spatial_shape = inp.spatial_shape
        G = inp.n_groups
        dim = inp.dim
        Nx = int(np.prod(spatial_shape))

        sigma_a_flat = inp.sigma_a.reshape(Nx, G)
        sigma_s_flat = inp.sigma_s.reshape(Nx, G)
        q_flat = inp.q.reshape(Nx, G)
        sigma_t = sigma_a_flat + sigma_s_flat

        # Uncollided flux: q / sigma_t
        # In void/near-void cells (sigma_t ≈ 0) with nonzero source the expression
        # q/sigma_t diverges.  Physically, the flux in a void is set by streaming
        # geometry, not by the local optical depth.  We regularise with a floor
        # of 1/(domain_size) so that q/sigma_t ≤ q * L_domain (free-streaming).
        x = qry.x
        L_domain = float(np.ptp(x, axis=0).max()) if len(x) > 1 else 1.0
        sigma_t_floor = 1.0 / max(L_domain, 1.0)
        phi = q_flat / np.maximum(sigma_t, sigma_t_floor)

        # Scattering source iterations
        for _ in range(self.n_iter):
            scat_src = sigma_s_flat * phi / (4 * np.pi)  # isotropic scattering
            phi_new = (q_flat + scat_src) / np.maximum(sigma_t, sigma_t_floor)
            phi = phi_new

        phi = np.clip(phi, 1e-15, None)

        # Approximate current from diffusion relation
        x = qry.x
        D = 1.0 / (3.0 * np.maximum(sigma_t, 1e-8))
        J = self._compute_gradient(phi, x, D)

        return phi.astype(np.float32), J.astype(np.float32)

    def _compute_gradient(self, phi: np.ndarray, x: np.ndarray,
                          D: np.ndarray) -> np.ndarray:
        """
        Compute J = -D∇φ on a structured grid using numpy.gradient.

        D is flux-limited: D * |∇φ| ≤ φ to avoid blow-up in near-vacuum regions
        (relevant for void benchmarks like Kobayashi where σ_t ≈ 0).
        """
        Nx, G = phi.shape
        dim = x.shape[1]

        # Infer grid shape from point count and dimensionality
        n = round(Nx ** (1.0 / dim))
        try:
            spatial_shape = tuple([n] * dim)
            assert np.prod(spatial_shape) == Nx
        except AssertionError:
            # Non-cubic grid: fall back to zero current (isotropic)
            return np.zeros((Nx, dim, G), dtype=np.float64)

        J = np.zeros((Nx, dim, G), dtype=np.float64)
        # Compute cell spacing along each dimension
        for d in range(dim):
            coords_d = x[:, d].reshape(spatial_shape)
            # Use first column to get spacing (uniform grid assumed)
            idx = [0] * dim
            idx[d] = slice(None)
            coords_1d = coords_d[tuple(idx)]
            spacing = float(coords_1d[1] - coords_1d[0]) if len(coords_1d) > 1 else 1.0
            spacing = max(abs(spacing), 1e-30)

            for g in range(G):
                phi_grid = phi[:, g].reshape(spatial_shape)
                grad_d = np.gradient(phi_grid, spacing, axis=d).reshape(Nx)  # [Nx]
                # Flux-limited D: D * |∇φ| ≤ φ  (free-streaming limit)
                phi_safe = np.maximum(phi[:, g], 1e-30)
                D_lim = np.minimum(D[:, g], phi_safe / (np.abs(grad_d) + 1e-30))
                J[:, d, g] = -D_lim * grad_d

        return J

    def _reconstruct_intensity(self, phi: np.ndarray, J: np.ndarray,
                                qry: QueryPoints, inp: InputFields, g: float) -> np.ndarray:
        """
        Reconstruct I(x, omega) from moments using P1 angular reconstruction.

        2D: I(x,Ω) = φ/(2π) * [1 + (3·J·Ω) / (2π·φ)]
        3D: I(x,Ω) = φ/(4π) * [1 + (3·J·Ω) / (4π·φ)]

        With optional Henyey-Greenstein correction for anisotropic scattering.
        """
        Nx, G = phi.shape
        omega = qry.omega  # [Nw, dim]
        dim = inp.dim

        norm = 4 * np.pi if dim == 3 else 2 * np.pi

        # Isotropic baseline
        I = phi[:, np.newaxis, :] / norm  # [Nx, 1, G]  (broadcasts to [Nx, Nw, G])

        # P1 correction: I += (3/norm) * (J·Ω) / norm * phi  =>  multiply I_iso by correction
        if J is not None:
            # J: [Nx, dim, G],  omega: [Nw, dim]
            J_arr = J.reshape(Nx, dim, G)
            J_dot_omega = np.einsum('xdg,wd->xwg', J_arr, omega)       # [Nx, Nw, G]
            phi_safe = np.maximum(phi[:, np.newaxis, :], 1e-30)
            correction = 1.0 + (3.0 / norm) * J_dot_omega / phi_safe
            correction = np.maximum(correction, 0.0)
            I = I * correction

        # Henyey-Greenstein anisotropy correction for scattering problems
        if abs(g) > 1e-4:
            z_hat = np.zeros(omega.shape[1], dtype=np.float32)
            z_hat[-1] = 1.0
            mu = omega @ z_hat                                          # [Nw]
            hg = (1 - g**2) / (1 + g**2 - 2 * g * mu + 1e-8)**1.5    # [Nw]
            hg_norm = hg.mean() + 1e-8
            I = I * (hg[np.newaxis, :, np.newaxis] / hg_norm)

        return np.clip(I.astype(np.float32), 1e-30, None)

    def batch_solve(self, samples: list, show_progress: bool = True) -> list:
        """Solve a batch of samples."""
        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc="MockSolver") if show_progress else samples
        except ImportError:
            iterator = samples

        return [self.solve(s) for s in iterator]


class InputSpec:
    """
    Documented input/output specification for solver interfaces.
    Use this as a reference when implementing real solver wrappers.
    """

    INPUT_FIELDS = {
        "sigma_a": "absorption XS, shape [*spatial, G], units cm^-1",
        "sigma_s": "scattering XS, shape [*spatial, G], units cm^-1",
        "sigma_t": "total XS = sigma_a + sigma_s (derived)",
        "q": "isotropic source, shape [*spatial, G], units n/cm^3/s",
        "bc_type": "boundary condition type per face",
        "bc_inflow": "inflow flux per face",
        "geometry": "spatial grid coordinates [*spatial, dim]",
        "quadrature": "angular quadrature (omega directions + weights)",
        "n_groups": "number of energy groups",
        "epsilon": "Knudsen/mean-free-path parameter",
        "g": "scattering anisotropy (Henyey-Greenstein)",
        "t_start": "start time for time-dependent problems",
        "t_end": "end time",
        "dt": "time step",
    }

    OUTPUT_FIELDS = {
        "phi": "scalar flux, shape [*spatial, G]",
        "J": "current vector, shape [*spatial, dim, G]",
        "I": "angular flux, shape [*spatial, n_omega, G]",
        "k_eff": "effective multiplication factor (eigenvalue problems)",
        "detector_responses": "integrated fluxes over detector regions",
    }
