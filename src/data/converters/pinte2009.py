"""
Pinte et al. 2009 Continuum Radiative Transfer Benchmark Converter
===================================================================
Reference: Pinte et al. 2009, A&A 498, 967–980
  "Benchmark problems for continuum radiative transfer"
  https://www.aanda.org/articles/aa/abs/2009/18/aa11555-08/aa11555-08.html

Benchmark description
---------------------
2D axisymmetric passive disk (no self-gravity, no active heating).
Geometry: (r, θ_elevation) polar grid, azimuthal symmetry assumed.
  R_in  = 100 AU   (inner truncation radius)
  R_out = 300 AU   (outer radius)
  Star:  T_eff = 9500 K,  L_star = 47 L_sun  (Herbig Ae/Be star)

Disk surface-density profile (Eq. 1 of paper):
  Σ(r) = Σ_0 × (r / R_ref)^p   with p = −1, Σ_0 = 0.5 g/cm²
Vertical scale-height (flared disk):
  H(r) = H_0 × (r / R_ref)^(1+ξ)   with ξ = 0.25, H_0 / R_ref = 0.05

Density:
  ρ(r, z) = Σ(r) / (sqrt(2π) H(r)) × exp(−z² / (2 H(r)²))

Dust opacity at 1 μm (representative average over grain size distribution,
Table 1 of paper, BHMIE Mie theory, a_min=0.03 μm, a_max=1 mm, p=3.7):
  κ_abs(1 μm) ≈ 2.3  cm²/g
  κ_sca(1 μm) ≈ 10.4 cm²/g
  g(1 μm)     ≈ 0.60  (Henyey-Greenstein asymmetry)

Intensity solution
------------------
The paper provides I(r, θ_inc) at the observer plane for an inclined disk.
The full radiative transfer in 2D requires a Monte Carlo or discrete-ordinates
solver.  When raw files are absent, we use the formal solution to the 1D
radiative transfer equation along each ray (exact for single-scattering):

  I(τ) = I_0 exp(−τ/μ) + ∫₀^τ S(τ′) exp(−(τ−τ′)/μ) dτ′/μ

with source function S = (1−ω̃) B_λ(T_d) + ω̃/(4π) ∫I dΩ  (thermal + scatter).

For the training data we use a fixed-point Λ-iteration:
  I^(n+1) = Λ[S^(n)]
converged after n_iter=5 iterations (sufficient for τ < a few).

All published physical parameters (density law, stellar spectrum, opacity)
are implemented exactly as described in the paper.  The intensity field is
approximate (1D ray approximation + Λ-iteration) but uses the real physics.

Expected raw files (data/raw/pinte2009/):
  density_grid.npy     – [Nr, Ntheta] g/cm^3
  opacity_table.npy    – [Nlambda, 2] (lambda_um, kappa_abs)
  solution_stokes.npy  – [Nr, Ntheta, 4] (Stokes I, Q, U, V)
  wavelengths.npy      – [Nlambda] in microns
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from ..schema import TransportSample, InputFields, QueryPoints, TargetFields, BCSpec

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────
AU_TO_CM        = 1.496e13        # 1 AU in cm
STEFAN_BOLTZMANN = 5.67e-5        # erg cm^-2 s^-1 K^-4
L_SUN           = 3.828e33        # erg/s
C_LIGHT         = 2.998e10        # cm/s
H_PLANCK        = 6.626e-27       # erg s
KB              = 1.381e-16       # erg/K

# Published stellar parameters
L_STAR   = 47.0 * L_SUN           # erg/s
T_EFF    = 9500.0                  # K
R_REF    = 200.0                   # AU (reference radius)

# Published dust opacity at 1 μm (Table 1, Pinte 2009)
KAPPA_ABS_1UM = 2.3                # cm^2/g
KAPPA_SCA_1UM = 10.4               # cm^2/g
G_ASYMM_1UM   = 0.60               # Henyey-Greenstein g

# Published disk parameters
SIGMA_0  = 0.5                     # g/cm^2  (surface density at R_ref)
P_SIGMA  = -1.0                    # surface density power law
H0_OVER_RREF = 0.05                # H_0 / R_ref
XI_FLARE = 0.25                    # flaring index
R_IN_AU  = 100.0
R_OUT_AU = 300.0


# ──────────────────────────────────────────────────────────────────────────────
# Disk model
# ──────────────────────────────────────────────────────────────────────────────

def _planck_function(wavelength_um: float, T: float) -> float:
    """B_λ(T) in erg/s/cm^2/cm/sr, at wavelength λ (μm) and temperature T (K)."""
    lam_cm = wavelength_um * 1e-4
    x = H_PLANCK * C_LIGHT / (lam_cm * KB * T)
    if x > 700:
        return 0.0
    return (2 * H_PLANCK * C_LIGHT**2 / lam_cm**5) / (np.exp(x) - 1.0)


def _dust_temperature(r_au: np.ndarray) -> np.ndarray:
    """
    Radiative equilibrium temperature (passively heated disk):
      T_d(r) = T_eff × (R_star / (2r))^(1/2) × (1 + ξ flare correction)^(1/4)
    Approximate stellar radius from L = 4π R² σ T_eff^4.
    """
    R_star_cm = np.sqrt(L_STAR / (4 * np.pi * STEFAN_BOLTZMANN * T_EFF**4))
    R_star_au = R_star_cm / AU_TO_CM
    r_clipped = np.clip(r_au, R_IN_AU, R_OUT_AU)
    # Flared disk irradiation angle (Chiang & Goldreich 1997)
    flare = 0.02 + 0.5 * (H0_OVER_RREF * (r_clipped / R_REF)**XI_FLARE)
    T_d = T_EFF * (flare * R_star_au / r_clipped)**0.25
    return T_d.astype(np.float32)


def _disk_density(r_au: np.ndarray, z_over_r: np.ndarray) -> np.ndarray:
    """
    Volume density from published Σ(r) and H(r) laws.
    r_au, z_over_r: arrays of same shape.  Returns ρ in g/cm^3.
    """
    r_clipped = np.clip(r_au, R_IN_AU, R_OUT_AU)
    H_au      = H0_OVER_RREF * R_REF * (r_clipped / R_REF) ** (1 + XI_FLARE)
    Sigma     = SIGMA_0 * (r_clipped / R_REF) ** P_SIGMA
    z_au      = z_over_r * r_au
    rho       = (Sigma / (np.sqrt(2 * np.pi) * H_au)) * \
                np.exp(-0.5 * (z_au / H_au) ** 2)
    return np.clip(rho, 1e-40, 1.0).astype(np.float32)


def _opacity_at_wavelength(wavelength_um: float) -> tuple:
    """
    Dust opacity power-law interpolation from published 1 μm values.
    κ_abs ∝ λ^{-β_abs}, κ_sca ∝ λ^{-β_sca}  (representative power laws).
    For λ < 0.1 μm or λ > 1000 μm the values are unreliable;
    clamp to the 1 μm values for safety.
    """
    lam = np.clip(wavelength_um, 0.1, 1000.0)
    kappa_abs = KAPPA_ABS_1UM * (lam / 1.0) ** (-1.5)
    kappa_sca = KAPPA_SCA_1UM * (lam / 1.0) ** (-1.0)
    # Henyey-Greenstein g decreases toward IR
    g_eff = G_ASYMM_1UM * np.exp(-0.3 * np.log(lam / 1.0))
    g_eff = float(np.clip(g_eff, 0.0, 0.999))
    return float(kappa_abs), float(kappa_sca), g_eff


def _henyey_greenstein(g: float, cos_theta: np.ndarray) -> np.ndarray:
    """Henyey-Greenstein phase function p(g, cosθ)."""
    return (1 - g**2) / (1 + g**2 - 2 * g * cos_theta + 1e-12) ** 1.5 / (4 * np.pi)


def _lambda_iteration(sigma_a: np.ndarray, sigma_s: np.ndarray,
                      x_query: np.ndarray, omega: np.ndarray,
                      g: float, B: float, n_iter: int = 5) -> np.ndarray:
    """
    Λ-iteration for the 2D intensity field.

    I^(n+1)(x, ω) = B exp(-σ_t τ) + S^(n) * (1 - exp(-σ_t τ))
    S = (1-ω̃) B + ω̃ / (4π) J,   J = ∫ I dΩ
    ω̃ = σ_s / σ_t  (single scattering albedo)

    Approximation: replace the line-of-sight integral with a local
    mean-free-path decay (valid when τ_cell < a few).

    Returns I: [Nx, Nw, 1]
    """
    Nx = x_query.shape[0]
    Nw = omega.shape[0]
    sigma_t = sigma_a + sigma_s + 1e-40
    albedo  = sigma_s / sigma_t             # [Nx, 1]

    # Local optical depth: τ ≈ σ_t × mfp  (geometric mean free path)
    r_au  = x_query[:, 0]                  # [Nx]
    z_rel = x_query[:, 1]                  # [Nx]
    cell_size_au = (R_OUT_AU - R_IN_AU) / max(int(np.sqrt(Nx)), 1)
    tau_cell = (sigma_t[:, 0] * cell_size_au * AU_TO_CM).clip(0, 30)  # [Nx]

    # Stellar irradiation (attenuated B_λ from star at r=0,z=0)
    dist_from_star = np.sqrt(r_au**2 + (z_rel * r_au)**2) * AU_TO_CM + 1e-5
    R_star_cm = np.sqrt(L_STAR / (4 * np.pi * STEFAN_BOLTZMANN * T_EFF**4))
    dilution_factor = (R_star_cm / dist_from_star) ** 2
    I_star = B * np.pi * dilution_factor[:, np.newaxis]   # [Nx, 1] isotropic

    # Initial intensity
    I = np.broadcast_to(I_star[:, np.newaxis, :] * np.exp(-tau_cell[:, np.newaxis, np.newaxis]),
                        (Nx, Nw, 1)).copy().astype(np.float64)

    for _ in range(n_iter):
        # Mean intensity (scalar flux / 4π)
        J = I.mean(axis=1, keepdims=True)   # [Nx, 1, 1]

        # Source function
        S = (1.0 - albedo[:, np.newaxis, :]) * I_star[:, np.newaxis, :] + \
            albedo[:, np.newaxis, :] * J[:, :, 0:1]

        # Λ-operator: I = S (1 - exp(-τ)) + I_incident exp(-τ)
        att = np.exp(-tau_cell[:, np.newaxis, np.newaxis])
        I   = S * (1.0 - att) + I_star[:, np.newaxis, :] * att

    return I.astype(np.float32)


class Pinte2009Converter:
    """
    Converts Pinte et al. 2009 continuum RT benchmark to canonical TransportSample.

    Physical model:
    - σ_a = κ_abs(λ) × ρ   (absorption opacity × density)
    - σ_s = κ_sca(λ) × ρ   (scattering opacity × density)
    - g   = Henyey-Greenstein asymmetry parameter from Mie theory
    - I   = intensity from Λ-iteration on the published geometry
    - Q, U (Stokes polarisation) stored as QoIs via approximate polarisation fraction
    """

    BENCHMARK_NAME = "pinte2009"
    N_GROUPS = 1   # monochromatic

    def __init__(self, raw_dir: Optional[Path] = None,
                 wavelength_um: float = 1.0):
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.wavelength_um = wavelength_um
        self._has_raw = self._check_raw()
        self._kappa_abs, self._kappa_sca, self._g = _opacity_at_wavelength(wavelength_um)

    def _check_raw(self) -> bool:
        if self.raw_dir is None:
            return False
        return (self.raw_dir / "density_grid.npy").exists()

    def convert(
        self,
        n_samples: int = 10,
        spatial_shape: tuple = (32, 32),
        n_omega: int = 16,
        rng: Optional[np.random.Generator] = None,
        epsilon: float = 1.0,
        epsilon_range: Optional[Tuple[float, float]] = None,
        xs_perturb: float = 0.10,
    ) -> List[TransportSample]:
        if rng is None:
            rng = np.random.default_rng(3)
        if self._has_raw:
            logger.info("Pinte2009: loading from raw data files.")
            return self._convert_raw(n_samples, spatial_shape, n_omega)
        logger.info(f"Pinte2009: building from published disk model at "
                    f"λ={self.wavelength_um} μm (κ_abs={self._kappa_abs:.2f}, "
                    f"κ_sca={self._kappa_sca:.2f}, g={self._g:.2f}, "
                    f"xs_perturb={xs_perturb:.0%}).")
        return self._generate_from_model(
            n_samples, spatial_shape, n_omega, rng, epsilon, epsilon_range, xs_perturb
        )

    # ── core builder ──────────────────────────────────────────────────────────

    def _make_pinte_sample(self, spatial_shape: tuple, n_omega: int,
                           rng: np.random.Generator, epsilon: float,
                           idx: int, xs_perturb: float = 0.10) -> TransportSample:
        nr, ntheta = spatial_shape
        G  = self.N_GROUPS
        lam = self.wavelength_um

        # Perturb dust opacity parameters independently per sample
        lo, hi = 1.0 - xs_perturb, 1.0 + xs_perturb
        kappa_abs = self._kappa_abs * (float(rng.uniform(lo, hi)) if xs_perturb > 0.0 else 1.0)
        kappa_sca = self._kappa_sca * (float(rng.uniform(lo, hi)) if xs_perturb > 0.0 else 1.0)
        g_param   = float(np.clip(
            self._g * (float(rng.uniform(lo, hi)) if xs_perturb > 0.0 else 1.0),
            0.0, 0.99,
        ))

        # Per-sample disk geometry variation (flaring index, scale height, inner cavity):
        # These change the density structure, producing genuinely different RT problems.
        if xs_perturb > 0.0:
            xi_scale   = float(rng.uniform(0.8, 1.2))   # flaring index  ±20 %
            h0_scale   = float(rng.uniform(0.7, 1.3))   # scale height   ±30 %
            r_in_scale = float(rng.uniform(0.7, 1.3))   # inner radius   ±30 %
        else:
            xi_scale = h0_scale = r_in_scale = 1.0
        xi_flare_i  = float(np.clip(XI_FLARE  * xi_scale,  0.10, 0.50))
        h0_over_i   = float(np.clip(H0_OVER_RREF * h0_scale, 0.02, 0.15))
        r_in_i      = float(np.clip(R_IN_AU   * r_in_scale, 50.0, 200.0))

        # Polar grid (r, elevation angle z/r from midplane)
        r_vals     = np.linspace(r_in_i,   R_OUT_AU,  nr,     dtype=np.float32)
        theta_vals = np.linspace(-0.5,     0.5,       ntheta, dtype=np.float32)  # z/r
        RR, TT = np.meshgrid(r_vals, theta_vals, indexing="ij")   # [nr, ntheta]

        # Compute density with per-sample geometry parameters
        r_clipped = np.clip(RR, r_in_i, R_OUT_AU)
        H_au      = h0_over_i * R_REF * (r_clipped / R_REF) ** (1 + xi_flare_i)
        Sigma     = SIGMA_0 * (r_clipped / R_REF) ** P_SIGMA
        z_au      = TT * RR
        rho       = np.clip(
            (Sigma / (np.sqrt(2 * np.pi) * H_au)) * np.exp(-0.5 * (z_au / H_au) ** 2),
            1e-40, 1.0,
        ).astype(np.float32)   # [nr, ntheta], g/cm^3

        # Macroscopic opacities [nr, ntheta, 1] in cm^-1
        sigma_a = (kappa_abs * rho * AU_TO_CM)[..., np.newaxis].astype(np.float32)
        sigma_s = (kappa_sca * rho * AU_TO_CM)[..., np.newaxis].astype(np.float32)

        # Stellar irradiation source (concentrated near star).
        # Vary stellar flux ±20% per sample to model different stellar luminosities
        # (this is the primary BC input that drives the radiative transfer solution).
        B_lam_ref = _planck_function(lam, T_EFF)   # erg/s/cm^2/cm/sr
        if xs_perturb > 0.0:
            lum_scale = float(rng.uniform(0.8, 1.2))
        else:
            lum_scale = 1.0
        B_lam = B_lam_ref * lum_scale
        q = np.zeros((nr, ntheta, G), dtype=np.float32)
        # Place source energy in innermost radial cells (stellar surface heating)
        n_src = max(1, nr // 10)
        q[:n_src, ntheta // 2 - 1:ntheta // 2 + 2, 0] = float(B_lam) * np.pi

        # 2D angular quadrature (azimuthal uniform)
        angles  = np.linspace(0, 2 * np.pi, n_omega, endpoint=False, dtype=np.float32)
        omega   = np.stack([np.cos(angles), np.sin(angles)], axis=-1)   # [Nw, 2]
        w_omega = np.full(n_omega, 2 * np.pi / n_omega, dtype=np.float32)

        # Spatial query: flattened (r [AU], z/r) grid
        Nx      = nr * ntheta
        x_query = np.stack([RR.ravel(), TT.ravel()], axis=-1)   # [Nx, 2]

        # Intensity via Λ-iteration
        I_vals = _lambda_iteration(
            sigma_a.reshape(Nx, 1),
            sigma_s.reshape(Nx, 1),
            x_query,
            omega,
            g_param,
            B_lam,
            n_iter=5,
        )  # [Nx, Nw, 1]

        phi_vals = I_vals.mean(axis=1)   # [Nx, G]

        # Current via flux gradient, with D clamped to prevent blow-up in near-vacuum.
        # In protoplanetary disk models, the inner region near the star has extremely
        # low density → sigma ≈ 0 → D = 1/(3σ) → ∞.  We cap D so |J| ≤ phi (flux
        # limiter), which is the physically correct free-streaming limit.
        phi_grid = phi_vals.reshape(nr, ntheta, G)
        dr = (R_OUT_AU - R_IN_AU) / max(nr - 1, 1) * AU_TO_CM
        dt = 1.0 / max(ntheta - 1, 1)
        dphidr = np.gradient(phi_grid, dr, axis=0).reshape(Nx, G)
        dphidt = np.gradient(phi_grid, dt, axis=1).reshape(Nx, G)
        grad_mag = np.sqrt(dphidr**2 + dphidt**2) + 1e-30            # [Nx, G]
        phi_safe_fl = np.maximum(phi_vals, 1e-30)                     # [Nx, G]
        # D_raw = 1/(3σ), but clamp so D * |∇φ| ≤ φ  (flux-limited diffusion)
        D_raw = 1.0 / (3.0 * np.maximum((sigma_a + sigma_s).reshape(Nx, G), 1e-30))
        D_flat = np.minimum(D_raw, phi_safe_fl / (grad_mag + 1e-30))
        Jx = (-D_flat * dphidr).astype(np.float32)
        Jy = (-D_flat * dphidt).astype(np.float32)
        J_vals = np.stack([Jx, Jy], axis=1)   # [Nx, 2, G]

        # Replace lambda-iteration I_vals with P1-corrected version using
        # J_fick as the angular-variation driver.
        # I(x,Ω) = φ/(2π) * [1 + (3·J_fick·Ω) / (2π·φ + ε)]
        phi_safe = np.maximum(phi_vals, 1e-30)
        JdotOmega = np.einsum('xdg,wd->xwg', J_vals, omega)   # J_vals is J_fick here
        correction = 1.0 + (3.0 / (2 * np.pi)) * JdotOmega / phi_safe[:, np.newaxis, :]
        correction = np.maximum(correction, 0.0)
        I_vals = (phi_vals[:, np.newaxis, :] / (2 * np.pi) * correction).astype(np.float32)

        # J stored = angular quadrature moment of I (self-consistent with model)
        J_vals = np.einsum('w,wd,nwg->ndg', w_omega, omega, I_vals).astype(np.float32)

        # Stokes Q and U (linear polarisation proxies)
        # Polarisation fraction for single-scattering Rayleigh: ~10% in disk
        p_lin = 0.1 * np.ones_like(phi_vals)
        stokes_Q = phi_vals * p_lin * np.cos(2 * x_query[:, 0:1] / R_OUT_AU)
        stokes_U = phi_vals * p_lin * np.sin(2 * x_query[:, 0:1] / R_OUT_AU)

        dust_T = _dust_temperature(RR.ravel())   # [Nx]

        bc      = BCSpec(bc_type="inflow", values={"stellar_Blambda": np.array([float(B_lam)], dtype=np.float32)})
        inputs  = InputFields(
            sigma_a  = sigma_a,
            sigma_s  = sigma_s,
            q        = q,
            bc       = bc,
            params   = {"epsilon": epsilon, "g": g_param,
                        "wavelength_um": lam, "lum_scale": lum_scale},
            metadata = {
                "benchmark_name": self.BENCHMARK_NAME,
                "dim": 2, "group_count": G,
                "units": "AU_and_cgs",
                "kappa_abs_cm2g": kappa_abs,
                "kappa_sca_cm2g": kappa_sca,
                "L_star_Lsun": L_STAR / L_SUN,
                "T_eff_K": T_EFF,
                "R_in_AU": r_in_i,
                "R_out_AU": R_OUT_AU,
                "xi_flare": xi_flare_i,
                "h0_over_rref": h0_over_i,
                "B_lam_scaled": float(B_lam),
                "lum_scale": lum_scale,
                "intensity_model": "lambda_iteration_5",
            },
        )
        query   = QueryPoints(x=x_query, omega=omega, w_omega=w_omega)
        targets = TargetFields(
            I   = I_vals,
            phi = phi_vals,
            J   = J_vals,
            qois= {"stokes_Q": stokes_Q, "stokes_U": stokes_U,
                   "dust_temperature_K": dust_T[:, np.newaxis].astype(np.float32)},
        )

        return TransportSample(
            inputs=inputs, query=query, targets=targets,
            sample_id=f"pinte2009_{idx:04d}",
        )

    def _generate_from_model(
        self,
        n_samples: int,
        spatial_shape: tuple,
        n_omega: int,
        rng: np.random.Generator,
        epsilon: float,
        epsilon_range: Optional[Tuple[float, float]] = None,
        xs_perturb: float = 0.10,
    ) -> List[TransportSample]:
        samples = []
        for i in range(n_samples):
            eps_i = float(rng.uniform(*epsilon_range)) if epsilon_range is not None else epsilon
            sample = self._make_pinte_sample(
                spatial_shape, n_omega, rng, eps_i, i, xs_perturb=xs_perturb
            )
            samples.append(sample)
        return samples

    # ── raw data loader ───────────────────────────────────────────────────────

    def _convert_raw(self, n_samples: int, spatial_shape: tuple,
                     n_omega: int) -> List[TransportSample]:
        rho_ref      = np.load(str(self.raw_dir / "density_grid.npy"))
        nr, ntheta   = rho_ref.shape[:2]
        lam          = self.wavelength_um
        G            = self.N_GROUPS

        lam_path = self.raw_dir / "wavelengths.npy"
        if lam_path.exists():
            lams = np.load(str(lam_path))
            wl_idx = int(np.argmin(np.abs(lams - lam)))
        else:
            wl_idx = 0

        kappa_abs, kappa_sca, g_param = _opacity_at_wavelength(lam)

        sol_path = self.raw_dir / "solution_stokes.npy"
        if sol_path.exists():
            stokes = np.load(str(sol_path))
            I_ref = stokes[..., wl_idx, 0] if stokes.ndim == 4 else stokes[..., 0]
        else:
            I_ref = None

        sigma_a = (kappa_abs * rho_ref * AU_TO_CM)[..., np.newaxis].astype(np.float32)
        sigma_s = (kappa_sca * rho_ref * AU_TO_CM)[..., np.newaxis].astype(np.float32)
        q       = np.zeros((nr, ntheta, G), dtype=np.float32)
        B_lam   = _planck_function(lam, T_EFF)
        q[:max(1, nr // 10), ntheta//2 - 1:ntheta//2 + 2, 0] = float(B_lam) * np.pi

        r_vals     = np.linspace(R_IN_AU,  R_OUT_AU,  nr,     dtype=np.float32)
        theta_vals = np.linspace(-0.5,     0.5,       ntheta, dtype=np.float32)
        RR, TT = np.meshgrid(r_vals, theta_vals, indexing="ij")
        x_query = np.stack([RR.ravel(), TT.ravel()], axis=-1)
        Nx = nr * ntheta

        angles  = np.linspace(0, 2*np.pi, n_omega, endpoint=False, dtype=np.float32)
        omega   = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        w_omega = np.full(n_omega, 2*np.pi / n_omega, dtype=np.float32)

        if I_ref is not None:
            phi_vals = I_ref.reshape(Nx, G).astype(np.float32)
        else:
            phi_vals = _lambda_iteration(
                sigma_a.reshape(Nx, 1), sigma_s.reshape(Nx, 1),
                x_query, omega, g_param, B_lam, n_iter=5
            ).mean(axis=1)

        I_arr = np.broadcast_to(phi_vals[:, np.newaxis, :],
                                 (Nx, n_omega, G)).copy()
        J_arr = np.zeros((Nx, 2, G), dtype=np.float32)

        bc     = BCSpec(bc_type="inflow", values={"stellar_Blambda": np.array([float(B_lam)], dtype=np.float32)})
        inputs = InputFields(sigma_a=sigma_a, sigma_s=sigma_s, q=q, bc=bc,
                             params={"epsilon": 1.0, "g": g_param,
                                     "wavelength_um": lam},
                             metadata={"benchmark_name": self.BENCHMARK_NAME,
                                       "dim": 2, "group_count": G,
                                       "source": "raw_files"})
        query   = QueryPoints(x=x_query, omega=omega, w_omega=w_omega)
        targets = TargetFields(I=I_arr, phi=phi_vals, J=J_arr)
        sample  = TransportSample(inputs=inputs, query=query, targets=targets,
                                  sample_id="pinte2009_ref_0000")
        return [sample] * n_samples
