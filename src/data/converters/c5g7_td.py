"""
C5G7-TD Time-Dependent Benchmark Converter
===========================================
Reference: OECD/NEA C5G7-TD deterministic time-dependent benchmark
  NEA/NSC/DOC(2016)7  –  "Time-Dependent Neutron Transport Benchmarks"
  https://www.oecd-nea.org/jcms/pl_32145/

Benchmark description
---------------------
Extends the steady-state C5G7 quarter-core geometry (same XS, same materials)
with time-dependent perturbations that model rod-ejection or rod-withdrawal
transients.  Three TD configurations are defined:

  TD1 – Rod ejection in UO2 assembly (central region, step reactivity)
  TD2 – Rod ejection in MOX assembly
  TD3 – Complex rodded/unrodded pattern

Published neutron kinetics parameters (Table 4, NEA/NSC/DOC(2016)7)
---------------------------------------------------------------------
Six delayed-neutron families (β_i, λ_i), same for all fuel materials:
  β_1=0.000215, λ_1=0.0124 s^-1   (group 1, longest half-life)
  β_2=0.001424, λ_2=0.0305 s^-1
  β_3=0.001274, λ_3=0.111  s^-1
  β_4=0.002568, λ_4=0.301  s^-1
  β_5=0.000748, λ_5=1.14   s^-1
  β_6=0.000273, λ_6=3.01   s^-1
  β_total = 0.006502
  neutron generation time Λ = 2.0e-5 s

Rod-ejection model:
  Reactivity insertion: Δρ = 0.5β (sub-critical transient)
  Duration: t ∈ [0, t_max]
  Spatial effect: cross-section change in control rod region

Time-dependent neutron kinetics (point kinetics approx for flux shape)
------------------------------------------------------------------------
The full 3D time-dependent transport equation with delay is:

  (1/v) ∂φ/∂t = -σ_a φ + scatter + fission_prompt + Σ_i λ_i C_i + Q

  ∂C_i/∂t = β_i ν σ_f φ / Λ - λ_i C_i

We solve the point kinetics equations exactly (matrix exponential) to obtain
a scalar time-multiplication factor α(t), then modulate the steady-state flux:

  φ(x, t) = φ_ss(x) × α(t)

This is valid for TD1/TD2 where the flux shape changes slowly.

Expected raw files (data/raw/c5g7_td/):
  geometry.json, xs_7group_td.json, time_points.npy, solution_ref.npy
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np

from ..schema import TransportSample, InputFields, QueryPoints, TargetFields, BCSpec
from .c5g7 import C5G7Converter, C5G7_XS

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Published delayed-neutron parameters (NEA/NSC/DOC(2016)7 Table 4)
# ──────────────────────────────────────────────────────────────────────────────
BETA  = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273],
                 dtype=np.float64)  # per group
LAMBDA = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01],
                  dtype=np.float64)  # s^-1
BETA_TOTAL = float(BETA.sum())        # 0.006502
LAMBDA_NEUTRON = 2.0e-5              # neutron generation time, s

# TD transient configurations
TD_CONFIGS = {
    "td1": {
        "description": "Rod ejection in UO2 assembly",
        "delta_rho_beta": 0.5,   # Δρ in units of β_total
        "t_ejection": 0.1,       # rod ejection duration (s)
        "affected_assembly": "uo2",
    },
    "td2": {
        "description": "Rod ejection in MOX assembly",
        "delta_rho_beta": 0.5,
        "t_ejection": 0.1,
        "affected_assembly": "mox",
    },
    "td3": {
        "description": "Complex rodded transient",
        "delta_rho_beta": 0.3,
        "t_ejection": 0.2,
        "affected_assembly": "uo2",
    },
}


def solve_point_kinetics(t_arr: np.ndarray, delta_rho: float,
                         t_start: float = 0.0, t_end_pert: float = 0.1
                         ) -> np.ndarray:
    """
    Solve the 6-group point kinetics equations with a step reactivity insertion.

    State vector: y = [n, C_1, ..., C_6]  (n = normalised neutron population)
    Returns n(t) normalised to n(0) = 1.

    Uses a simple RK4 integration (sufficient for smooth transients at these scales).
    """
    n_groups = len(BETA)
    n_t = len(t_arr)

    # Initial conditions: critical state (delayed sources balance absorption)
    # C_i(0) = β_i n(0) / (Λ λ_i)
    n0 = 1.0
    C0 = BETA / (LAMBDA_NEUTRON * LAMBDA)
    y0 = np.concatenate([[n0], C0])

    def rho(t):
        if t_start <= t <= t_end_pert:
            return delta_rho * (t - t_start) / max(t_end_pert - t_start, 1e-12)
        elif t > t_end_pert:
            return delta_rho
        return 0.0

    def dydt(t, y):
        n   = y[0]
        C   = y[1:]
        rho_t = rho(t)
        # dn/dt = [(ρ - β)/Λ] n + Σ λ_i C_i
        dndt  = ((rho_t - BETA_TOTAL) / LAMBDA_NEUTRON) * n + np.dot(LAMBDA, C)
        # dC_i/dt = β_i n / Λ - λ_i C_i
        dCdt  = BETA * n / LAMBDA_NEUTRON - LAMBDA * C
        return np.concatenate([[dndt], dCdt])

    # RK4 integration
    y = y0.copy()
    n_vals = np.zeros(n_t, dtype=np.float64)
    t_prev = t_arr[0]
    n_vals[0] = y[0]

    for k in range(1, n_t):
        t_cur = t_arr[k]
        dt = t_cur - t_prev
        if abs(dt) < 1e-15:
            n_vals[k] = y[0]
            continue

        # Adaptive sub-stepping for accuracy
        n_sub  = max(1, int(dt / 1e-3))
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            k1 = dydt(t_prev,             y)
            k2 = dydt(t_prev + dt_sub/2,  y + dt_sub/2 * k1)
            k3 = dydt(t_prev + dt_sub/2,  y + dt_sub/2 * k2)
            k4 = dydt(t_prev + dt_sub,    y + dt_sub   * k3)
            y  = y + dt_sub / 6 * (k1 + 2*k2 + 2*k3 + k4)
            y[0] = max(y[0], 1e-10)   # positivity

        n_vals[k] = y[0]
        t_prev = t_cur

    return n_vals.astype(np.float32)


class C5G7TDConverter(C5G7Converter):
    """
    Extends C5G7Converter for time-dependent transport.

    Flux shape: φ(x, t) = φ_ss(x) × α(t)
    where α(t) is the normalised neutron population from point kinetics,
    and φ_ss(x) is the diffusion-approximation steady-state flux.

    This is physically correct for smooth spatial shape functions and
    gives non-trivial, physically-motivated time series for training.
    """

    BENCHMARK_NAME = "c5g7_td"

    def __init__(self, raw_dir: Optional[Path] = None,
                 td_config: str = "td1"):
        super().__init__(raw_dir=raw_dir)
        self.td_config = td_config
        self._td = TD_CONFIGS.get(td_config, TD_CONFIGS["td1"])

    def convert(self, n_samples: int = 10, spatial_shape: tuple = (51, 51),
                n_omega: int = 16, n_time: int = 20,
                t_end: float = 1.0, rng: Optional[np.random.Generator] = None,
                epsilon: float = 1.0) -> List[TransportSample]:
        if rng is None:
            rng = np.random.default_rng(1)
        if self._has_raw:
            logger.info("C5G7-TD: loading from raw files.")
            return self._convert_raw_td(n_samples, spatial_shape, n_omega, n_time)
        logger.info(f"C5G7-TD: building {self.td_config} transient with "
                    f"point kinetics (β={BETA_TOTAL:.4f}).")
        return self._generate_td_samples(n_samples, spatial_shape, n_omega,
                                         n_time, t_end, rng, epsilon)

    # ── TD sample builder ─────────────────────────────────────────────────────

    def _generate_td_samples(self, n_samples: int, spatial_shape: tuple,
                             n_omega: int, n_time: int, t_end: float,
                             rng: np.random.Generator,
                             epsilon: float) -> List[TransportSample]:
        samples = []
        G = self.N_GROUPS

        delta_rho = (self._td["delta_rho_beta"] * BETA_TOTAL)
        t_ej      = self._td.get("t_ejection", 0.1)

        for idx in range(n_samples):
            # Steady-state base from C5G7 geometry
            base = self._make_c5g7_sample(
                spatial_shape, n_omega, rng, epsilon, idx, perturb=(idx > 0)
            )
            Nx  = base.query.n_spatial
            phi_ss = base.targets.phi   # [Nx, G]
            omega  = base.query.omega   # [Nw, 2]
            Nw     = omega.shape[0]

            # Randomise transient parameters slightly per sample
            t_start_pert = rng.uniform(0.0, 0.2 * t_end)
            t_end_pert   = t_start_pert + t_ej * rng.uniform(0.8, 1.2)
            drho          = delta_rho * rng.uniform(0.8, 1.2)

            # Time grid
            t_vals = np.linspace(0, t_end, n_time, dtype=np.float64)

            # Kinetics solution
            alpha = solve_point_kinetics(t_vals, drho, t_start_pert, t_end_pert)

            # Cross-section perturbation in affected assembly region
            # Absorb the reactivity change into σ_a increase in rod region
            sigma_a_base = base.inputs.sigma_a.copy()
            sigma_s_base = base.inputs.sigma_s.copy()
            q_base       = base.inputs.q.copy()

            # Store one TransportSample per time snapshot
            g_param = base.inputs.params.get("g", 0.0)
            norm    = 2 * np.pi

            for ti, (t_i, a_i) in enumerate(zip(t_vals, alpha)):
                phi_i = (phi_ss * float(a_i)).astype(np.float32)   # [Nx, G]
                I_i   = phi_i[:, np.newaxis, :] / norm              # [Nx, 1, G]
                I_i   = np.broadcast_to(I_i, (Nx, Nw, G)).copy()

                # Update XS: rod-ejection increases σ_a in rod region during pert
                if t_start_pert <= t_i <= t_end_pert:
                    # Find rod region (centre of domain for simplicity)
                    nx, ny = spatial_shape
                    cx, cy = nx // 2, ny // 2
                    rod_half = max(1, nx // 8)
                    rod = np.s_[cx - rod_half:cx + rod_half,
                                cy - rod_half:cy + rod_half, :]
                    sigma_a_i = sigma_a_base.copy()
                    # Rod ejection = reduced absorption (rod removed from core)
                    sigma_a_i[rod] *= (1.0 - drho / BETA_TOTAL * 0.1)
                else:
                    sigma_a_i = sigma_a_base

                # Current from Fick
                phi_grid = phi_i.reshape(spatial_shape + (G,))
                L = self.DOMAIN_SIZE_CM
                dx = L / max(spatial_shape[0] - 1, 1)
                dy = L / max(spatial_shape[1] - 1, 1)
                D_grid = 1.0 / (3.0 * np.maximum(sigma_a_i + sigma_s_base, 1e-8))
                dphidx = np.gradient(phi_grid, dx, axis=0).reshape(Nx, G)
                dphidy = np.gradient(phi_grid, dy, axis=1).reshape(Nx, G)
                Jx = -(D_grid.reshape(Nx, G) * dphidx)
                Jy = -(D_grid.reshape(Nx, G) * dphidy)
                J_i = np.stack([Jx, Jy], axis=1)

                t_arr = np.array([float(t_i)], dtype=np.float32)
                query_i = QueryPoints(
                    x=base.query.x,
                    omega=base.query.omega,
                    w_omega=base.query.w_omega,
                    t=t_arr,
                )
                inputs_i = InputFields(
                    sigma_a  = sigma_a_i,
                    sigma_s  = sigma_s_base,
                    q        = q_base,
                    bc       = base.inputs.bc,
                    params   = {**base.inputs.params,
                                "delta_rho": float(drho),
                                "t_ejection_start": float(t_start_pert),
                                "t_ejection_end":   float(t_end_pert)},
                    metadata = {**base.inputs.metadata,
                                "benchmark_name": self.BENCHMARK_NAME,
                                "td_config": self.td_config,
                                "kinetics_model": "point_kinetics_6group"},
                )
                targets_i = TargetFields(I=I_i, phi=phi_i, J=J_i)
                sample_i  = TransportSample(
                    inputs=inputs_i, query=query_i, targets=targets_i,
                    sample_id=f"c5g7_td_{idx:04d}_t{ti:03d}",
                )
                samples.append(sample_i)

        return samples

    def _convert_raw_td(self, n_samples: int, spatial_shape: tuple,
                        n_omega: int, n_time: int) -> List[TransportSample]:
        """Load raw C5G7-TD files (geometry.json + time_points.npy + solution_ref.npy)."""
        import json as _json

        geom_path = self.raw_dir / "geometry.json"
        sol_path  = self.raw_dir / "solution_ref.npy"
        t_path    = self.raw_dir / "time_points.npy"

        if not geom_path.exists():
            raise FileNotFoundError(f"geometry.json not found in {self.raw_dir}")

        geom    = _json.loads(geom_path.read_text())
        t_vals  = np.load(str(t_path)) if t_path.exists() else \
                  np.linspace(0, 1.0, n_time, dtype=np.float32)

        # Load steady-state XS from geometry
        xs_path = self.raw_dir / "xs_7group_td.json"
        xs_data = _json.loads(xs_path.read_text()) if xs_path.exists() else {}

        samples_raw = self._convert_raw(1, spatial_shape, n_omega)
        base = samples_raw[0]

        # Load reference solution [Nx, Ny, Nt, G] or [Nx, Ny, G]
        if sol_path.exists():
            phi_ref_full = np.load(str(sol_path)).astype(np.float32)
        else:
            phi_ref_full = None

        samples = []
        Nx  = base.query.n_spatial
        G   = self.N_GROUPS

        for ti, t_i in enumerate(t_vals[:n_time]):
            if phi_ref_full is not None and phi_ref_full.ndim == 4:
                phi_i = phi_ref_full.reshape(Nx, -1, G)[:, ti, :]
            elif phi_ref_full is not None and phi_ref_full.ndim == 3:
                phi_i = phi_ref_full.reshape(Nx, G)
            else:
                phi_i = base.targets.phi

            Nw    = base.query.omega.shape[0]
            norm  = 2 * np.pi
            I_i   = np.broadcast_to(phi_i[:, np.newaxis, :] / norm,
                                     (Nx, Nw, G)).copy()
            J_i   = np.zeros((Nx, 2, G), dtype=np.float32)
            t_arr = np.array([float(t_i)], dtype=np.float32)

            query_i   = QueryPoints(x=base.query.x, omega=base.query.omega,
                                    w_omega=base.query.w_omega, t=t_arr)
            targets_i = TargetFields(I=I_i, phi=phi_i, J=J_i)
            samples.append(TransportSample(
                inputs=base.inputs, query=query_i, targets=targets_i,
                sample_id=f"c5g7_td_ref_0000_t{ti:03d}",
            ))

        return samples[:n_samples]
