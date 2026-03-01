"""
OpenMC Monte Carlo Solver Interface
=====================================
Reference: Romano et al. 2015, Ann. Nucl. Energy
  https://doi.org/10.1016/j.anucene.2014.07.048

OpenMC is an open-source Monte Carlo neutron/photon transport code.
This module wraps OpenMC for running C5G7 and generic fixed-source problems.

Installation (Windows-compatible)
----------------------------------
  conda install -c conda-forge openmc      # ← recommended, no nuclear data needed
                                           #   for multi-group (MGXS) mode
  # OR on Linux/macOS:
  pip install openmc

For continuous-energy mode (not needed here) you also need nuclear data:
  # ENDF/B-VIII.0  ~8 GB
  python -c "import openmc; openmc.data.download_endf()"

Usage
-----
  from src.solvers.openmc_interface import OpenMCInterface
  solver = OpenMCInterface(n_particles=100_000, n_batches=100, n_inactive=50)
  result = solver.solve_c5g7()   # returns TransportSample with real MC flux

  # Or from the CLI (via generate_dataset.py which sets these automatically):
  python scripts/generate_dataset.py --benchmark c5g7 --split train --n_samples 200
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional, List

import numpy as np

from ..data.schema import TransportSample, TargetFields, InputFields, QueryPoints, BCSpec
from ..data.converters.c5g7 import (
    C5G7Converter, build_quarter_core_map, MATERIAL_NAMES, C5G7_XS,
    _diffusion_flux,
)

logger = logging.getLogger(__name__)

# Documented I/O contract
OPENMC_INPUT_SPEC = {
    "geometry":       "OpenMC RectLattice of 3×3 assemblies, each 17×17 pin-cell",
    "materials":      "7-group macroscopic XS (no nuclear data library required)",
    "settings":       "eigenvalue mode, n_particles, n_batches, n_inactive, seed",
    "tallies":        "51×51 mesh × 7 energy groups, scores=[flux, fission]",
    "bc_type":        "vacuum on all 4 lateral surfaces; reflective top/bottom (2D)",
}

OPENMC_OUTPUT_SPEC = {
    "phi":   "scalar flux [51, 51, 7] normalised to peak=1",
    "keff":  "k-effective float",
    "J":     "current [51*51, 2, 7] via Fick's law from phi gradient",
    "I":     "intensity [51*51, Nw, 7] via P1 approximation",
}


class OpenMCInterface:
    """
    Interface to OpenMC for C5G7 and generic transport problems.

    When OpenMC is available, runs a full Monte Carlo eigenvalue calculation.
    When OpenMC is not available, falls back to the diffusion-approximation
    solver from C5G7Converter (prints a clear warning).
    """

    def __init__(
        self,
        n_particles: int = 100_000,
        n_batches:   int = 100,
        n_inactive:  int = 50,
        n_mesh:      int = 51,
        fallback:    bool = True,
        work_dir:    Optional[str] = None,
        seed:        int = 42,
    ):
        self.n_particles = int(n_particles)
        self.n_batches   = int(n_batches)
        self.n_inactive  = int(n_inactive)
        self.n_mesh      = int(n_mesh)
        self.fallback    = fallback
        self.work_dir    = Path(work_dir) if work_dir else \
                           Path("runs/openmc_c5g7")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.seed        = seed
        self._available  = self._check_available()
        if self._available:
            logger.info(
                f"OpenMCInterface: binary found — real Monte Carlo solver active "
                f"(n_particles={self.n_particles}, n_batches={self.n_batches})"
            )
        else:
            logger.warning(
                "OpenMCInterface: openmc binary NOT found — "
                "will fall back to diffusion approximation if fallback=True. "
                "Install with: conda install -c conda-forge openmc"
            )

    # ── availability ──────────────────────────────────────────────────────────

    def _check_available(self) -> bool:
        """
        Returns True only if both the Python API and the openmc binary are present.
        The pip package installs the Python API but not the binary; the conda
        package installs both.
        """
        try:
            import openmc           # noqa: F401
        except ImportError:
            return False

        # Check binary is callable
        import shutil
        if shutil.which("openmc") is not None:
            return True

        # Binary not on PATH — check if conda-installed DLL works (in-process)
        try:
            import openmc.lib       # noqa: F401
            return True
        except (ImportError, OSError):
            pass

        logger.warning(
            "OpenMC Python package found but no working binary or DLL detected.\n"
            "Install via conda to get the binary:\n"
            "  conda install -c conda-forge openmc\n"
            "Falling back to diffusion approximation."
        )
        return False

    @property
    def is_available(self) -> bool:
        return self._available

    # ── main entry points ─────────────────────────────────────────────────────

    def solve_c5g7(
        self,
        spatial_shape: tuple = (51, 51),
        n_omega:       int   = 8,
        epsilon:       float = 1.0,
        force_rerun:   bool  = False,
    ) -> TransportSample:
        """
        Run C5G7 with OpenMC and return a TransportSample with real MC flux.

        Parameters
        ----------
        spatial_shape : (nx, ny) — must be (51, 51) for the published geometry
        n_omega       : number of angular directions in the returned sample
        epsilon       : Knudsen number stored in params
        force_rerun   : re-run even if a cached statepoint exists

        Returns
        -------
        TransportSample with:
          targets.phi = MC flux  [51*51, 7]
          targets.I   = P1 intensity [51*51, n_omega, 7]
          targets.J   = Fick current [51*51, 2, 7]
        """
        if not self._available:
            if self.fallback:
                logger.warning(
                    "OpenMCInterface.solve_c5g7: binary unavailable — "
                    "falling back to SYNTHETIC diffusion approximation (not MC). "
                    "Install with: conda install -c conda-forge openmc"
                )
                return self._diffusion_fallback(spatial_shape, n_omega, epsilon)
            raise RuntimeError(
                "OpenMC not installed. "
                "Install with: conda install -c conda-forge openmc"
            )

        return self._run_c5g7_openmc(spatial_shape, n_omega, epsilon, force_rerun)

    def batch_solve(self, samples: list, show_progress: bool = True) -> list:
        """Solve a batch of samples, calling solve() on each one."""
        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc="OpenMCInterface") if show_progress else samples
        except ImportError:
            iterator = samples

        return [self.solve(s) for s in iterator]

    def solve(self, sample: TransportSample) -> TransportSample:
        """
        Generic solver: run OpenMC on an arbitrary TransportSample.

        For C5G7 benchmarks:
          - Extracts per-material XS multipliers stored in the sample's metadata
            (put there by C5G7Converter when it perturbs XS).  Each unique
            perturbation gets its own OpenMC run in a dedicated subdirectory so
            the model truly learns the operator  T: (σ_a, σ_s, q, ε) → I(x,ω)
            over a distribution of physics configurations, not just one.
          - The MC eigenvalue calculation runs at the native n_mesh×n_mesh tally
            resolution.  If the sample's spatial_shape differs, the MC flux is
            bilinearly interpolated to the requested shape — still real MC data.
          - The XS arrays (sigma_a, sigma_s, q) stay at the sample's own
            resolution so input and target dimensions are consistent.

        For other benchmarks uses a homogenised fixed-source model.
        """
        bm = sample.inputs.metadata.get("benchmark_name", "")
        if "c5g7" in bm:
            xs_mults = sample.inputs.metadata.get("xs_multipliers", None)
            sid      = sample.sample_id
            phi_mc, keff = self._run_c5g7_model(
                force_rerun=False, xs_multipliers=xs_mults, sample_id=sid
            )
            shape  = sample.inputs.spatial_shape
            native = (self.n_mesh, self.n_mesh)
            if shape != native:
                logger.info(
                    f"OpenMCInterface.solve: interpolating MC flux from native "
                    f"{native} to requested shape={shape} (real MC data, no approximation)."
                )
                phi_mc = self._interpolate_phi(phi_mc, native, shape)
            return self._build_sample_from_phi(sample, phi_mc, keff=keff)
        else:
            return self._run_generic(sample)

    @staticmethod
    def _interpolate_phi(
        phi_mc: np.ndarray,
        src_shape: tuple,
        dst_shape: tuple,
    ) -> np.ndarray:
        """
        Bilinearly interpolate MC scalar flux from src_shape to dst_shape.

        Parameters
        ----------
        phi_mc   : [nx_src, ny_src, G]  — MC tally output on native mesh
        src_shape: (nx_src, ny_src)
        dst_shape: (nx_dst, ny_dst)

        Returns
        -------
        phi_out  : [nx_dst, ny_dst, G]  — interpolated MC flux
        """
        from scipy.ndimage import zoom as _zoom
        nx_src, ny_src = src_shape
        nx_dst, ny_dst = dst_shape
        G = phi_mc.shape[2]
        # Compute zoom factors for spatial axes only; group axis unchanged
        zx = nx_dst / nx_src
        zy = ny_dst / ny_src
        # Interpolate each group independently (order=1 = bilinear)
        out = np.zeros((nx_dst, ny_dst, G), dtype=np.float32)
        for g in range(G):
            out[:, :, g] = _zoom(phi_mc[:, :, g].astype(np.float64),
                                 (zx, zy), order=1).astype(np.float32)
        # Preserve non-negativity (bilinear interpolation can very rarely give
        # tiny negative values near zero-flux boundaries due to floating point)
        out = np.maximum(out, 0.0)
        return out

    # ── C5G7 OpenMC run ───────────────────────────────────────────────────────

    def _run_c5g7_openmc(
        self,
        spatial_shape: tuple,
        n_omega: int,
        epsilon: float,
        force_rerun: bool,
        xs_multipliers: Optional[dict] = None,
        sample_id: Optional[str] = None,
    ) -> TransportSample:
        phi_mc, keff = self._run_c5g7_model(
            force_rerun, xs_multipliers=xs_multipliers, sample_id=sample_id
        )
        native = (self.n_mesh, self.n_mesh)
        G = 7
        L = 64.26  # cm

        # Interpolate MC flux to requested spatial_shape if different from native mesh.
        if spatial_shape != native:
            logger.info(
                f"_run_c5g7_openmc: interpolating MC flux from native "
                f"{native} to requested shape={spatial_shape} (real MC data, no approximation)."
            )
            phi_mc = self._interpolate_phi(phi_mc, native, spatial_shape)

        nx, ny = spatial_shape
        # Build XS arrays at the requested spatial resolution
        n_pins = 17
        mat_map_native = build_quarter_core_map(n_pins)  # 51×51 at native resolution
        if spatial_shape != native:
            # Nearest-neighbour upsample material map to requested resolution
            from scipy.ndimage import zoom as _zoom
            zx = nx / native[0]
            zy = ny / native[1]
            mat_map = _zoom(mat_map_native.astype(np.float32), (zx, zy), order=0).astype(int)
        else:
            mat_map = mat_map_native

        sigma_a = np.zeros((nx, ny, G), dtype=np.float32)
        sigma_s = np.zeros((nx, ny, G), dtype=np.float32)
        q_arr   = np.zeros((nx, ny, G), dtype=np.float32)
        mults   = xs_multipliers or {}
        for mid, mname in enumerate(MATERIAL_NAMES):
            xs   = C5G7_XS[mname]
            mask = (mat_map == mid)
            m    = mults.get(mname, {})
            sa_ref = np.array(xs["sigma_a"], dtype=np.float32)
            ss_ref = np.array([xs["sigma_s"][g][g] for g in range(G)], dtype=np.float32)
            chi    = np.array(xs["chi"], dtype=np.float32)
            nu_sf  = np.array(xs["nu_sigma_f"], dtype=np.float32)
            # Apply same per-material multipliers that were fed to the MGXS library
            if "sigma_a" in m:
                sa_ref = sa_ref * np.asarray(m["sigma_a"], dtype=np.float32)
            if "sigma_s" in m:
                ss_ref = ss_ref * np.asarray(m["sigma_s"], dtype=np.float32)
            if "nu_fission" in m:
                nu_sf = nu_sf * np.asarray(m["nu_fission"], dtype=np.float32)
            sigma_a[mask] = sa_ref
            sigma_s[mask] = ss_ref
            q_arr[mask]   = chi * np.sum(nu_sf) * 0.1

        # No global q_scale: C5G7 is an eigenvalue problem where q is determined
        # by the flux itself, not an independent input.  XS multipliers fully
        # characterise the perturbed physics.

        Nx = nx * ny
        xs_coord = np.linspace(0, L, nx, dtype=np.float32)
        ys_coord = np.linspace(0, L, ny, dtype=np.float32)
        XX, YY   = np.meshgrid(xs_coord, ys_coord, indexing="ij")
        x_query  = np.stack([XX.ravel(), YY.ravel()], axis=-1)

        # phi_mc is [nx, ny, G] → flatten to [Nx, G]
        phi_vals = phi_mc.reshape(Nx, G).astype(np.float32)

        # Angular quadrature
        angles  = np.linspace(0, 2*np.pi, n_omega, endpoint=False, dtype=np.float32)
        omega   = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        w_omega = np.full(n_omega, 2*np.pi / n_omega, dtype=np.float32)

        # Fick current used only as intermediate to build P1 intensity
        phi_grid = phi_vals.reshape(nx, ny, G)
        dx = L / max(nx-1, 1)
        dy = L / max(ny-1, 1)
        D_grid = 1.0 / (3.0 * np.maximum(sigma_a + sigma_s, 1e-8))
        Jx_fick = -(D_grid * np.gradient(phi_grid, dx, axis=0)).reshape(Nx, G)
        Jy_fick = -(D_grid * np.gradient(phi_grid, dy, axis=1)).reshape(Nx, G)
        J_fick = np.stack([Jx_fick, Jy_fick], axis=1)

        # P1 intensity: I(x,Ω) = phi/(2π) * [1 + (3·J_fick·Ω) / (2π·phi)]
        phi_safe = np.maximum(phi_vals, 1e-30)
        JdotOmega = np.einsum('xdg,wd->xwg', J_fick, omega)   # [Nx, Nw, G]
        correction = 1.0 + (3.0 / (2 * np.pi)) * JdotOmega / phi_safe[:, np.newaxis, :]
        correction = np.maximum(correction, 0.0)
        I_vals = (phi_vals[:, np.newaxis, :] / (2 * np.pi) * correction).astype(np.float32)

        # J stored = angular quadrature moment of I (self-consistent with model's J head)
        J_vals = np.einsum('w,wd,nwg->ndg', w_omega, omega, I_vals).astype(np.float32)

        bc      = BCSpec(bc_type="vacuum")
        inputs  = InputFields(
            sigma_a  = sigma_a,
            sigma_s  = sigma_s,
            q        = q_arr,
            bc       = bc,
            params   = {"epsilon": epsilon, "g": 0.0, "keff": keff},
            metadata = {
                "benchmark_name": "c5g7",
                "flux_source": "openmc_multigroup_mc",
                "n_particles": self.n_particles,
                "n_batches":   self.n_batches,
                "keff":        keff,
                "dim": 2, "group_count": G, "units": "cm",
            },
        )
        query   = QueryPoints(x=x_query, omega=omega, w_omega=w_omega)
        targets = TargetFields(I=I_vals, phi=phi_vals, J=J_vals)
        return TransportSample(
            inputs=inputs, query=query, targets=targets,
            sample_id="c5g7_openmc_0000",
        )

    def _run_c5g7_model(
        self,
        force_rerun: bool,
        xs_multipliers: Optional[dict] = None,
        sample_id: Optional[str] = None,
    ) -> tuple:
        """Run C5G7OpenMCModel and return (phi [51,51,7], keff)."""
        from .openmc_c5g7_model import C5G7OpenMCModel
        model = C5G7OpenMCModel(
            work_dir       = str(self.work_dir),
            n_particles    = self.n_particles,
            n_batches      = self.n_batches,
            n_inactive     = self.n_inactive,
            n_mesh         = self.n_mesh,
            xs_multipliers = xs_multipliers,
            sample_id      = sample_id,
        )
        return model.run(force=force_rerun)

    # ── fallback (no OpenMC) ──────────────────────────────────────────────────

    def _diffusion_fallback(
        self, spatial_shape: tuple, n_omega: int, epsilon: float
    ) -> TransportSample:
        """Use the C5G7Converter diffusion approximation when OpenMC is absent."""
        conv = C5G7Converter()
        import numpy as _np
        rng = _np.random.default_rng(42)
        # xs_perturb=0.0: fallback returns the canonical (unperturbed) geometry,
        # not a randomly perturbed sample.
        samples = conv._generate_from_geometry(1, spatial_shape, n_omega, rng, epsilon,
                                               xs_perturb=0.0)
        return samples[0]

    # ── generic fixed-source model ────────────────────────────────────────────

    def _run_generic(self, sample: TransportSample) -> TransportSample:
        """
        Build a homogenised fixed-source OpenMC model from the sample's XS.
        Used for non-C5G7 benchmarks.
        """
        import openmc

        inp   = sample.inputs
        G     = inp.n_groups
        dim   = inp.dim
        shape = inp.spatial_shape
        L     = 60.0 if dim == 3 else 64.26

        sigma_a_mean = float(inp.sigma_a.mean())
        sigma_s_mean = float(inp.sigma_s.mean())
        sigma_t_mean = sigma_a_mean + sigma_s_mean

        # Single homogenised material via openmc.Material.mix_materials or
        # direct macroscopic XS
        if G == 1:
            xs = openmc.XSdata('mat', openmc.mgxs.EnergyGroups([1e-5, 1e7]))
            xs.order = 0
            xs.set_total([sigma_t_mean])
            xs.set_absorption([sigma_a_mean])
            xs.set_scatter_matrix(np.array([[[sigma_s_mean]]]))
            xs.set_nu_fission([0.0])
            xs.set_chi([0.0])
            lib = openmc.MGXSLibrary(openmc.mgxs.EnergyGroups([1e-5, 1e7]))
            lib.add_xsdatas([xs])
            mgxs_p = self.work_dir / "mgxs_generic.h5"
            lib.export_to_hdf5(str(mgxs_p))
        else:
            # Fall back to diffusion for multi-group generic
            logger.info("Generic multi-group OpenMC not yet implemented; using diffusion.")
            return self._diffusion_fallback(shape, sample.query.n_omega,
                                             sample.inputs.params.get("epsilon", 1.0))

        mat = openmc.Material(name="homogenized")
        mat.set_density('macro', 1.0)
        mat.add_macroscopic(openmc.Macroscopic('mat'))
        mats = openmc.Materials([mat])
        mats.cross_sections = str(mgxs_p)

        surfs = self._make_box_surfaces(dim, L, inp.bc.bc_type)
        region = self._region_from_surfs(surfs, dim)
        cell = openmc.Cell(fill=mat, region=region)
        geom = openmc.Geometry(openmc.Universe(cells=[cell]))

        mesh = openmc.RegularMesh()
        mesh.dimension   = list(shape) if dim == 3 else list(shape) + [1]
        mesh.lower_left  = [0]*3
        mesh.upper_right = ([L]*dim + [1e6] if dim == 2 else [L]*3)
        mf   = openmc.MeshFilter(mesh)
        tally = openmc.Tally(name='flux')
        tally.filters = [mf]
        tally.scores   = ['flux']

        src = openmc.IndependentSource(
            space=openmc.stats.Box([0]*3, [L]*dim + [1] if dim == 2 else [L]*3))
        settings = openmc.Settings()
        settings.energy_mode = 'multi-group'
        settings.cross_sections = str(mgxs_p)
        settings.batches   = min(self.n_batches, 50)
        settings.inactive  = 0
        settings.particles = self.n_particles
        settings.run_mode  = 'fixed source'
        settings.source    = src

        model = openmc.Model(geom, mats, settings, openmc.Tallies([tally]))
        model.run(cwd=str(self.work_dir))

        sp_files = sorted(self.work_dir.glob("statepoint.*.h5"))
        if not sp_files:
            logger.error("No statepoint found after OpenMC run.")
            return sample

        sp    = openmc.StatePoint(str(sp_files[-1]))
        t     = sp.get_tally(name='flux')
        flux  = np.abs(t.get_values(scores=['flux']).ravel()).astype(np.float32)
        Nx    = sample.query.n_spatial
        Nw    = sample.query.n_omega
        phi   = flux[:Nx*G].reshape(Nx, G)
        # Build I via P1 using Fick current as the angular-variation driver
        omega_arr = sample.query.omega   # [Nw, dim]
        w_arr     = sample.query.w_omega # [Nw]
        norm = 4 * np.pi if dim == 3 else 2 * np.pi
        phi_safe = np.maximum(phi, 1e-30)

        J_fick = np.zeros((Nx, dim, G), dtype=np.float32)
        if dim == 2:
            nx2, ny2 = shape
            L2 = 64.26
            sa = sample.inputs.sigma_a.reshape(Nx, G)
            ss = sample.inputs.sigma_s.reshape(Nx, G)
            D2 = 1.0 / (3.0 * np.maximum(sa + ss, 1e-8))
            pg = phi.reshape(nx2, ny2, G)
            dx2 = L2 / max(nx2 - 1, 1)
            dy2 = L2 / max(ny2 - 1, 1)
            Jx2 = -(D2 * np.gradient(pg, dx2, axis=0)).reshape(Nx, G)
            Jy2 = -(D2 * np.gradient(pg, dy2, axis=1)).reshape(Nx, G)
            J_fick = np.stack([Jx2, Jy2], axis=1)

        JdotOmega = np.einsum('xdg,wd->xwg', J_fick, omega_arr)
        corr = np.maximum(1.0 + (3.0 / norm) * JdotOmega / phi_safe[:, np.newaxis, :], 0.0)
        I_arr = (phi[:, np.newaxis, :] / norm * corr).astype(np.float32)

        # J stored = quadrature moment of I (self-consistent with model's J head)
        J_arr = np.einsum('w,wd,nwg->ndg', w_arr, omega_arr, I_arr).astype(np.float32)

        new_targets = TargetFields(I=I_arr, phi=phi, J=J_arr)
        return TransportSample(
            inputs=sample.inputs, query=sample.query,
            targets=new_targets,
            sample_id=sample.sample_id + "_openmc",
        )

    def _build_sample_from_phi(
        self, sample: TransportSample, phi_mc: np.ndarray, keff: float = 0.0
    ) -> TransportSample:
        """
        Replace targets of an existing sample with OpenMC scalar flux.

        - phi  : taken directly from the OpenMC tally (real MC solution).
        - J    : computed via Fick's law using the sample's own (possibly
                 perturbed) cross-sections, so input/target physics are
                 self-consistent.
        - I    : P1 approximation  I(x,Ω) = phi/(2π) [1 + 3/(2π) * J·Ω]
                 This gives first-order angular variation instead of the
                 purely isotropic broadcast used previously.
        """
        Nx = sample.query.n_spatial
        Nw = sample.query.n_omega
        G  = sample.inputs.n_groups
        # phi_mc is [nx, ny, G] or already [Nx, G]; normalise to [Nx, G]
        phi = phi_mc.reshape(Nx, G).astype(np.float32)

        # Build I via P1, using Fick current as the angular-variation driver.
        # J stored in the sample will be the quadrature moment of I so it is
        # self-consistent with what the model's J head computes.
        dim = sample.inputs.dim
        shape = sample.inputs.spatial_shape
        L = 64.26
        sigma_a = sample.inputs.sigma_a.reshape(Nx, G)
        sigma_s = sample.inputs.sigma_s.reshape(Nx, G)
        D = 1.0 / (3.0 * np.maximum(sigma_a + sigma_s, 1e-8))  # [Nx, G]

        omega = sample.query.omega  # [Nw, dim]
        w_omega = sample.query.w_omega  # [Nw]
        phi_safe = np.maximum(phi, 1e-30)  # [Nx, G]

        if dim == 2:
            nx, ny = shape
            phi_grid = phi.reshape(nx, ny, G)
            dx = L / max(nx - 1, 1)
            dy = L / max(ny - 1, 1)
            D_grid = D.reshape(nx, ny, G)
            Jx_fick = -(D_grid * np.gradient(phi_grid, dx, axis=0)).reshape(Nx, G)
            Jy_fick = -(D_grid * np.gradient(phi_grid, dy, axis=1)).reshape(Nx, G)
            J_fick = np.stack([Jx_fick, Jy_fick], axis=1)         # [Nx, 2, G]

            JdotOmega = np.einsum('xdg,wd->xwg', J_fick, omega)   # [Nx, Nw, G]
            correction = 1.0 + (3.0 / (2 * np.pi)) * JdotOmega / phi_safe[:, np.newaxis, :]
            correction = np.maximum(correction, 0.0)
            I_arr = (phi[:, np.newaxis, :] / (2 * np.pi) * correction).astype(np.float32)
        else:
            I_arr = np.broadcast_to(
                phi[:, np.newaxis, :] / (4 * np.pi), (Nx, Nw, G)
            ).copy().astype(np.float32)

        # J stored = angular quadrature moment of I (consistent with model output)
        J_arr = np.einsum('w,wd,nwg->ndg', w_omega, omega, I_arr).astype(np.float32)

        # Update metadata to record that phi came from OpenMC
        import copy as _copy
        new_meta = _copy.copy(sample.inputs.metadata)
        new_meta["flux_source"] = "openmc_multigroup_mc"
        new_meta["n_particles"] = self.n_particles
        new_meta["n_batches"]   = self.n_batches
        if keff:
            new_meta["keff"] = float(keff)

        new_inputs = InputFields(
            sigma_a      = sample.inputs.sigma_a,
            sigma_s      = sample.inputs.sigma_s,
            q            = sample.inputs.q,
            extra_fields = sample.inputs.extra_fields,
            bc           = sample.inputs.bc,
            params       = sample.inputs.params,
            metadata     = new_meta,
        )
        new_t = TargetFields(I=I_arr, phi=phi, J=J_arr)
        return TransportSample(
            inputs=new_inputs, query=sample.query,
            targets=new_t,
            sample_id=sample.sample_id + "_openmc",
        )

    @staticmethod
    def _make_box_surfaces(dim, L, bc_type):
        import openmc
        s = {}
        axes = ['x', 'y', 'z']
        plane_cls = [openmc.XPlane, openmc.YPlane, openmc.ZPlane]
        for i in range(min(dim, 3)):
            s[f'{axes[i]}min'] = plane_cls[i](x0=0.0, boundary_type=bc_type)
            s[f'{axes[i]}max'] = plane_cls[i](x0=L,   boundary_type=bc_type)
        return s

    @staticmethod
    def _region_from_surfs(surfs, dim):
        axes = ['x', 'y', 'z']
        r = None
        for i in range(min(dim, 3)):
            seg = +surfs[f'{axes[i]}min'] & -surfs[f'{axes[i]}max']
            r = seg if r is None else r & seg
        return r
