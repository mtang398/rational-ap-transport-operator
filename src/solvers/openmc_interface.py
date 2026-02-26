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
  solver = OpenMCInterface(n_particles=100_000)
  result = solver.solve_c5g7()   # returns TransportSample with real MC flux

  # Or from the CLI:
  python scripts/run_openmc_c5g7.py --n_particles 100000 --output runs/datasets/
"""

from __future__ import annotations
import logging
import os
import tempfile
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
        n_batches:   int = 300,
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
                           Path(tempfile.mkdtemp(prefix="openmc_c5g7_"))
        self.seed        = seed
        self._available  = self._check_available()

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
                    "OpenMC is not installed → using diffusion-approximation flux.\n"
                    "To get real Monte Carlo flux:\n"
                    "  conda install -c conda-forge openmc\n"
                    "Then re-run this script."
                )
                return self._diffusion_fallback(spatial_shape, n_omega, epsilon)
            raise RuntimeError(
                "OpenMC not installed. "
                "Install with: conda install -c conda-forge openmc"
            )

        return self._run_c5g7_openmc(spatial_shape, n_omega, epsilon, force_rerun)

    def solve(self, sample: TransportSample) -> TransportSample:
        """
        Generic solver: run OpenMC on an arbitrary TransportSample.
        For C5G7-like samples uses the full C5G7 model;
        for other samples uses a homogenised fixed-source model.
        """
        bm = sample.inputs.metadata.get("benchmark_name", "")
        if "c5g7" in bm:
            phi_mc, keff = self._run_c5g7_model(force_rerun=False)
            return self._build_sample_from_phi(sample, phi_mc)
        else:
            return self._run_generic(sample)

    # ── C5G7 OpenMC run ───────────────────────────────────────────────────────

    def _run_c5g7_openmc(
        self,
        spatial_shape: tuple,
        n_omega: int,
        epsilon: float,
        force_rerun: bool,
    ) -> TransportSample:
        phi_mc, keff = self._run_c5g7_model(force_rerun)
        nx, ny = self.n_mesh, self.n_mesh
        G = 7
        L = 64.26  # cm

        # Inputs from published C5G7 geometry
        mat_map = build_quarter_core_map(17)  # 51×51
        sigma_a = np.zeros((nx, ny, G), dtype=np.float32)
        sigma_s = np.zeros((nx, ny, G), dtype=np.float32)
        q_arr   = np.zeros((nx, ny, G), dtype=np.float32)
        for mid, mname in enumerate(MATERIAL_NAMES):
            xs = C5G7_XS[mname]
            mask = (mat_map == mid)
            sigma_a[mask] = xs["sigma_a"]
            sigma_s[mask] = [xs["sigma_s"][g][g] for g in range(G)]
            chi   = np.array(xs["chi"], dtype=np.float32)
            nu_sf = np.array(xs["nu_sigma_f"], dtype=np.float32)
            q_arr[mask] = chi * np.sum(nu_sf) * 0.1

        Nx = nx * ny
        xs_coord = np.linspace(0, L, nx, dtype=np.float32)
        ys_coord = np.linspace(0, L, ny, dtype=np.float32)
        XX, YY   = np.meshgrid(xs_coord, ys_coord, indexing="ij")
        x_query  = np.stack([XX.ravel(), YY.ravel()], axis=-1)

        # phi_mc is [nx, ny, G] → flatten to [Nx, G]
        phi_vals = phi_mc.reshape(Nx, G)

        # Angular quadrature
        angles  = np.linspace(0, 2*np.pi, n_omega, endpoint=False, dtype=np.float32)
        omega   = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        w_omega = np.full(n_omega, 2*np.pi / n_omega, dtype=np.float32)

        # P1 intensity
        norm   = 2 * np.pi
        I_vals = np.broadcast_to(phi_vals[:, np.newaxis, :] / norm,
                                  (Nx, n_omega, G)).copy()

        # Current via Fick's law
        phi_grid = phi_vals.reshape(nx, ny, G)
        dx = L / max(nx-1, 1);  dy = L / max(ny-1, 1)
        D_grid = 1.0 / (3.0 * np.maximum(sigma_a + sigma_s, 1e-8))
        Jx = -(D_grid * np.gradient(phi_grid, dx, axis=0)).reshape(Nx, G)
        Jy = -(D_grid * np.gradient(phi_grid, dy, axis=1)).reshape(Nx, G)
        J_vals = np.stack([Jx, Jy], axis=1)

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

    def _run_c5g7_model(self, force_rerun: bool) -> tuple:
        """Run C5G7OpenMCModel and return (phi [51,51,7], keff)."""
        from .openmc_c5g7_model import C5G7OpenMCModel
        model = C5G7OpenMCModel(
            work_dir    = str(self.work_dir),
            n_particles = self.n_particles,
            n_batches   = self.n_batches,
            n_inactive  = self.n_inactive,
            n_mesh      = self.n_mesh,
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
        samples = conv._generate_from_geometry(1, spatial_shape, n_omega, rng, epsilon)
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
        I_arr = np.broadcast_to(phi[:, np.newaxis, :] / (4*np.pi),
                                 (Nx, Nw, G)).copy()
        J_arr = np.zeros((Nx, dim, G), dtype=np.float32)

        new_targets = TargetFields(I=I_arr, phi=phi, J=J_arr)
        return TransportSample(
            inputs=sample.inputs, query=sample.query,
            targets=new_targets,
            sample_id=sample.sample_id + "_openmc",
        )

    def _build_sample_from_phi(
        self, sample: TransportSample, phi_mc: np.ndarray
    ) -> TransportSample:
        """Replace targets of an existing sample with OpenMC flux."""
        Nx = sample.query.n_spatial
        Nw = sample.query.n_omega
        G  = sample.inputs.n_groups
        phi = phi_mc.reshape(Nx, G)
        I_arr = np.broadcast_to(phi[:, np.newaxis, :] / (2*np.pi),
                                 (Nx, Nw, G)).copy()
        J_arr = np.zeros((Nx, sample.inputs.dim, G), dtype=np.float32)
        new_t = TargetFields(I=I_arr, phi=phi, J=J_arr)
        return TransportSample(
            inputs=sample.inputs, query=sample.query,
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
