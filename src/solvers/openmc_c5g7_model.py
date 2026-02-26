"""
Self-contained OpenMC multi-group model for the C5G7 MOX benchmark.

Based on the MIT CRPG reference implementation:
  https://github.com/mit-crpg/benchmarks/tree/master/c5g7/openmc

This module builds the complete C5G7 2D quarter-core geometry using
OpenMC's Python API with the published 7-group multi-group cross sections.
No external nuclear data library is required (uses macro XS directly).

The model produces:
  - 7-group scalar flux phi[Nx, Ny, 7] on a 51x51 mesh
  - k-effective (eigenvalue)
  - Per-group fission rate

Install OpenMC:
  conda install -c conda-forge openmc      # recommended (Windows compatible)
  # OR
  pip install openmc                        # Linux/macOS only via pip

Usage:
  from src.solvers.openmc_c5g7_model import C5G7OpenMCModel
  model = C5G7OpenMCModel(work_dir='runs/openmc_c5g7', n_particles=100_000)
  phi, keff = model.run()                  # returns [51, 51, 7], float
"""

from __future__ import annotations
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Domain constants (from NEA benchmark) ─────────────────────────────────────
HALF_CORE   = 32.13   # cm  (half of 64.26 cm full core)
PIN_PITCH   = 1.26    # cm  (UO2/MOX pin pitch)
ASM_PITCH   = 21.42   # cm  (17 pins × 1.26 cm)
N_PINS      = 17      # pins per assembly per direction
N_ASM       = 3       # assemblies per direction (quarter core)
N_MESH      = 51      # tally mesh cells per direction (= 3 × 17)

# ── 7-group energy bounds (eV) ─────────────────────────────────────────────────
# Standard C5G7 energy group structure (fast to thermal)
GROUP_BOUNDS = [1e-5, 0.058, 0.14, 0.28, 0.625, 4.0, 5530.0, 1e7]  # eV, 8 boundaries


def _make_mgxs_library(openmc):
    """Build the 7-group MGXS library from published C5G7 cross-sections."""
    import openmc.mgxs
    groups = openmc.mgxs.EnergyGroups(np.array(GROUP_BOUNDS))

    def _xs(name, total, absorption, scatter_rows, nu_fission, chi):
        xs = openmc.XSdata(name, groups)
        xs.order = 0
        xs.set_total(np.array(total))
        xs.set_absorption(np.array(absorption))
        sm = np.array(scatter_rows)[np.newaxis, :, :]  # shape [1,7,7]
        sm = np.rollaxis(sm, 0, 3)                      # shape [7,7,1]
        xs.set_scatter_matrix(sm)
        xs.set_nu_fission(np.array(nu_fission))
        xs.set_chi(np.array(chi))
        return xs

    # --- UO2 ---
    uo2 = _xs('uo2',
        total      =[1.77949e-01, 3.29805e-01, 4.80388e-01, 5.54367e-01, 3.11801e-01, 3.95168e-01, 5.64406e-01],
        absorption =[8.02480e-03, 3.71740e-03, 2.67690e-02, 9.62360e-02, 3.00200e-02, 1.11260e-01, 2.82780e-01],
        scatter_rows=[
            [1.27537e-01, 4.23780e-02, 9.43740e-06, 5.51630e-09, 0, 0, 0],
            [0, 3.24456e-01, 1.63140e-03, 3.14270e-09, 0, 0, 0],
            [0, 0, 4.50940e-01, 2.67920e-03, 0, 0, 0],
            [0, 0, 0, 4.52565e-01, 5.56640e-03, 0, 0],
            [0, 0, 0, 1.25250e-04, 2.71401e-01, 1.02550e-02, 1.00210e-08],
            [0, 0, 0, 0, 1.29680e-03, 2.65802e-01, 1.68090e-02],
            [0, 0, 0, 0, 0, 8.54580e-03, 2.73080e-01]],
        nu_fission =[2.00600e-02, 2.02730e-03, 1.57060e-02, 4.51830e-02, 4.33421e-02, 2.02090e-01, 5.25711e-01],
        chi        =[5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0, 0, 0])

    # --- MOX 4.3% ---
    mox43 = _xs('mox43',
        total      =[1.78731e-01, 3.30849e-01, 4.83772e-01, 5.66922e-01, 4.26227e-01, 6.78997e-01, 6.82852e-01],
        absorption =[8.43390e-03, 3.75770e-03, 2.79700e-02, 1.04210e-01, 1.39940e-01, 4.09180e-01, 4.09350e-01],
        scatter_rows=[
            [1.28876e-01, 4.14130e-02, 8.22900e-06, 5.04050e-09, 0, 0, 0],
            [0, 3.25452e-01, 1.63950e-03, 1.59820e-09, 0, 0, 0],
            [0, 0, 4.53188e-01, 2.61420e-03, 0, 0, 0],
            [0, 0, 0, 4.57173e-01, 5.53940e-03, 0, 0],
            [0, 0, 0, 1.60460e-04, 2.76814e-01, 9.31270e-03, 9.16560e-09],
            [0, 0, 0, 0, 2.00510e-03, 2.52962e-01, 1.48500e-02],
            [0, 0, 0, 0, 0, 8.49480e-03, 2.65007e-01]],
        nu_fission =[2.17530e-02, 2.53510e-03, 1.62680e-02, 6.54741e-02, 3.07241e-02, 6.66651e-01, 7.13990e-01],
        chi        =[5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0, 0, 0])

    # --- MOX 7.0% ---
    mox7 = _xs('mox7',
        total      =[1.81323e-01, 3.34368e-01, 4.93785e-01, 5.91216e-01, 4.74198e-01, 8.33601e-01, 8.53603e-01],
        absorption =[9.06570e-03, 4.29670e-03, 3.28810e-02, 1.22030e-01, 1.82980e-01, 5.68460e-01, 5.85210e-01],
        scatter_rows=[
            [1.30457e-01, 4.17920e-02, 8.51050e-06, 5.13290e-09, 0, 0, 0],
            [0, 3.28428e-01, 1.64360e-03, 2.20170e-09, 0, 0, 0],
            [0, 0, 4.58371e-01, 2.53310e-03, 0, 0, 0],
            [0, 0, 0, 4.63709e-01, 5.47660e-03, 0, 0],
            [0, 0, 0, 1.76190e-04, 2.82313e-01, 8.72890e-03, 9.00160e-09],
            [0, 0, 0, 0, 2.27600e-03, 2.49751e-01, 1.31140e-02],
            [0, 0, 0, 0, 0, 8.86450e-03, 2.59529e-01]],
        nu_fission =[2.38140e-02, 3.85869e-03, 2.41340e-02, 9.43662e-02, 4.57699e-02, 9.28181e-01, 1.04320e+00],
        chi        =[5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0, 0, 0])

    # --- MOX 8.7% ---
    mox87 = _xs('mox87',
        total      =[1.83045e-01, 3.36705e-01, 5.00507e-01, 6.06174e-01, 5.02754e-01, 9.21028e-01, 9.55231e-01],
        absorption =[9.48620e-03, 4.65560e-03, 3.62400e-02, 1.32720e-01, 2.08400e-01, 6.58700e-01, 6.90170e-01],
        scatter_rows=[
            [1.31504e-01, 4.20460e-02, 8.69720e-06, 5.19380e-09, 0, 0, 0],
            [0, 3.30403e-01, 1.64630e-03, 2.60060e-09, 0, 0, 0],
            [0, 0, 4.61792e-01, 2.47490e-03, 0, 0, 0],
            [0, 0, 0, 4.68021e-01, 5.43300e-03, 0, 0],
            [0, 0, 0, 1.85970e-04, 2.85771e-01, 8.39730e-03, 8.92800e-09],
            [0, 0, 0, 0, 2.39160e-03, 2.47614e-01, 1.23220e-02],
            [0, 0, 0, 0, 0, 8.96810e-03, 2.56093e-01]],
        nu_fission =[2.51860e-02, 4.73951e-03, 2.94781e-02, 1.12250e-01, 5.53030e-02, 1.07500e+00, 1.23930e+00],
        chi        =[5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0, 0, 0])

    # --- Fission Chamber ---
    fiss_ch = _xs('fiss_chamber',
        total      =[1.26032e-01, 2.93160e-01, 2.84250e-01, 2.81020e-01, 3.34460e-01, 5.65640e-01, 1.17214e+00],
        absorption =[5.11320e-04, 7.58130e-05, 3.16430e-04, 1.16750e-03, 3.39770e-03, 9.18860e-03, 2.32440e-02],
        scatter_rows=[
            [6.61659e-02, 5.90700e-02, 2.83340e-04, 1.46220e-06, 2.06420e-08, 0, 0],
            [0, 2.40377e-01, 5.24350e-02, 2.49900e-04, 1.92390e-05, 2.98750e-06, 4.21400e-07],
            [0, 0, 1.83425e-01, 9.22880e-02, 6.93650e-03, 1.07900e-03, 2.05430e-04],
            [0, 0, 0, 7.90769e-02, 1.69990e-01, 2.58600e-02, 4.92560e-03],
            [0, 0, 0, 3.73400e-05, 9.97570e-02, 2.06790e-01, 2.44780e-02],
            [0, 0, 0, 0, 9.17420e-04, 3.16774e-01, 2.38760e-01],
            [0, 0, 0, 0, 0, 4.97930e-02, 1.09910e+00]],
        nu_fission =[1.32340e-08, 1.43450e-08, 1.12860e-06, 1.27630e-05, 3.53850e-07, 1.74010e-06, 5.06330e-06],
        chi        =[5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0, 0, 0])

    # --- Guide Tube ---
    guide_tube = _xs('guide_tube',
        total      =[1.26032e-01, 2.93160e-01, 2.84240e-01, 2.80960e-01, 3.34440e-01, 5.65640e-01, 1.17215e+00],
        absorption =[5.11320e-04, 7.58010e-05, 3.15720e-04, 1.15820e-03, 3.39750e-03, 9.18780e-03, 2.32420e-02],
        scatter_rows=[
            [6.61659e-02, 5.90700e-02, 2.83340e-04, 1.46220e-06, 2.06420e-08, 0, 0],
            [0, 2.40377e-01, 5.24350e-02, 2.49900e-04, 1.92390e-05, 2.98750e-06, 4.21400e-07],
            [0, 0, 1.83297e-01, 9.23970e-02, 6.94460e-03, 1.08030e-03, 2.05670e-04],
            [0, 0, 0, 7.88511e-02, 1.70140e-01, 2.58810e-02, 4.92970e-03],
            [0, 0, 0, 3.73330e-05, 9.97372e-02, 2.06790e-01, 2.44780e-02],
            [0, 0, 0, 0, 9.17260e-04, 3.16765e-01, 2.38770e-01],
            [0, 0, 0, 0, 0, 4.97920e-02, 1.09912e+00]],
        nu_fission =[0]*7,
        chi        =[0]*7)

    # --- Water (moderator / reflector) ---
    water = _xs('water',
        total      =[1.59206e-01, 4.12970e-01, 5.90310e-01, 5.84350e-01, 7.18000e-01, 1.25445e+00, 2.65038e+00],
        absorption =[6.01050e-04, 1.57930e-05, 3.37160e-04, 1.94060e-03, 5.74160e-03, 1.50010e-02, 3.72390e-02],
        scatter_rows=[
            [4.44777e-02, 1.13400e-01, 7.23470e-04, 3.74990e-06, 5.31840e-08, 0, 0],
            [0, 2.82334e-01, 1.29940e-01, 6.23400e-04, 4.80020e-05, 7.44860e-06, 1.04550e-06],
            [0, 0, 3.45256e-01, 2.24570e-01, 1.69990e-02, 2.64430e-03, 5.03440e-04],
            [0, 0, 0, 9.10284e-02, 4.15510e-01, 6.37320e-02, 1.21390e-02],
            [0, 0, 0, 7.14370e-05, 1.39138e-01, 5.11820e-01, 6.12290e-02],
            [0, 0, 0, 0, 2.21570e-03, 6.99913e-01, 5.37320e-01],
            [0, 0, 0, 0, 0, 1.32440e-01, 2.48070e+00]],
        nu_fission =[0]*7,
        chi        =[0]*7)

    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdatas([uo2, mox43, mox7, mox87, fiss_ch, guide_tube, water])
    return lib


# Guide-tube positions in a 17×17 assembly (0-indexed row, col)
_GT_POSITIONS = [
    (2,2),(2,14),(3,8),(5,3),(5,13),(8,3),(8,5),(8,8),(8,11),(8,13),
    (11,3),(11,13),(13,8),(14,2),(14,14),
]


class C5G7OpenMCModel:
    """
    Builds and runs the C5G7 2D quarter-core benchmark with OpenMC.

    Geometry: 3×3 assembly lattice (UO2 | MOX | Reflector) × 3 rows
    Physics:  7-group multi-group eigenvalue calculation
    Tallies:  51×51 mesh flux [phi_mesh], fission rate

    Parameters
    ----------
    work_dir    : directory where OpenMC writes XML inputs and HDF5 outputs
    n_particles : Monte Carlo particles per batch
    n_batches   : total batches (active + inactive)
    n_inactive  : inactive (source-convergence) batches
    n_mesh      : tally mesh resolution per direction (default 51 = 3×17)
    """

    def __init__(
        self,
        work_dir: str = "runs/openmc_c5g7",
        n_particles: int = 100_000,
        n_batches: int = 300,
        n_inactive: int = 50,
        n_mesh: int = N_MESH,
    ):
        self.work_dir   = Path(work_dir)
        self.n_particles = int(n_particles)
        self.n_batches  = int(n_batches)
        self.n_inactive = int(n_inactive)
        self.n_mesh     = int(n_mesh)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, force: bool = False) -> Tuple[np.ndarray, float]:
        """
        Build the OpenMC model, run it, and return results.

        Returns
        -------
        phi  : ndarray [n_mesh, n_mesh, 7]  scalar flux per group
        keff : float                         k-effective
        """
        import openmc

        sp_file = self._find_statepoint()
        if sp_file and not force:
            logger.info(f"Found existing statepoint: {sp_file}; loading results.")
            return self._parse_results(sp_file)

        logger.info("Building C5G7 OpenMC model…")
        self._build_and_export(openmc)
        logger.info(f"Running OpenMC: {self.n_particles} particles × "
                    f"{self.n_batches} batches ({self.n_inactive} inactive)…")
        try:
            openmc.run(cwd=str(self.work_dir))
        except FileNotFoundError:
            raise RuntimeError(
                "\n\nOpenMC Python package is installed but the 'openmc' executable "
                "was not found on PATH.\n\n"
                "The pip-installed openmc package only contains the Python API;\n"
                "the compiled transport solver binary must be installed separately.\n\n"
                "To get a working OpenMC binary on Windows:\n\n"
                "  Option 1 – conda (easiest, provides pre-compiled binary):\n"
                "    conda create -n openmc_env python=3.11\n"
                "    conda activate openmc_env\n"
                "    conda install -c conda-forge openmc\n"
                "    # Then run this script inside that environment\n\n"
                "  Option 2 – WSL2 (Linux on Windows):\n"
                "    wsl --install                          # PowerShell as admin\n"
                "    # Then inside Ubuntu:\n"
                "    conda install -c conda-forge openmc\n"
                "    cd /mnt/c/Users/Maosen/2026\\ Neurips\n"
                "    python scripts/run_openmc_c5g7.py\n\n"
                "  Option 3 – build from source (advanced):\n"
                "    https://docs.openmc.org/en/stable/quickinstall.html\n"
            )

        sp_file = self._find_statepoint()
        if sp_file is None:
            raise RuntimeError("OpenMC run finished but no statepoint file found.")
        return self._parse_results(sp_file)

    # ── model builder ─────────────────────────────────────────────────────────

    def _build_and_export(self, openmc):
        """Write materials.xml, geometry.xml, settings.xml, tallies.xml."""
        import openmc.mgxs

        # 1 – MGXS library
        lib = _make_mgxs_library(openmc)
        mgxs_path = self.work_dir / "mgxs.h5"
        lib.export_to_hdf5(str(mgxs_path))

        # 2 – Materials
        def _mat(name, xs_name):
            m = openmc.Material(name=name)
            m.set_density('macro', 1.0)
            m.add_macroscopic(openmc.Macroscopic(xs_name))
            return m

        mat_uo2   = _mat('UO2',            'uo2')
        mat_mox43 = _mat('MOX 4.3%',       'mox43')
        mat_mox7  = _mat('MOX 7.0%',       'mox7')
        mat_mox87 = _mat('MOX 8.7%',       'mox87')
        mat_fc    = _mat('Fission Chamber','fiss_chamber')
        mat_gt    = _mat('Guide Tube',     'guide_tube')
        mat_water = _mat('Water',          'water')

        materials_file = openmc.Materials(
            [mat_uo2, mat_mox43, mat_mox7, mat_mox87, mat_fc, mat_gt, mat_water])
        materials_file.cross_sections = str(mgxs_path)
        materials_file.export_to_xml(str(self.work_dir / 'materials.xml'))

        # 3 – Geometry: pin universes → assembly lattices → core lattice
        p = PIN_PITCH

        def _pin_cell(fuel_mat, pin_r=0.54):
            """Build a single pin-cell universe with fuel + water."""
            fuel_cyl = openmc.ZCylinder(r=pin_r)
            fuel_cell  = openmc.Cell(fill=fuel_mat, region=-fuel_cyl)
            water_cell = openmc.Cell(fill=mat_water, region=+fuel_cyl)
            u = openmc.Universe(cells=[fuel_cell, water_cell])
            return u

        u_uo2   = _pin_cell(mat_uo2)
        u_mox43 = _pin_cell(mat_mox43)
        u_mox7  = _pin_cell(mat_mox7)
        u_mox87 = _pin_cell(mat_mox87)
        u_fc    = _pin_cell(mat_fc)
        u_gt    = _pin_cell(mat_gt)
        u_water = openmc.Universe(cells=[openmc.Cell(fill=mat_water)])

        def _assembly_lattice(pin_grid_fn):
            """Build a 17×17 assembly lattice from a grid-function."""
            lat = openmc.RectLattice()
            lat.lower_left = [-N_PINS * p / 2, -N_PINS * p / 2]
            lat.pitch      = [p, p]
            lat.universes  = [[pin_grid_fn(r, c) for c in range(N_PINS)]
                               for r in range(N_PINS)]
            return lat

        def _uo2_grid(r, c):
            if (r, c) == (8, 8):
                return u_fc
            if (r, c) in _GT_POSITIONS:
                return u_gt
            return u_uo2

        def _mox_grid(r, c):
            if (r, c) == (8, 8):
                return u_fc
            if (r, c) in _GT_POSITIONS:
                return u_gt
            d = min(r, N_PINS-1-r, c, N_PINS-1-c)
            if d <= 1:
                return u_mox87
            elif d <= 3:
                return u_mox7
            return u_mox43

        def _water_grid(r, c):
            return u_water

        lat_uo2   = _assembly_lattice(_uo2_grid)
        lat_mox   = _assembly_lattice(_mox_grid)
        lat_water = _assembly_lattice(_water_grid)

        asm_sz = N_PINS * p  # = ASM_PITCH = 21.42 cm
        asm_bnd_xy = openmc.model.RectangularPrism(
            width=asm_sz, height=asm_sz, origin=(0,0))

        def _asm_universe(lattice):
            cell = openmc.Cell(fill=lattice, region=-asm_bnd_xy)
            return openmc.Universe(cells=[cell])

        u_asm_uo2   = _asm_universe(lat_uo2)
        u_asm_mox   = _asm_universe(lat_mox)
        u_asm_water = _asm_universe(lat_water)

        # Quarter-core: 3×3 assembly lattice
        # Layout (row 0 = top = y > 0):  UO2 | MOX | Reflector
        #                                  MOX | UO2 | Reflector
        #                                  Ref | Ref | Reflector
        core_lat = openmc.RectLattice()
        core_lat.lower_left = [-HALF_CORE, -HALF_CORE]
        core_lat.pitch      = [ASM_PITCH, ASM_PITCH]
        core_lat.universes  = [
            [u_asm_uo2,   u_asm_mox,   u_asm_water],
            [u_asm_mox,   u_asm_uo2,   u_asm_water],
            [u_asm_water, u_asm_water, u_asm_water],
        ]

        # Bounding surfaces (vacuum BC on all sides)
        xmin = openmc.XPlane(x0=-HALF_CORE, boundary_type='vacuum')
        xmax = openmc.XPlane(x0=+HALF_CORE, boundary_type='vacuum')
        ymin = openmc.YPlane(y0=-HALF_CORE, boundary_type='vacuum')
        ymax = openmc.YPlane(y0=+HALF_CORE, boundary_type='vacuum')
        zmin = openmc.ZPlane(z0=-1e6, boundary_type='reflective')
        zmax = openmc.ZPlane(z0=+1e6, boundary_type='reflective')

        core_region = +xmin & -xmax & +ymin & -ymax & +zmin & -zmax
        core_cell   = openmc.Cell(fill=core_lat, region=core_region)
        root_univ   = openmc.Universe(cells=[core_cell])
        geometry    = openmc.Geometry(root_univ)
        geometry.export_to_xml(str(self.work_dir / 'geometry.xml'))

        # 4 – Settings
        settings = openmc.Settings()
        settings.energy_mode    = 'multi-group'
        settings.cross_sections = str(mgxs_path)
        settings.batches        = self.n_batches
        settings.inactive       = self.n_inactive
        settings.particles      = self.n_particles
        settings.run_mode       = 'eigenvalue'
        settings.output         = {'tallies': True, 'summary': True}
        # Initial source: uniform in fissionable region (quarter of core)
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Box(
                [-HALF_CORE, -HALF_CORE, -1.0],
                [0.0,        0.0,         1.0]),
            constraints={'fissionable': True})
        settings.export_to_xml(str(self.work_dir / 'settings.xml'))

        # 5 – Tallies: 51×51 mesh, 7-group flux + fission rate
        mesh = openmc.RegularMesh()
        mesh.dimension   = [self.n_mesh, self.n_mesh, 1]
        mesh.lower_left  = [-HALF_CORE, -HALF_CORE, -1e6]
        mesh.upper_right = [+HALF_CORE, +HALF_CORE, +1e6]

        import openmc.mgxs
        e_filter   = openmc.EnergyFilter(GROUP_BOUNDS)
        mesh_filter = openmc.MeshFilter(mesh)

        flux_tally = openmc.Tally(name='flux')
        flux_tally.filters = [mesh_filter, e_filter]
        flux_tally.scores   = ['flux']

        fiss_tally = openmc.Tally(name='fission')
        fiss_tally.filters = [mesh_filter, e_filter]
        fiss_tally.scores   = ['fission', 'nu-fission']

        tallies_file = openmc.Tallies([flux_tally, fiss_tally])
        tallies_file.export_to_xml(str(self.work_dir / 'tallies.xml'))

        logger.info(f"Model exported to {self.work_dir}/")

    # ── result parser ─────────────────────────────────────────────────────────

    def _find_statepoint(self) -> Optional[Path]:
        files = sorted(self.work_dir.glob('statepoint.*.h5'))
        return files[-1] if files else None

    def _parse_results(self, sp_file: Path) -> Tuple[np.ndarray, float]:
        """
        Parse statepoint and return:
          phi  : [n_mesh, n_mesh, 7]  mean scalar flux per group (normalised)
          keff : float
        """
        import openmc
        sp = openmc.StatePoint(str(sp_file))

        # k-effective
        keff = float(sp.keff.n)
        logger.info(f"  k-effective = {keff:.6f} ± {sp.keff.s:.6f}")

        # Flux tally: shape [mesh_x, mesh_y, 1, 7] after reshaping
        tally  = sp.get_tally(name='flux')
        # get_values returns shape [n_realizations, n_scores, n_filters...]
        # We want mean: shape [n_mesh*n_mesh*1, 7] → [n_mesh, n_mesh, 7]
        flux   = tally.get_values(scores=['flux'])            # [1, 1, Ncells*G]
        n      = self.n_mesh
        G      = 7
        phi    = np.abs(flux.ravel()).reshape(n, n, G).astype(np.float32)

        # Normalise so maximum group-integrated flux = 1
        phi_tot = phi.sum(axis=-1, keepdims=True)
        phi     = phi / (phi_tot.max() + 1e-30)

        logger.info(f"  Flux shape: {phi.shape}, max={phi.max():.4e}")
        return phi, keff

    def save_flux_npy(self, phi: np.ndarray, out_path: str = "runs/openmc_c5g7/flux.npy"):
        """Save the flux array for use by the C5G7Converter."""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, phi)
        logger.info(f"Saved flux to {out_path}  shape={phi.shape}")
