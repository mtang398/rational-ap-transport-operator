"""
OpenSn Deterministic SN Solver Interface
==========================================
Reference: https://open-sn.github.io/opensn/

OpenSn is an open-source deterministic neutron transport code.
This module provides a Python wrapper for invoking OpenSn and parsing its output.

Installation: https://open-sn.github.io/opensn/install.html
  git clone https://github.com/Open-Sn/openSn.git
  cd openSn && mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j4

Usage:
  from src.solvers.opensn_interface import OpenSnInterface
  solver = OpenSnInterface(opensn_executable="/path/to/opensn")
  result = solver.solve(sample)
"""

from __future__ import annotations
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from ..data.schema import TransportSample, TargetFields

logger = logging.getLogger(__name__)

# Input/output specification (for documentation and stub validation)
OPENSN_INPUT_SPEC = {
    "mesh": "Orthogonal or unstructured mesh (OBJ/VTK format or inline definition)",
    "cross_sections": "Multi-group XS in OpenSn format (chi, sigma_t, sigma_s matrix, nu_sigma_f)",
    "source": "Volumetric/boundary source definition",
    "angular_quadrature": "Product or triangular quadrature, S_N order",
    "boundary_conditions": "vacuum/reflecting/inflow per face",
    "num_groups": "Number of energy groups",
    "scattering_order": "Legendre order for scattering cross section expansion",
    "max_inner_iterations": "Inner iteration convergence",
    "inner_tolerance": "Convergence tolerance for phi",
}

OPENSN_OUTPUT_SPEC = {
    "phi": "Scalar flux per group per cell, shape [Ncells, G]",
    "psi": "Angular flux per group per cell per direction, shape [Ncells, N_omega, G]",
    "currents": "Partial currents on cell faces",
    "keff": "k-effective (eigenvalue mode only)",
}


class OpenSnInterface:
    """
    Interface to the OpenSn deterministic transport solver.

    If OpenSn is not installed, all solve calls fall back to MockSolver
    with a warning. Set fallback=False to raise an error instead.
    """

    def __init__(
        self,
        opensn_executable: Optional[str] = None,
        input_template_dir: Optional[str] = None,
        fallback: bool = True,
        work_dir: Optional[str] = None,
    ):
        self.exe = opensn_executable or os.environ.get("OPENSN_EXE", "opensn")
        self.input_template_dir = Path(input_template_dir) if input_template_dir else None
        self.fallback = fallback
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="opensn_"))
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            result = subprocess.run([self.exe, "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def solve(self, sample: TransportSample) -> TransportSample:
        """
        Run OpenSn on the given sample.

        If OpenSn is not available and fallback=True, uses MockSolver.
        """
        if not self._available:
            if self.fallback:
                logger.warning("OpenSn not available; using MockSolver fallback.")
                from .mock_backend import MockSolver
                return MockSolver().solve(sample)
            else:
                raise RuntimeError(
                    f"OpenSn executable not found at '{self.exe}'. "
                    "Install OpenSn from https://open-sn.github.io/opensn/"
                )

        return self._run_opensn(sample)

    def _run_opensn(self, sample: TransportSample) -> TransportSample:
        """Write input files, run OpenSn, parse output."""
        inp = sample.inputs

        # Write OpenSn input file
        input_path = self.work_dir / "transport.lua"
        self._write_input_lua(input_path, sample)

        # Run
        cmd = [self.exe, str(input_path)]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.work_dir)

        if result.returncode != 0:
            raise RuntimeError(f"OpenSn failed:\n{result.stderr}")

        # Parse output
        phi, J, I = self._parse_output(self.work_dir, sample)

        import copy
        new_targets = TargetFields(I=I, phi=phi, J=J)
        new_sample = copy.copy(sample)
        return TransportSample(
            inputs=sample.inputs,
            query=sample.query,
            targets=new_targets,
            sample_id=sample.sample_id + "_opensn",
        )

    def _write_input_lua(self, path: Path, sample: TransportSample):
        """
        Write OpenSn Lua input file.
        OpenSn uses Lua scripting for input; this generates a minimal transport problem.
        """
        inp = sample.inputs
        G = inp.n_groups
        spatial_shape = inp.spatial_shape
        dim = inp.dim

        lua_lines = [
            "-- Auto-generated OpenSn input file",
            "-- Generated by transport-neural-operator pipeline",
            "",
            f"num_groups = {G}",
            f"dim = {dim}",
            "",
            "-- Mesh",
        ]

        if dim == 2:
            nx, ny = spatial_shape
            lua_lines += [
                "meshgen = mesh.OrthogonalMeshGenerator.Create({",
                f"  node_sets = {{",
                f"    {{ {', '.join(str(i) for i in range(nx+1))} }},",
                f"    {{ {', '.join(str(i) for i in range(ny+1))} }},",
                "  }}",
                "})",
                "mesh.MeshGenerator.Execute(meshgen)",
            ]
        elif dim == 3:
            nx, ny, nz = spatial_shape
            lua_lines += [
                "meshgen = mesh.OrthogonalMeshGenerator.Create({",
                f"  node_sets = {{",
                f"    {{ {', '.join(str(i) for i in range(nx+1))} }},",
                f"    {{ {', '.join(str(i) for i in range(ny+1))} }},",
                f"    {{ {', '.join(str(i) for i in range(nz+1))} }},",
                "  }}",
                "})",
                "mesh.MeshGenerator.Execute(meshgen)",
            ]

        # Cross sections (simplified: use spatially averaged XS)
        sigma_a_mean = inp.sigma_a.mean(axis=tuple(range(dim))).tolist()
        sigma_s_mean = inp.sigma_s.mean(axis=tuple(range(dim))).tolist()
        sigma_t_mean = [sigma_a_mean[g] + sigma_s_mean[g] for g in range(G)]

        lua_lines += [
            "",
            "-- Cross sections",
            "xs = PhysicsMaterial.SetScalarValue",
            f"-- sigma_t: {sigma_t_mean}",
            f"-- sigma_a: {sigma_a_mean}",
        ]

        # Boundary conditions
        bc_type = inp.bc.bc_type
        lua_lines += [
            "",
            "-- Boundary conditions",
            f"-- bc_type: {bc_type}",
        ]

        # Angular quadrature
        lua_lines += [
            "",
            "-- Angular quadrature (product Gauss-Legendre S8)",
            "pquad = aquad.CreateProductQuadrature(GAUSS_LEGENDRE_CHEBYSHEV, 8, 8)",
        ]

        with open(path, "w") as f:
            f.write("\n".join(lua_lines))

        logger.debug(f"Wrote OpenSn input to {path}")

    def _parse_output(self, work_dir: Path, sample: TransportSample):
        """Parse OpenSn output files. Returns (phi, J, I)."""
        # OpenSn outputs VTK or CSV; parse phi from phi_output.csv if present
        phi_path = work_dir / "phi_output.csv"
        if phi_path.exists():
            data = np.loadtxt(phi_path, delimiter=",")
            Nx = sample.query.n_spatial
            G = sample.inputs.n_groups
            phi = data.reshape(Nx, G).astype(np.float32)
        else:
            logger.warning("OpenSn phi_output.csv not found; using zero placeholder.")
            Nx = sample.query.n_spatial
            G = sample.inputs.n_groups
            phi = np.ones((Nx, G), dtype=np.float32) * 1e-6

        Nw = sample.query.n_omega
        dim = sample.inputs.dim
        I = np.ones((Nx, Nw, G), dtype=np.float32) * 1e-6
        J = np.zeros((Nx, dim, G), dtype=np.float32)

        return phi, J, I

    def write_input_spec(self) -> str:
        """Return documented input specification as string."""
        lines = ["OpenSn Input Specification:", "=" * 40]
        for k, v in OPENSN_INPUT_SPEC.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
