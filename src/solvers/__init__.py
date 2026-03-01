from .mock_backend import MockSolver
from .opensn_interface import OpenSnInterface
from .openmc_interface import OpenMCInterface

# Mapping: benchmark → preferred real solver (in priority order)
_BENCHMARK_SOLVER_PREFERENCE = {
    "c5g7":      ["openmc"],
    "pinte2009": [],   # no supported real solver yet (needs MCFOST/RADMC-3D)
}


def detect_best_solver(benchmark: str) -> str:
    """
    Return the name of the best available solver for this benchmark.

    Checks in priority order:
      c5g7      → openmc, then mock
      pinte2009 → mock (no supported real solver)

    Always returns "mock" if no real solver binary is found.
    """
    import logging
    log = logging.getLogger(__name__)

    for solver_name in _BENCHMARK_SOLVER_PREFERENCE.get(benchmark, []):
        if solver_name == "openmc":
            s = OpenMCInterface(fallback=False)
            if s.is_available:
                log.info(f"[{benchmark}] Real solver selected: openmc")
                return "openmc"
        elif solver_name == "opensn":
            s = OpenSnInterface(fallback=False)
            if s.is_available:
                log.info(f"[{benchmark}] Real solver selected: opensn")
                return "opensn"

    log.warning(
        f"[{benchmark}] No real solver binary found — falling back to MockSolver "
        "(approximate analytic labels, NOT a reference solution). "
        "Install OpenMC (conda install -c conda-forge openmc) for real transport solutions."
    )
    return "mock"


def get_solver(name: str, benchmark: str = "", **kwargs):
    """
    Factory function to get a solver by name.

    Pass name="auto" to automatically select the best available solver
    for the given benchmark (real solver preferred over mock).
    """
    if name == "auto":
        name = detect_best_solver(benchmark)

    if name == "mock":
        # MockSolver only accepts its own init params; drop pipeline-level keys
        mock_kwargs = {k: v for k, v in kwargs.items() if k not in ("benchmark", "fallback")}
        return MockSolver(**mock_kwargs)
    elif name == "openmc":
        return OpenMCInterface(**kwargs)
    else:
        raise ValueError(f"Unknown solver: {name}. Choose from: auto, mock, openmc")
