from .mock_backend import MockSolver
from .opensn_interface import OpenSnInterface
from .openmc_interface import OpenMCInterface


def get_solver(name: str, **kwargs):
    """Factory function to get a solver by name."""
    if name == "mock":
        return MockSolver(**kwargs)
    elif name == "opensn":
        return OpenSnInterface(**kwargs)
    elif name == "openmc":
        return OpenMCInterface(**kwargs)
    else:
        raise ValueError(f"Unknown solver: {name}. Choose from: mock, opensn, openmc")
