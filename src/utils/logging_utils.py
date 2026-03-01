"""Logging setup utilities."""
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
):
    """Configure root logger."""
    if format_str is None:
        format_str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_str,
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Data-source classification
# ---------------------------------------------------------------------------

# Benchmarks for which a real solver dataset can be present on disk
_REAL_SOLVER_BENCHMARKS = {
    "c5g7":      "OpenMC Monte Carlo (openmc_multigroup_mc)",
    "pinte2009": "MCFOST / RADMC-3D Monte Carlo RT",
}

_APPROX_METHODS = {
    "c5g7":      "multigroup diffusion (analytic cosh/exp)",
    "pinte2009": "5-step Λ-iteration radiative transfer",
    "mock":      "fully synthetic make_mock_sample (exp-decay φ)",
}


def _detect_dataset_source(benchmark_name: str, data_dir: Optional[str] = None) -> str:
    """
    Inspect the dataset on disk (if any) and return a human-readable string
    describing whether the flux targets are from a real solver or an approximation.

    Looks for `flux_source` in the HDF5/Zarr metadata of the first sample.
    Falls back to the known approximation for that benchmark if no file is found
    or no metadata key is present.
    """
    real_tag = "openmc_multigroup_mc"  # set by run_openmc_c5g7.py

    if data_dir is not None:
        zarr_path = Path(data_dir) / f"{benchmark_name}_train.zarr"
        h5_path   = Path(data_dir) / f"{benchmark_name}_train.zarr.h5"
        # A Zarr store is a *directory*; an HDF5 store is a *file*.
        zarr_exists = zarr_path.exists()   # True for both dir and file
        h5_exists   = h5_path.exists() and h5_path.is_file()
        candidate   = zarr_path if zarr_exists else (h5_path if h5_exists else None)

        if candidate is not None:
            # Try to read flux_source metadata to confirm it is real MC/SN data.
            flux_src = None
            try:
                import h5py
                target = candidate if candidate.is_file() else None
                if target is not None:
                    with h5py.File(str(target), "r") as f:
                        for key in f:
                            meta = f[key].get("inputs", {}).get("metadata", {})
                            if hasattr(meta, "get"):
                                flux_src = meta.get("flux_source", None)
                            elif "flux_source" in meta:
                                flux_src = bytes(meta["flux_source"]).decode("utf-8")
                            if flux_src:
                                break
            except Exception:
                pass

            if flux_src == real_tag:
                solver_name = _REAL_SOLVER_BENCHMARKS.get(benchmark_name, flux_src)
                return f"REAL solver output  [{solver_name}]  ← {candidate.name}"

            # File/dir exists but metadata probe inconclusive — report it as
            # cached data and note the source is unknown.
            solver_name = _REAL_SOLVER_BENCHMARKS.get(benchmark_name, "unknown solver")
            return (
                f"Cached disk data found  ← {candidate.name}"
                f"  (source: {solver_name if flux_src else 'unverified — metadata not readable'})"
            )

    # No disk data at all — check whether a real solver is installed so the
    # banner tells the user what will happen when data is generated.
    try:
        from src.solvers import detect_best_solver
        best = detect_best_solver(benchmark_name)
    except Exception:
        best = "mock"

    approx = _APPROX_METHODS.get(benchmark_name, "unknown approximation")
    if best != "mock":
        return (
            f"REAL solver will be used  [{best}]"
            f"  (no cached data yet — run generate_dataset.py first)"
        )
    return f"SYNTHETIC / approximate  [{approx}]  (no real solver installed)"


def log_environment_info(
    benchmark_name: str,
    model_name: Optional[str] = None,
    data_dir: Optional[str] = None,
    dataset_type: Optional[str] = None,
    source_override: Optional[str] = None,
):
    """
    Print a clearly formatted banner showing:
      - GPU / CPU device being used
      - Data source: REAL (solver output) vs SYNTHETIC (approximation)
      - Benchmark + model being run

    Call once at the start of train.py, eval.py, run_all.py, and sweep.py.

    Parameters
    ----------
    benchmark_name : e.g. "c5g7", "pinte2009", "mock"
    model_name     : e.g. "ap_micromacro", "fno", "deeponet"  (optional)
    data_dir       : path to runs/datasets (optional, used for disk inspection)
    dataset_type   : "mock" or "disk" — if "mock" is passed we skip disk probe
    """
    import torch

    sep = "=" * 68

    # ── GPU info ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_lines = []
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024 ** 3
            gpu_lines.append(
                f"  GPU {i}: {props.name}  "
                f"({mem_gb:.1f} GB, CUDA {props.major}.{props.minor})"
            )
        device_str = f"CUDA  ({n_gpus} device{'s' if n_gpus > 1 else ''})"
        device_lines = gpu_lines
    else:
        device_str = "CPU  (no CUDA GPU detected)"
        device_lines = []

    # ── Data source ───────────────────────────────────────────────────────
    if source_override is not None:
        data_source = source_override
    elif dataset_type == "mock" or benchmark_name == "mock":
        data_source = (
            "SYNTHETIC / in-memory MockDataset  "
            "[make_mock_sample — analytic exp-decay φ, no solver]"
        )
    else:
        data_source = _detect_dataset_source(benchmark_name, data_dir)

    # ── Print banner ──────────────────────────────────────────────────────
    lines = [
        sep,
        "  ENVIRONMENT SUMMARY",
        sep,
        f"  Benchmark   : {benchmark_name}" + (f"   Model: {model_name}" if model_name else ""),
        f"  Device      : {device_str}",
    ]
    lines += device_lines
    lines += [
        f"  Data source : {data_source}",
        sep,
    ]
    for line in lines:
        logger.info(line)
