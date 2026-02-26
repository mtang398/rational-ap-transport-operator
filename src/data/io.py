"""
Zarr and HDF5 dataset I/O with schema versioning and sanity checks.

Backend selection:
  - Windows  → h5py  (zarr v3 has atomic-rename + read-only ZipStore bugs on Windows)
  - Linux/macOS → zarr v3 LocalStore or zarr v2 DirectoryStore

You can force a backend: TRANSPORT_IO_BACKEND=h5py or TRANSPORT_IO_BACKEND=zarr
"""

from __future__ import annotations
import json
import os
import platform
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np

from .schema import TransportSample, SCHEMA_VERSION

# ── dependency checks ──────────────────────────────────────────────────────────
try:
    import zarr
    ZARR_AVAILABLE = True
    _ZARR_V3 = int(zarr.__version__.split(".")[0]) >= 3
except ImportError:
    ZARR_AVAILABLE = False
    _ZARR_V3 = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

_ON_WINDOWS = platform.system() == "Windows"

# Pick default backend: h5py on Windows, zarr elsewhere
_ENV_BACKEND = os.environ.get("TRANSPORT_IO_BACKEND", "").lower()
if _ENV_BACKEND in ("h5py", "zarr"):
    _DEFAULT_BACKEND = _ENV_BACKEND
elif _ON_WINDOWS:
    _DEFAULT_BACKEND = "h5py" if H5PY_AVAILABLE else "zarr"
else:
    _DEFAULT_BACKEND = "zarr" if ZARR_AVAILABLE else "h5py"


def _h5_path(path: str) -> str:
    """Ensure h5py path ends with .h5"""
    p = str(path)
    if not (p.endswith(".h5") or p.endswith(".hdf5")):
        return p + ".h5"
    return p


def _zarr_path(path: str) -> str:
    """Keep zarr path as-is (directory store)."""
    return path


# ── zarr helpers (non-Windows only) ───────────────────────────────────────────

def _open_zarr_group(path: str, mode: str):
    import zarr
    if _ZARR_V3:
        import zarr.storage
        store = zarr.storage.LocalStore(path)
        return zarr.open_group(store=store, mode=mode)
    else:
        store = zarr.DirectoryStore(path)
        return zarr.open_group(store, mode=mode)


# ── Public writer / reader (auto-selects backend) ─────────────────────────────

class ZarrDatasetWriter:
    """
    Writes TransportSamples to disk.
    Delegates to _H5Writer or _ZarrWriter based on platform / env var.

    All calling code uses this class; the backend is transparent.
    """

    def __new__(cls, path: Union[str, Path], mode: str = "w", **kwargs):
        if _DEFAULT_BACKEND == "h5py":
            return _H5Writer(_h5_path(str(path)), mode=mode)
        else:
            return _ZarrWriter(_zarr_path(str(path)), mode=mode)


class ZarrDatasetReader:
    """
    Reads TransportSamples from disk.
    Auto-detects backend from file extension (.h5 / .zarr).
    """

    def __new__(cls, path: Union[str, Path]):
        p = str(path)
        # Try h5 variants first
        for candidate in [p, p + ".h5", p.replace(".zarr", "") + ".h5"]:
            if Path(candidate).exists() and (candidate.endswith(".h5") or candidate.endswith(".hdf5")):
                return _H5Reader(candidate)
        # Try zarr variants
        for candidate in [p, p.rstrip("/\\")]:
            if Path(candidate).exists():
                return _ZarrReader(candidate)
        raise FileNotFoundError(
            f"Dataset not found: {p}\n"
            f"  Tried: {p}, {p+'.h5'}\n"
            f"  Make sure you ran generate_dataset.py first."
        )


# ── h5py backend ──────────────────────────────────────────────────────────────

class _H5Writer:
    """h5py-based writer. Works on all platforms."""

    def __init__(self, path: str, mode: str = "w"):
        if not H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required on Windows. Install with: pip install h5py"
            )
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, mode)
        self.n_samples = 0
        self._meta: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "n_samples": 0}
        if mode == "a" and "n_samples" in self.f.attrs:
            self.n_samples = int(self.f.attrs["n_samples"])

    def write(self, sample: TransportSample, idx: Optional[int] = None) -> int:
        if idx is None:
            idx = self.n_samples
        grp = self.f.require_group(f"samples/{idx}")

        def _w(name, arr):
            if arr is None:
                return
            arr = np.asarray(arr, dtype=np.float32)
            if name in grp:
                del grp[name]
            grp.create_dataset(name, data=arr, compression="gzip", compression_opts=4)

        _w("sigma_a", sample.inputs.sigma_a)
        _w("sigma_s", sample.inputs.sigma_s)
        _w("q",       sample.inputs.q)
        for k, v in sample.inputs.extra_fields.items():
            _w(f"extra_{k}", v)
        _w("x",     sample.query.x)
        _w("omega", sample.query.omega)
        if sample.query.w_omega is not None:
            _w("w_omega", sample.query.w_omega)
        if sample.query.t is not None:
            _w("t", sample.query.t)
        _w("I",   sample.targets.I)
        _w("phi", sample.targets.phi)
        _w("J",   sample.targets.J)
        for k, v in sample.targets.qois.items():
            _w(f"qoi_{k}", v)

        grp.attrs["bc"]        = json.dumps(sample.inputs.bc.to_dict())
        grp.attrs["params"]    = json.dumps(sample.inputs.params)
        grp.attrs["metadata"]  = json.dumps(sample.inputs.metadata)
        grp.attrs["sample_id"] = sample.sample_id

        if idx >= self.n_samples:
            self.n_samples = idx + 1
        return idx

    def flush(self, benchmark_name: str = ""):
        self._meta["n_samples"]      = self.n_samples
        self._meta["schema_version"] = SCHEMA_VERSION
        if benchmark_name:
            self._meta["benchmark_name"] = benchmark_name
        self.f.attrs["n_samples"]      = self.n_samples
        self.f.attrs["schema_version"] = SCHEMA_VERSION
        self.f.attrs["_meta"]          = json.dumps(self._meta)
        self.f.flush()

    def close(self):
        self.flush()
        self.f.close()


class _H5Reader:
    """h5py-based reader."""

    def __init__(self, path: str):
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required. pip install h5py")
        self.path = path
        self.f = h5py.File(path, "r")
        try:
            self._meta = json.loads(self.f.attrs.get("_meta", "{}"))
        except Exception:
            self._meta = {}
        self.n_samples = int(self.f.attrs.get("n_samples",
                             self._meta.get("n_samples", 0)))
        self._check_version()

    def _check_version(self):
        sv = self._meta.get("schema_version", "unknown")
        if sv != SCHEMA_VERSION:
            warnings.warn(f"Schema version mismatch: file={sv}, code={SCHEMA_VERSION}")

    def read(self, idx: int) -> TransportSample:
        from .schema import BCSpec, InputFields, QueryPoints, TargetFields
        grp = self.f[f"samples/{idx}"]

        def _r(name):
            return np.array(grp[name], dtype=np.float32) if name in grp else None

        extra = {k[6:]: np.array(grp[k], dtype=np.float32)
                 for k in grp.keys() if k.startswith("extra_")}
        qois  = {k[4:]:  np.array(grp[k], dtype=np.float32)
                 for k in grp.keys() if k.startswith("qoi_")}

        inputs = InputFields(
            sigma_a     = _r("sigma_a"),
            sigma_s     = _r("sigma_s"),
            q           = _r("q"),
            extra_fields= extra,
            bc          = BCSpec.from_dict(json.loads(grp.attrs.get("bc", "{}"))),
            params      = {k: float(v) for k, v in json.loads(grp.attrs.get("params", "{}")).items()},
            metadata    = json.loads(grp.attrs.get("metadata", "{}")),
        )
        query = QueryPoints(
            x       = _r("x"),
            omega   = _r("omega"),
            w_omega = _r("w_omega"),
            t       = _r("t"),
        )
        targets = TargetFields(
            I   = _r("I"),
            phi = _r("phi"),
            J   = _r("J"),
            qois= qois,
        )
        return TransportSample(
            inputs=inputs, query=query, targets=targets,
            sample_id=str(grp.attrs.get("sample_id", f"sample_{idx}")),
        )

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        for i in range(self.n_samples):
            yield self.read(i)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._meta


# ── zarr backend (Linux / macOS) ──────────────────────────────────────────────

class _ZarrWriter:
    def __init__(self, path: str, mode: str = "w"):
        if not ZARR_AVAILABLE:
            raise ImportError("zarr is required. pip install zarr")
        self.path = path
        self.root = _open_zarr_group(path, mode=mode)
        self.n_samples = 0
        self._meta: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "n_samples": 0}
        if mode == "a":
            try:
                stored = self.root.attrs.get("_meta", None)
                if stored:
                    self._meta = json.loads(stored)
                    self.n_samples = self._meta.get("n_samples", 0)
            except Exception:
                pass

    def write(self, sample: TransportSample, idx: Optional[int] = None) -> int:
        if idx is None:
            idx = self.n_samples
        grp = self.root.require_group(f"samples/{idx}")

        def _w(group, name, arr):
            if arr is None:
                return
            arr = np.asarray(arr, dtype=np.float32)
            if name in group:
                del group[name]
            group.create_array(name=name, data=arr, overwrite=True)

        inp = grp.require_group("inputs")
        _w(inp, "sigma_a", sample.inputs.sigma_a)
        _w(inp, "sigma_s", sample.inputs.sigma_s)
        _w(inp, "q",       sample.inputs.q)
        for k, v in sample.inputs.extra_fields.items():
            _w(inp, f"extra_{k}", v)
        inp.attrs["bc"]       = json.dumps(sample.inputs.bc.to_dict())
        inp.attrs["params"]   = json.dumps(sample.inputs.params)
        inp.attrs["metadata"] = json.dumps(sample.inputs.metadata)

        qry = grp.require_group("query")
        _w(qry, "x",     sample.query.x)
        _w(qry, "omega", sample.query.omega)
        if sample.query.w_omega is not None:
            _w(qry, "w_omega", sample.query.w_omega)
        if sample.query.t is not None:
            _w(qry, "t", sample.query.t)

        tgt = grp.require_group("targets")
        _w(tgt, "I",   sample.targets.I)
        _w(tgt, "phi", sample.targets.phi)
        _w(tgt, "J",   sample.targets.J)
        for k, v in sample.targets.qois.items():
            _w(tgt, f"qoi_{k}", v)

        grp.attrs["sample_id"]      = sample.sample_id
        grp.attrs["schema_version"] = SCHEMA_VERSION

        if idx >= self.n_samples:
            self.n_samples = idx + 1
        return idx

    def flush(self, benchmark_name: str = ""):
        self._meta["n_samples"]      = self.n_samples
        self._meta["schema_version"] = SCHEMA_VERSION
        if benchmark_name:
            self._meta["benchmark_name"] = benchmark_name
        self.root.attrs["_meta"] = json.dumps(self._meta)

    def close(self):
        self.flush()


class _ZarrReader:
    def __init__(self, path: str):
        if not ZARR_AVAILABLE:
            raise ImportError("zarr is required. pip install zarr")
        self.path = path
        self.root = _open_zarr_group(path, mode="r")
        try:
            stored = self.root.attrs.get("_meta", None)
            self._meta = json.loads(stored) if stored else {}
        except Exception:
            self._meta = {}
        if not self._meta.get("n_samples"):
            try:
                self._meta["n_samples"] = len(list(self.root["samples"].keys()))
            except Exception:
                self._meta["n_samples"] = 0
        self.n_samples = self._meta.get("n_samples", 0)

    def read(self, idx: int) -> TransportSample:
        from .schema import BCSpec, InputFields, QueryPoints, TargetFields
        grp = self.root[f"samples/{idx}"]

        def _r(group, name):
            return np.array(group[name], dtype=np.float32) if name in group else None

        inp_grp = grp["inputs"]
        inputs = InputFields(
            sigma_a     = _r(inp_grp, "sigma_a"),
            sigma_s     = _r(inp_grp, "sigma_s"),
            q           = _r(inp_grp, "q"),
            extra_fields= {k[6:]: np.array(inp_grp[k], dtype=np.float32)
                           for k in inp_grp.keys() if k.startswith("extra_")},
            bc          = BCSpec.from_dict(json.loads(inp_grp.attrs.get("bc", "{}"))),
            params      = {k: float(v) for k, v in json.loads(inp_grp.attrs.get("params", "{}")).items()},
            metadata    = json.loads(inp_grp.attrs.get("metadata", "{}")),
        )
        qry_grp = grp["query"]
        query = QueryPoints(
            x       = _r(qry_grp, "x"),
            omega   = _r(qry_grp, "omega"),
            w_omega = _r(qry_grp, "w_omega"),
            t       = _r(qry_grp, "t"),
        )
        tgt_grp = grp["targets"]
        targets = TargetFields(
            I    = _r(tgt_grp, "I"),
            phi  = _r(tgt_grp, "phi"),
            J    = _r(tgt_grp, "J"),
            qois = {k[4:]: np.array(tgt_grp[k], dtype=np.float32)
                   for k in tgt_grp.keys() if k.startswith("qoi_")},
        )
        return TransportSample(
            inputs=inputs, query=query, targets=targets,
            sample_id=str(grp.attrs.get("sample_id", f"sample_{idx}")),
        )

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        for i in range(self.n_samples):
            yield self.read(i)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._meta
