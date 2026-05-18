from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from typing import Any, Dict

import filelock
import h5py
import numpy as np
import torch
from atria_logger._api import get_logger
from PIL import Image as PILImageModule
from PIL.Image import Image as PILImage
from pydantic import BaseModel, ConfigDict

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cross-process + cross-thread locking
# ---------------------------------------------------------------------------

# One threading.Lock per HDF5 path, shared across all calls in this process.
# Protected by a mutex so the dict itself is thread-safe to update.
_thread_locks: dict[Path, threading.Lock] = {}
_thread_locks_mutex = threading.Lock()


def _get_thread_lock(path: Path) -> threading.Lock:
    with _thread_locks_mutex:
        if path not in _thread_locks:
            _thread_locks[path] = threading.Lock()
        return _thread_locks[path]


def _hdf5_lock(hdf5_path: Path) -> tuple[threading.Lock, filelock.FileLock]:
    """Return (thread_lock, file_lock) for the given HDF5 path.

    Usage::

        thread_lock, file_lock = _hdf5_lock(path)
        with thread_lock, file_lock:
            with h5py.File(path, "r") as f:
                ...
    """
    return _get_thread_lock(hdf5_path), filelock.FileLock(
        hdf5_path.with_suffix(".lock")
    )


# ---------------------------------------------------------------------------
# Internal helpers – type conversion
# ---------------------------------------------------------------------------


def _tensor_to_numpy_recursive(tensor: Any) -> Any:
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    elif isinstance(tensor, dict):
        return {key: _tensor_to_numpy_recursive(value) for key, value in tensor.items()}
    elif isinstance(tensor, list):
        return [_tensor_to_numpy_recursive(item) for item in tensor]
    elif isinstance(tensor, tuple):
        return tuple(_tensor_to_numpy_recursive(item) for item in tensor)
    else:
        return tensor


def _numpy_to_item_recursive(array: Any) -> Any:
    if isinstance(array, np.ndarray) and array.shape == ():
        return array.item()
    elif isinstance(array, dict):
        return {key: _numpy_to_item_recursive(value) for key, value in array.items()}
    elif isinstance(array, list):
        return [_numpy_to_item_recursive(item) for item in array]
    elif isinstance(array, tuple):
        return tuple(_numpy_to_item_recursive(item) for item in array)
    else:
        return array


def _to_json_serializable(value: Any) -> Any:
    """Recursively convert a value to a JSON-serializable Python type."""
    import enum

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# HDF5 helpers – recursive read/write for arbitrary nested dicts
# ---------------------------------------------------------------------------


def _write_dict_to_hdf5_group(group: h5py.Group, data: Dict[str, Any]) -> None:
    """Recursively write a dict into an HDF5 group.

    - numpy arrays / torch tensors  → datasets
    - nested dicts                  → subgroups (recursive)
    - lists / other scalars         → JSON-encoded string dataset
    """
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            group.create_dataset(key, data=value.detach().cpu().numpy())
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif isinstance(value, dict):
            subgroup = group.require_group(key)
            _write_dict_to_hdf5_group(subgroup, value)
        else:
            serialised = json.dumps(_to_json_serializable(value))
            ds = group.create_dataset(key, data=np.bytes_(serialised))
            ds.attrs["__json__"] = True


def _read_dict_from_hdf5_group(group: h5py.Group) -> Dict[str, Any]:
    """Reverse of _write_dict_to_hdf5_group."""
    result: Dict[str, Any] = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            result[key] = _read_dict_from_hdf5_group(item)
        else:
            if item.attrs.get("__json__", False):
                raw = item[()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                elif isinstance(raw, np.bytes_):
                    raw = raw.tobytes().decode("utf-8")
                result[key] = json.loads(raw)
            else:
                arr = item[()]
                result[key] = arr.item() if arr.ndim == 0 else arr
    return result


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _image_to_png_bytes(image: PILImage) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _png_bytes_to_image(data: bytes) -> PILImage:
    return PILImageModule.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# Path models
# ---------------------------------------------------------------------------


class SampleSummaryPaths(BaseModel):
    root_dir: Path
    hdf5_path: Path

    def exists(self) -> bool:
        return self.hdf5_path.exists()


class ExplanationSummaryPaths(BaseModel):
    root_dir: Path
    hdf5_path: Path

    def exists(self) -> bool:
        return self.hdf5_path.exists()


# ---------------------------------------------------------------------------
# SampleSummary
# ---------------------------------------------------------------------------

_SAMPLE_HDF5_FILENAME = "sample_summaries.h5"


class SampleSummary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: str
    metadata: Dict[str, Any]
    image: PILImage

    @classmethod
    def build_paths(cls, root_dir: Path) -> SampleSummaryPaths:
        return SampleSummaryPaths(
            root_dir=root_dir,
            hdf5_path=root_dir / _SAMPLE_HDF5_FILENAME,
        )

    @classmethod
    def get_paths(cls, root_dir: Path) -> SampleSummaryPaths:
        return cls.build_paths(root_dir)

    @classmethod
    def exists(cls, sample_id: str, root_dir: Path) -> bool:
        paths = cls.get_paths(root_dir)
        if not paths.hdf5_path.exists():
            return False
        thread_lock, file_lock = _hdf5_lock(paths.hdf5_path)
        with thread_lock, file_lock:
            with h5py.File(paths.hdf5_path, "r") as f:
                return sample_id in f

    def save_to_disk(self, root_dir: Path, save_image: bool = True) -> None:
        paths = self.get_paths(root_dir)
        paths.root_dir.mkdir(parents=True, exist_ok=True)
        thread_lock, file_lock = _hdf5_lock(paths.hdf5_path)
        with thread_lock, file_lock:
            with h5py.File(paths.hdf5_path, "a") as f:
                if self.sample_id in f:
                    return
                grp = f.require_group(self.sample_id)
                grp.attrs["metadata"] = json.dumps(_to_json_serializable(self.metadata))
                if save_image:
                    png_bytes = _image_to_png_bytes(self.image)
                    grp.create_dataset(
                        "image",
                        data=np.frombuffer(png_bytes, dtype=np.uint8),
                    )

    @classmethod
    def load_from_disk(
        cls, sample_id: str, root_dir: Path, load_image: bool = True
    ) -> "SampleSummary":
        paths = cls.build_paths(root_dir)
        thread_lock, file_lock = _hdf5_lock(paths.hdf5_path)
        with thread_lock, file_lock:
            with h5py.File(paths.hdf5_path, "r") as f:
                if sample_id not in f:
                    raise KeyError(
                        f"sample_id '{sample_id}' not found in {paths.hdf5_path}"
                    )
                grp = f[sample_id]
                metadata = json.loads(grp.attrs["metadata"])
                if load_image:
                    png_bytes = grp["image"][()].tobytes()
                    image = _png_bytes_to_image(png_bytes)
                else:
                    image = None
        return cls(sample_id=sample_id, metadata=metadata, image=image)

    @classmethod
    def list_sample_ids(cls, root_dir: Path) -> list[str]:
        paths = cls.build_paths(root_dir)
        if not paths.hdf5_path.exists():
            return []
        thread_lock, file_lock = _hdf5_lock(paths.hdf5_path)
        with thread_lock, file_lock:
            with h5py.File(paths.hdf5_path, "r") as f:
                return list(f.keys())


# ---------------------------------------------------------------------------
# ExplanationSummary
# ---------------------------------------------------------------------------

_EXPLANATION_HDF5_FILENAME = "explanation_summaries.h5"


class ExplanationSummary(BaseModel):
    """Fully serializable summary using HDF5 storage.

    ``explanation_units`` is either:
    - ``Dict[str, Any]`` for single-target (keys are unit names).
    - ``List[Dict[str, Any]]`` for multi-target (one dict per target, ordered
      by target index).
    """

    sample_id: str
    metadata: Dict[str, Any]
    explanation_units: Dict[str, Any] | list[Dict[str, Any]] | None = None

    @classmethod
    def build_paths(cls, root_dir: Path) -> ExplanationSummaryPaths:
        return ExplanationSummaryPaths(
            root_dir=root_dir,
            hdf5_path=root_dir / _EXPLANATION_HDF5_FILENAME,
        )

    @classmethod
    def get_paths(cls, root_dir: Path) -> ExplanationSummaryPaths:
        return cls.build_paths(root_dir)

    @classmethod
    def load_metadata_from_disk(cls, sample_id: str, root_dir: Path) -> Dict[str, Any]:
        paths = cls.build_paths(root_dir)
        thread_lock, file_lock = _hdf5_lock(paths.hdf5_path)
        with thread_lock, file_lock:
            with h5py.File(paths.hdf5_path, "r") as f:
                if sample_id not in f:
                    raise KeyError(
                        f"sample_id '{sample_id}' not found in {paths.hdf5_path}"
                    )
                grp = f[sample_id]
                metadata = json.loads(grp.attrs["metadata"])
        return metadata

    @classmethod
    def exists(cls, sample_id: str, root_dir: Path) -> bool:
        paths = cls.get_paths(root_dir)
        if not paths.hdf5_path.exists():
            return False
        thread_lock, file_lock = _hdf5_lock(paths.hdf5_path)
        with thread_lock, file_lock:
            with h5py.File(paths.hdf5_path, "r") as f:
                return sample_id in f

    def save_to_disk(self, root_dir: Path, save_explanations: bool = True) -> None:
        paths = self.get_paths(root_dir)
        paths.root_dir.mkdir(parents=True, exist_ok=True)
        thread_lock, file_lock = _hdf5_lock(paths.hdf5_path)
        with thread_lock, file_lock:
            with h5py.File(paths.hdf5_path, "a") as f:
                if self.sample_id in f:
                    del f[self.sample_id]
                grp = f.require_group(self.sample_id)
                grp.attrs["metadata"] = json.dumps(_to_json_serializable(self.metadata))
                if save_explanations:
                    units_grp = grp.require_group("explanation_units")
                    assert self.explanation_units is not None, (
                        "explanation_units cannot be None if save_explanations=True"
                    )
                    if isinstance(self.explanation_units, list):
                        units_grp.attrs["__is_list__"] = True
                        units_grp.attrs["__list_len__"] = len(self.explanation_units)
                        for i, target_dict in enumerate(self.explanation_units):
                            target_grp = units_grp.require_group(str(i))
                            _write_dict_to_hdf5_group(target_grp, target_dict)
                    else:
                        units_grp.attrs["__is_list__"] = False
                        _write_dict_to_hdf5_group(units_grp, self.explanation_units)

    @classmethod
    def load_from_disk(
        cls, sample_id: str, root_dir: Path, load_explanations: bool = True
    ) -> "ExplanationSummary":
        paths = cls.build_paths(root_dir)
        thread_lock, file_lock = _hdf5_lock(paths.hdf5_path)
        with thread_lock, file_lock:
            with h5py.File(paths.hdf5_path, "r") as f:
                if sample_id not in f:
                    raise KeyError(
                        f"sample_id '{sample_id}' not found in {paths.hdf5_path}"
                    )
                grp = f[sample_id]
                metadata = json.loads(grp.attrs["metadata"])
                if load_explanations:
                    units_grp = grp["explanation_units"]
                    is_list = bool(units_grp.attrs.get("__is_list__", False))
                    if is_list:
                        n = int(units_grp.attrs["__list_len__"])
                        explanation_units: list[Dict[str, Any]] | Dict[str, Any] = [
                            _read_dict_from_hdf5_group(units_grp[str(i)])
                            for i in range(n)
                        ]
                    else:
                        explanation_units = _read_dict_from_hdf5_group(units_grp)
                else:
                    explanation_units = None
        return cls(
            sample_id=sample_id,
            metadata=metadata,
            explanation_units=explanation_units,
        )

    @classmethod
    def list_sample_ids(cls, root_dir: Path) -> list[str]:
        paths = cls.build_paths(root_dir)
        if not paths.hdf5_path.exists():
            return []
        thread_lock, file_lock = _hdf5_lock(paths.hdf5_path)
        with thread_lock, file_lock:
            with h5py.File(paths.hdf5_path, "r") as f:
                return list(f.keys())
