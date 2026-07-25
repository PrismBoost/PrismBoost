"""Pickle/save support for sklearn estimators backed by ``_sefr_boost_core``."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, TypeVar

from ._cpp_backend import CPP_AVAILABLE, SEFRBoostClassifierCore, SEFRBoostRegressorCore

T = TypeVar("T")


def _pop_cpp_core_for_pickle(state: dict[str, Any], bytes_key: str, core: Any) -> None:
    state[bytes_key] = core.to_bytes()


def _restore_cpp_core_from_pickle(
    state: dict[str, Any],
    bytes_key: str,
    core_cls: type,
    core_key: str = "_cpp_core_",
) -> None:
    blob = state.pop(bytes_key, None)
    if blob is not None:
        state[core_key] = core_cls.from_bytes(blob)


def cpp_estimator_getstate(self, *, core_cls, bytes_key: str) -> dict[str, Any]:
    state = self.__dict__.copy()
    core = state.pop("_cpp_core_", None)
    if core is not None:
        _pop_cpp_core_for_pickle(state, bytes_key, core)
    return state


def cpp_estimator_setstate(self, state: dict[str, Any], *, core_cls, bytes_key: str) -> None:
    _restore_cpp_core_from_pickle(state, bytes_key, core_cls)
    self.__dict__.update(state)


def save_estimator(estimator: T, path: str | Path) -> Path:
    """Pickle a fitted estimator (including C++ core bytes) to ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(estimator, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_estimator(path: str | Path) -> Any:
    """Load an estimator saved with :func:`save_estimator`."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def serialized_size_bytes(obj: Any) -> int:
    """Return the pickled serialized size in bytes (benchmark-compatible)."""
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


__all__ = [
    "CPP_AVAILABLE",
    "cpp_estimator_getstate",
    "cpp_estimator_setstate",
    "load_estimator",
    "save_estimator",
    "serialized_size_bytes",
]
