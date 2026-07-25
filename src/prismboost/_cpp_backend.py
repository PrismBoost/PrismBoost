"""Optional C++ backend for PrismBoost (``prismboost._sefr_boost_core``)."""

from __future__ import annotations

CPP_AVAILABLE = False
CPP_IMPORT_ERROR: Exception | None = None

try:
    from ._sefr_boost_core import SEFRBoostClassifierCore, SEFRBoostRegressorCore

    CPP_AVAILABLE = True
except ImportError as exc:
    SEFRBoostClassifierCore = None  # type: ignore[misc, assignment]
    SEFRBoostRegressorCore = None  # type: ignore[misc, assignment]
    CPP_IMPORT_ERROR = exc


def cpp_backend_status() -> str:
    if CPP_AVAILABLE:
        return "C++ backend available"
    return f"C++ backend unavailable: {CPP_IMPORT_ERROR}"
