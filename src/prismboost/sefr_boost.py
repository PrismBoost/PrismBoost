"""Legacy import path for PrismBoost, kept for backwards compatibility.

The implementation lives in :mod:`prismboost.prism_boost`; this module
re-exports it under the ``SEFRBoost*`` names used before the rename. New code
should use :class:`prismboost.PrismBoostClassifier`.
"""

from typing import Any

from . import prism_boost as _prism_boost
from .prism_boost import (
    AUTO_PARAM_NAMES,
    SPLIT_MODE_OPTIONS,
    PrismBoostClassifier,
    PrismBoostRegressor,
    SEFRBoostClassifier,
    SEFRBoostRegressor,
    SEFRGradientBoostingClassifier,
    SEFRGradientBoostingRegressor,
    auto_boosting_config,
)


def __getattr__(name: str) -> Any:
    try:
        return getattr(_prism_boost, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

__all__ = [
    "AUTO_PARAM_NAMES",
    "PrismBoostClassifier",
    "PrismBoostRegressor",
    "SEFRBoostClassifier",
    "SEFRBoostRegressor",
    "SEFRGradientBoostingClassifier",
    "SEFRGradientBoostingRegressor",
    "SPLIT_MODE_OPTIONS",
    "auto_boosting_config",
]
