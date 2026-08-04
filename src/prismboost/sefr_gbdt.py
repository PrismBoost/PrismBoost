"""Legacy import path for PrismBoost, kept for backwards compatibility.

The implementation moved to :mod:`prismboost.prism_boost` when the project was
renamed. This module re-exports it, including the old ``SEFRGradientBoosting*``
class names, so existing code and pickles written as
``prismboost.sefr_gbdt.SEFRGradientBoostingClassifier`` keep working. New code
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
    # Pickles written before the rename reference internals such as
    # ``prismboost.sefr_gbdt._SEFRTree`` by module path, so forward every
    # remaining lookup to the implementation module.
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
