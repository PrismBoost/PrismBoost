"""PrismBoost / SEFRBoost: gradient boosting with SEFR oblique splits.

Re-exports the implementation in :mod:`prismboost.sefr_gbdt`, including the
optional C++ backend (``prismboost._sefr_boost_core``).
"""

from .sefr_gbdt import (
    AUTO_PARAM_NAMES,
    SPLIT_MODE_OPTIONS,
    PrismBoostClassifier,
    PrismBoostRegressor,
    SEFRBoostClassifier,
    SEFRBoostRegressor,
    auto_boosting_config,
)

__all__ = [
    "AUTO_PARAM_NAMES",
    "PrismBoostClassifier",
    "PrismBoostRegressor",
    "SEFRBoostClassifier",
    "SEFRBoostRegressor",
    "SPLIT_MODE_OPTIONS",
    "auto_boosting_config",
]
