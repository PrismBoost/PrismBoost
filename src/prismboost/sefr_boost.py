"""PrismBoost / SEFRBoost: gradient boosting with SEFR oblique splits.

Re-exports the implementation in :mod:`prismboost.sefr_gbdt`, including the
optional C++ backend (``prismboost._sefr_boost_core``).
"""

from .sefr_gbdt import (
    SPLIT_MODE_OPTIONS,
    PrismBoostClassifier,
    PrismBoostRegressor,
    SEFRBoostClassifier,
    SEFRBoostRegressor,
)

__all__ = [
    "PrismBoostClassifier",
    "PrismBoostRegressor",
    "SEFRBoostClassifier",
    "SEFRBoostRegressor",
    "SPLIT_MODE_OPTIONS",
]
