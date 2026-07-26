"""PrismBoost: gradient boosting with SEFR oblique splits."""

__version__ = "0.1.1"

from .sefr import SEFR
from .sefr_gbdt import (
    PrismBoostClassifier,
    PrismBoostRegressor,
    SEFRBoostClassifier,
    SEFRBoostRegressor,
    SEFRGradientBoostingClassifier,
    SEFRGradientBoostingRegressor,
    SPLIT_MODE_OPTIONS,
)

__all__ = [
    "PrismBoostClassifier",
    "PrismBoostRegressor",
    "SEFR",
    "SEFRBoostClassifier",
    "SEFRBoostRegressor",
    "SEFRGradientBoostingClassifier",
    "SEFRGradientBoostingRegressor",
    "SPLIT_MODE_OPTIONS",
]
