"""PrismBoost: gradient boosting with SEFR oblique splits."""

__version__ = "0.2.0"

from .sefr import SEFR
from .sefr_gbdt import (
    AUTO_PARAM_NAMES,
    PrismBoostClassifier,
    PrismBoostRegressor,
    SEFRBoostClassifier,
    SEFRBoostRegressor,
    SEFRGradientBoostingClassifier,
    SEFRGradientBoostingRegressor,
    SPLIT_MODE_OPTIONS,
    auto_boosting_config,
)

__all__ = [
    "AUTO_PARAM_NAMES",
    "PrismBoostClassifier",
    "PrismBoostRegressor",
    "SEFR",
    "SEFRBoostClassifier",
    "SEFRBoostRegressor",
    "SEFRGradientBoostingClassifier",
    "SEFRGradientBoostingRegressor",
    "SPLIT_MODE_OPTIONS",
    "auto_boosting_config",
]
