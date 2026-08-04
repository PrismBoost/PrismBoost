"""PrismBoost: gradient boosting with SEFR oblique splits.

The estimators are :class:`PrismBoostClassifier` and
:class:`PrismBoostRegressor`. The pre-rename names ``SEFRBoostClassifier`` and
``SEFRGradientBoostingClassifier`` (and their regressor counterparts) remain
available as aliases of the same classes.
"""

__version__ = "0.2.0"

from .sefr import SEFR
from .prism_boost import (
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
