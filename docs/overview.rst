What Is PrismBoost?
===================

``PrismBoostClassifier`` is gradient boosting with **SEFR oblique splits** at
each internal tree node (hyperplane splits instead of axis-aligned thresholds).

- Preferred name: ``PrismBoostClassifier`` / ``PrismBoostRegressor``
- Compatibility aliases: ``SEFRBoostClassifier`` / ``SEFRBoostRegressor``
- Long names: ``SEFRGradientBoostingClassifier`` / ``SEFRGradientBoostingRegressor``
- Module: ``prismboost.sefr_gbdt``

At a high level:

1. It builds an additive boosting model over multiple trees.
2. Each internal split is oblique (linear) and uses SEFR-derived hyperplanes.
3. It supports binary and multiclass classification, plus regression, with
   ``predict`` / ``predict_proba`` where applicable.

When to use it
--------------

- You want boosted performance with oblique (non-axis-aligned) split geometry.
- You need a sklearn-compatible estimator for pipelines and cross-validation.
- You want an optional C++ backend for faster fit/predict.
