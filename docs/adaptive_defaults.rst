Adaptive Defaults
=================

Every capacity parameter of ``PrismBoostClassifier`` and ``PrismBoostRegressor``
defaults to ``"auto"`` and is resolved from the training-set shape inside
``fit``, the way CatBoost adapts its learning rate to dataset size. The rules
were calibrated on the per-dataset Optuna optima of the 121 PMLB benchmark
datasets, so an untuned model starts near a reasonable configuration instead of
a single fixed point that under-fits large data and over-fits small data.

The rules
---------

============================ =================================================
Parameter                    ``"auto"`` rule
============================ =================================================
``n_estimators``             200
``learning_rate``            0.05 below 500 rows, else 0.1
``max_depth``                4 below 500 rows, else 6
``min_samples_leaf``         ``sqrt(n_samples) / 2`` clipped to ``[5, 30]``
``min_samples_split``        ``2 * min_samples_leaf``
``subsample``                0.8, or 1.0 below 100 rows
``split_mode``               ``hybrid`` up to 50 features, else
                             ``hybrid_sampled``
============================ =================================================

``learning_rate`` and ``n_estimators`` are chosen together: their product (the
shrinkage budget) lands near 10 on small data and near 20 from ~500 rows upward,
which is where the tuned optima sit.

Only the shape of ``X`` is used, never ``y``, so equally sized cross-validation
folds resolve to the same configuration and no label information leaks into the
choice.

Inspecting what was chosen
--------------------------

.. code-block:: python

   from prismboost import PrismBoostClassifier, auto_boosting_config
   from sklearn.datasets import load_breast_cancer

   X, y = load_breast_cancer(return_X_y=True)
   clf = PrismBoostClassifier(random_state=0).fit(X, y)

   clf.auto_config_     # {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6, ...}
   clf.max_depth_       # 6  (value actually used)
   clf.max_depth        # 'auto'  (constructor parameter, so clone/get_params stay stable)

   auto_boosting_config(n_samples=1000, n_features=20)   # the rules, without fitting

Mixing explicit and adaptive values
-----------------------------------

Passing a value disables adaptation for that parameter only:

.. code-block:: python

   clf = PrismBoostClassifier(max_depth=3, random_state=0).fit(X, y)
   clf.max_depth_, clf.n_estimators_    # (3, 200)
   "max_depth" in clf.auto_config_      # False

Notes
-----

- Class imbalance is left alone (``class_weight=None``), matching the XGBoost and
  CatBoost defaults. Set ``class_weight="balanced"`` when you want reweighting.
- Hyperparameter search is unaffected: an Optuna or ``GridSearchCV`` space that
  sets these parameters explicitly overrides the adaptive values entirely.
- To restore the pre-0.2 defaults, pass them explicitly::

      PrismBoostClassifier(
          n_estimators=100,
          learning_rate=0.1,
          max_depth=3,
          min_samples_leaf=10,
          min_samples_split=2,
          subsample=1.0,
          split_mode="hybrid_sampled",
      )
