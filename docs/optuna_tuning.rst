Optuna Tuning Example
=====================

PrismBoost is sklearn-compatible, so it plugs into Optuna the same way as other
estimators. A runnable script lives at ``examples/optuna_tuning.py``.

Install Optuna
--------------

.. code-block:: bash

   pip install "prismboost[examples]"
   # or: pip install optuna

Run the example
---------------

.. code-block:: bash

   python examples/optuna_tuning.py
   python examples/optuna_tuning.py --n-trials 20

What it does
------------

1. Holds out a stratified test set.
2. Tunes ``PrismBoostClassifier`` (+ feature scaler / ``split_mode``) with Optuna
   using stratified CV ROC-AUC on the training split.
3. Refits the best pipeline on the full training set and reports hold-out metrics.

Compact inline version
----------------------

.. code-block:: python

   import optuna
   import numpy as np
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import MinMaxScaler
   from prismboost import PrismBoostClassifier

   X, y = load_breast_cancer(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, stratify=y, random_state=42
   )
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

   def objective(trial):
       params = {
           "n_estimators": trial.suggest_int("n_estimators", 20, 200),
           "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
           "max_depth": trial.suggest_int("max_depth", 2, 6),
           "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
           "min_samples_split": trial.suggest_int("min_samples_split", 10, 80),
           "subsample": trial.suggest_float("subsample", 0.6, 1.0),
           "split_mode": trial.suggest_categorical(
               "split_mode", ["sefr_only", "axis_fallback", "hybrid_sampled", "hybrid"]
           ),
           "random_state": 42,
       }
       pipe = Pipeline([
           ("scale", MinMaxScaler()),
           ("clf", PrismBoostClassifier(**params)),
       ])
       return float(np.mean(cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")))

   study = optuna.create_study(direction="maximize")
   study.optimize(objective, n_trials=30)
   print("Best CV ROC-AUC:", study.best_value)
   print("Best params:", study.best_params)

Notes
-----

- For laptops, start with ``--n-trials 20`` to ``30``.
- Keep a fixed ``random_state`` for reproducible comparisons.
- Always reserve a held-out test set and score it only after tuning.
- ``MinMaxScaler`` is a good default for SEFR-style splits (non-negative features);
  the full example also searches ``standard`` and ``quantile-normal``.
