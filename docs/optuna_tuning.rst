Optuna Tuning Example
=====================

This example shows a compact Optuna workflow for ``PrismBoostClassifier``.

.. code-block:: python

   import optuna
   import numpy as np
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import StratifiedKFold, train_test_split
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.metrics import f1_score
   from prismboost import PrismBoostClassifier

   X, y = load_breast_cancer(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, stratify=y, random_state=42
   )
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

   def objective(trial):
       params = {
           "n_estimators": trial.suggest_int("n_estimators", 20, 150),
           "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
           "max_depth": trial.suggest_int("max_depth", 2, 6),
           "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
           "min_samples_split": trial.suggest_int("min_samples_split", 10, 80),
           "subsample": trial.suggest_float("subsample", 0.6, 1.0),
           "random_state": 42,
       }
       pipe = Pipeline([
           ("scale", StandardScaler()),
           ("clf", PrismBoostClassifier(**params)),
       ])
       scores = []
       for tr_idx, va_idx in cv.split(X_train, y_train):
           pipe.fit(X_train[tr_idx], y_train[tr_idx])
           pred = pipe.predict(X_train[va_idx])
           scores.append(f1_score(y_train[va_idx], pred, average="weighted"))
       return float(np.mean(scores))

   study = optuna.create_study(direction="maximize")
   study.optimize(objective, n_trials=30)
   print("Best CV score:", study.best_value)
   print("Best params:", study.best_params)

Notes
-----

- For laptops, start with ``n_trials=20`` to ``30``.
- Keep a fixed ``random_state`` for reproducible comparisons.
- Use a held-out test set for final reporting after tuning.

