#!/usr/bin/env python3
"""Tune PrismBoostClassifier with Optuna (CV on the training split).

Install extras first::

    pip install "prismboost[examples]"
    # or: pip install optuna

Run::

    python examples/optuna_tuning.py
    python examples/optuna_tuning.py --n-trials 20
"""

from __future__ import annotations

import argparse

import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from prismboost import PrismBoostClassifier, SPLIT_MODE_OPTIONS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_scaler(name: str, random_state: int):
    if name == "minmax":
        return MinMaxScaler()
    if name == "standard":
        return StandardScaler()
    if name == "quantile-normal":
        return QuantileTransformer(
            output_distribution="normal",
            n_quantiles=100,
            random_state=random_state,
        )
    raise ValueError(f"Unknown scaler: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials (default: 30)")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for the objective")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed
    )
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    def objective(trial: optuna.Trial) -> float:
        scaler_name = trial.suggest_categorical(
            "scaler", ["minmax", "standard", "quantile-normal"]
        )
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "split_mode": trial.suggest_categorical("split_mode", list(SPLIT_MODE_OPTIONS)),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state": args.seed,
        }
        pipe = Pipeline(
            [
                ("scale", _make_scaler(scaler_name, args.seed)),
                ("clf", PrismBoostClassifier(**params)),
            ]
        )
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    scaler_name = best.pop("scaler")
    best_model = Pipeline(
        [
            ("scale", _make_scaler(scaler_name, args.seed)),
            ("clf", PrismBoostClassifier(**best, random_state=args.seed)),
        ]
    )
    best_model.fit(X_train, y_train)
    proba = best_model.predict_proba(X_test)[:, 1]
    pred = best_model.predict(X_test)

    print(f"Best CV ROC-AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Hold-out accuracy: {best_model.score(X_test, y_test):.4f}")
    print(f"Hold-out F1 (weighted): {f1_score(y_test, pred, average='weighted'):.4f}")
    print(f"Hold-out ROC-AUC: {roc_auc_score(y_test, proba):.4f}")


if __name__ == "__main__":
    main()
