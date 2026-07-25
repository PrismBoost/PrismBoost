#!/usr/bin/env python3
"""Minimal PrismBoost classification example."""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from prismboost import PrismBoostClassifier


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("scale", MinMaxScaler()),
            (
                "model",
                PrismBoostClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    print(f"accuracy={accuracy_score(y_test, pred):.4f}")
    print(f"roc_auc={roc_auc_score(y_test, proba):.4f}")


if __name__ == "__main__":
    main()
