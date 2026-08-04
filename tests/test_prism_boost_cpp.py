"""Tests for the optional C++ PrismBoost backend."""

import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_allclose

from prismboost._cpp_backend import CPP_AVAILABLE, cpp_backend_status
from prismboost._cpp_pickle import serialized_size_bytes
from prismboost import PrismBoostClassifier, PrismBoostRegressor


@pytest.mark.skipif(not CPP_AVAILABLE, reason=cpp_backend_status())
def test_cpp_classifier_pickle_roundtrip():
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float64)
    clf = PrismBoostClassifier(
        n_estimators=15, max_depth=2, min_samples_leaf=5, random_state=0, use_cpp=True
    )
    clf.fit(X, y)
    size = serialized_size_bytes(clf)
    assert size > 0
    restored = pickle.loads(pickle.dumps(clf))
    assert_allclose(clf.predict_proba(X), restored.predict_proba(X), rtol=1e-12)


@pytest.mark.skipif(not CPP_AVAILABLE, reason=cpp_backend_status())
def test_cpp_pipeline_pickle_size_nonzero():
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float64)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", PrismBoostClassifier(n_estimators=10, max_depth=2, use_cpp=True, random_state=0)),
    ])
    pipe.fit(X, y)
    assert len(pickle.dumps(pipe, protocol=pickle.HIGHEST_PROTOCOL)) > 0


@pytest.mark.skipif(not CPP_AVAILABLE, reason=cpp_backend_status())
def test_cpp_classifier_save_load(tmp_path: Path):
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float64)
    clf = PrismBoostClassifier(n_estimators=12, max_depth=2, use_cpp=True, random_state=1)
    clf.fit(X, y)
    path = tmp_path / "sefrboost.pkl"
    clf.save(path)
    assert path.stat().st_size > 0
    loaded = PrismBoostClassifier.load(path)
    assert_allclose(clf.decision_function(X), loaded.decision_function(X), rtol=1e-12)


@pytest.mark.skipif(not CPP_AVAILABLE, reason=cpp_backend_status())
def test_cpp_regressor_save_load(tmp_path: Path):
    X, y = make_regression(n_samples=120, n_features=8, random_state=3)
    reg = PrismBoostRegressor(n_estimators=10, max_depth=2, use_cpp=True, random_state=3)
    reg.fit(X, y)
    path = tmp_path / "sefrboost_reg.pkl"
    reg.save(path)
    loaded = PrismBoostRegressor.load(path)
    assert_allclose(reg.predict(X), loaded.predict(X), rtol=1e-12)


@pytest.mark.skipif(not CPP_AVAILABLE, reason=cpp_backend_status())
def test_cpp_classifier_matches_python_close():
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    params = dict(
        n_estimators=25,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=5,
        split_mode="sefr_only",
        random_state=0,
    )
    py_clf = PrismBoostClassifier(**params, use_cpp=False)
    cpp_clf = PrismBoostClassifier(**params, use_cpp=True)
    py_clf.fit(X_train, y_train)
    cpp_clf.fit(X_train, y_train)
    py_acc = py_clf.score(X_test, y_test)
    cpp_acc = cpp_clf.score(X_test, y_test)
    assert abs(py_acc - cpp_acc) < 0.05


@pytest.mark.skipif(not CPP_AVAILABLE, reason=cpp_backend_status())
def test_cpp_classifier_multiclass_smoke():
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_informative=8,
        n_classes=4,
        random_state=1,
    )
    clf = PrismBoostClassifier(
        n_estimators=20,
        max_depth=2,
        min_samples_leaf=5,
        split_mode="hybrid_sampled",
        random_state=1,
        use_cpp=True,
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (300,)


@pytest.mark.skipif(not CPP_AVAILABLE, reason=cpp_backend_status())
def test_cpp_regressor_smoke():
    X, y = make_regression(n_samples=200, n_features=10, noise=1.0, random_state=2)
    reg = PrismBoostRegressor(
        n_estimators=30,
        max_depth=2,
        min_samples_leaf=5,
        random_state=2,
        use_cpp=True,
    )
    reg.fit(X, y)
    pred = reg.predict(X)
    assert pred.shape == (200,)
    assert reg.score(X, y) > 0.5
