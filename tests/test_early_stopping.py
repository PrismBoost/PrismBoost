"""Tests for ``eval_set`` and ``early_stopping_rounds``."""

import pickle

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import log_loss, mean_squared_error

from prismboost import PrismBoostClassifier, PrismBoostRegressor
from prismboost._cpp_backend import CPP_AVAILABLE

BACKENDS = [
    pytest.param(True, id="cpp", marks=pytest.mark.skipif(not CPP_AVAILABLE, reason="no C++")),
    pytest.param(False, id="python"),
]
TASKS = ["binary", "multiclass", "regression"]


def _split(task):
    if task == "regression":
        X, y = make_regression(n_samples=500, n_features=8, noise=20.0, random_state=0)
    else:
        X, y = make_classification(
            n_samples=500,
            n_features=8,
            n_informative=5,
            n_classes=2 if task == "binary" else 3,
            flip_y=0.15,
            random_state=0,
        )
    return X[:350], y[:350], X[350:], y[350:]


def _model(task, **params):
    params = {"n_estimators": 300, "learning_rate": 0.3, "random_state": 1, **params}
    if task == "regression":
        return PrismBoostRegressor(**params)
    return PrismBoostClassifier(**params)


def _output(model, X):
    if isinstance(model, PrismBoostRegressor):
        return model.predict(X)
    return model.predict_proba(X)


def _loss(model, X, y):
    if isinstance(model, PrismBoostRegressor):
        return mean_squared_error(y, model.predict(X))
    return log_loss(y, model.predict_proba(X), labels=model.classes_)


@pytest.mark.parametrize("use_cpp", BACKENDS)
@pytest.mark.parametrize("task", TASKS)
def test_early_stopping_truncates_at_best_stage(task, use_cpp):
    X, y, X_val, y_val = _split(task)
    model = _model(task, early_stopping_rounds=10, use_cpp=use_cpp)
    model.fit(X, y, eval_set=(X_val, y_val))

    losses = model.validation_loss_
    assert len(losses) < model.n_estimators_
    assert model.best_iteration_ == int(np.argmin(losses)) + 1
    assert len(losses) == model.best_iteration_ + 10
    assert _loss(model, X_val, y_val) == pytest.approx(losses[model.best_iteration_ - 1])


@pytest.mark.parametrize("use_cpp", BACKENDS)
@pytest.mark.parametrize("task", TASKS)
def test_refit_with_best_iteration_reproduces_model(task, use_cpp):
    X, y, X_val, y_val = _split(task)
    stopped = _model(task, early_stopping_rounds=10, use_cpp=use_cpp)
    stopped.fit(X, y, eval_set=(X_val, y_val))

    refit = _model(task, n_estimators=stopped.best_iteration_, use_cpp=use_cpp).fit(X, y)

    np.testing.assert_array_equal(_output(stopped, X_val), _output(refit, X_val))


@pytest.mark.parametrize("use_cpp", BACKENDS)
@pytest.mark.parametrize("task", TASKS)
def test_eval_set_without_early_stopping_only_records(task, use_cpp):
    X, y, X_val, y_val = _split(task)
    model = _model(task, n_estimators=40, use_cpp=use_cpp)
    model.fit(X, y, eval_set=(X_val, y_val))
    plain = _model(task, n_estimators=40, use_cpp=use_cpp).fit(X, y)

    assert len(model.validation_loss_) == 40
    assert not hasattr(model, "best_iteration_")
    assert model.validation_loss_[-1] == pytest.approx(_loss(model, X_val, y_val))
    np.testing.assert_array_equal(_output(model, X_val), _output(plain, X_val))


@pytest.mark.parametrize("task", TASKS)
def test_early_stopping_requires_eval_set(task):
    X, y, _, _ = _split(task)
    with pytest.raises(ValueError, match="requires eval_set"):
        _model(task, early_stopping_rounds=5).fit(X, y)


@pytest.mark.parametrize("task", TASKS)
def test_eval_set_is_validated(task):
    X, y, X_val, y_val = _split(task)
    model = _model(task, early_stopping_rounds=5)
    with pytest.raises(ValueError, match="tuple"):
        model.fit(X, y, eval_set=X_val)
    with pytest.raises(ValueError, match="features"):
        model.fit(X, y, eval_set=(X_val[:, :-1], y_val))
    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(X, y, eval_set=(X_val, y_val[:-1]))


@pytest.mark.parametrize("use_cpp", BACKENDS)
def test_unseen_validation_class_is_left_out_of_loss(use_cpp):
    X, y, X_val, y_val = _split("multiclass")
    # Rows labeled 3 belong to a class the training split never saw.
    y_val_unseen = y_val.copy()
    y_val_unseen[:20] = 3

    params = {"n_estimators": 30, "use_cpp": use_cpp}
    with_unseen = _model("multiclass", **params).fit(X, y, eval_set=(X_val, y_val_unseen))
    without = _model("multiclass", **params).fit(X, y, eval_set=(X_val[20:], y_val[20:]))

    np.testing.assert_allclose(with_unseen.validation_loss_, without.validation_loss_)

    with pytest.raises(ValueError, match="no rows with a class seen in training"):
        _model("multiclass", **params).fit(X, y, eval_set=(X_val, np.full_like(y_val, 7)))


@pytest.mark.parametrize("use_cpp", BACKENDS)
def test_string_labels(use_cpp):
    X, y, X_val, y_val = _split("binary")
    names = np.array(["no", "yes"])
    model = _model("binary", early_stopping_rounds=10, use_cpp=use_cpp)
    model.fit(X, names[y], eval_set=(X_val, names[y_val]))

    assert set(model.predict(X_val)) <= set(names)
    assert model.best_iteration_ >= 1


@pytest.mark.parametrize("use_cpp", BACKENDS)
@pytest.mark.parametrize("task", TASKS)
def test_pickle_keeps_early_stopped_model(task, use_cpp):
    X, y, X_val, y_val = _split(task)
    model = _model(task, early_stopping_rounds=10, use_cpp=use_cpp)
    model.fit(X, y, eval_set=(X_val, y_val))

    restored = pickle.loads(pickle.dumps(model))

    assert restored.best_iteration_ == model.best_iteration_
    np.testing.assert_array_equal(restored.validation_loss_, model.validation_loss_)
    np.testing.assert_array_equal(_output(restored, X_val), _output(model, X_val))


@pytest.mark.parametrize("task", TASKS)
def test_refit_without_eval_set_clears_validation_results(task):
    X, y, X_val, y_val = _split(task)
    model = _model(task, early_stopping_rounds=10)
    model.fit(X, y, eval_set=(X_val, y_val))

    model.set_params(early_stopping_rounds=None, n_estimators=20).fit(X, y)

    assert not hasattr(model, "validation_loss_")
    assert not hasattr(model, "best_iteration_")


def test_old_pickle_without_parameter_defaults_to_none():
    X, y, _, _ = _split("binary")
    model = _model("binary", n_estimators=5).fit(X, y)
    state = model.__getstate__()
    del state["early_stopping_rounds"]

    restored = PrismBoostClassifier.__new__(PrismBoostClassifier)
    restored.__setstate__(state)

    assert restored.early_stopping_rounds is None
