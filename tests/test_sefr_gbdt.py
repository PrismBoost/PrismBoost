import numpy as np
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.datasets import load_breast_cancer, make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_allclose

from prismboost.sefr_gbdt import (
    AUTO_PARAM_NAMES,
    SEFRBoostClassifier,
    SEFRBoostRegressor,
    SEFRGradientBoostingClassifier,
    SEFRGradientBoostingRegressor,
    _mse_leaf_value,
    _newton_leaf_value,
    auto_boosting_config,
)


def test_public_aliases_are_same_classes():
    from prismboost import PrismBoostClassifier, PrismBoostRegressor

    assert SEFRBoostClassifier is SEFRGradientBoostingClassifier
    assert SEFRBoostRegressor is SEFRGradientBoostingRegressor
    assert PrismBoostClassifier is SEFRBoostClassifier
    assert PrismBoostRegressor is SEFRBoostRegressor


from _utils import check_estimator, get_expected_failed_tests


def test_sefr_gbdt_estimator():
    check_estimator(
        SEFRGradientBoostingClassifier(
            n_estimators=5, max_depth=2, min_samples_leaf=5, random_state=0
        ),
        expected_failed_checks=get_expected_failed_tests(
            SEFRGradientBoostingClassifier()
        ),
    )
    assert is_classifier(SEFRGradientBoostingClassifier())


def test_newton_leaf_value():
    r = np.array([0.5, -0.5, 0.25])
    p = np.array([0.5, 0.5, 0.5])
    w = np.ones(3)
    v = _newton_leaf_value(r, p, w)
    assert np.isfinite(v)


def test_mse_leaf_value():
    r = np.array([1.0, -1.0, 2.0])
    w = np.ones(3)
    v = _mse_leaf_value(r, w)
    assert_allclose(v, np.mean(r))


def test_sefr_gbdt_regressor_estimator():
    check_estimator(
        SEFRGradientBoostingRegressor(
            n_estimators=5, max_depth=2, min_samples_leaf=5, random_state=0
        ),
        expected_failed_checks=get_expected_failed_tests(
            SEFRGradientBoostingRegressor()
        ),
    )
    assert is_regressor(SEFRGradientBoostingRegressor())


def test_regression_fit_predict_smoke():
    X, y = make_regression(n_samples=200, n_features=12, noise=5.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    reg = SEFRGradientBoostingRegressor(
        n_estimators=40,
        learning_rate=0.08,
        max_depth=3,
        min_samples_leaf=8,
        random_state=0,
    )
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    assert pred.shape == y_test.shape
    r2 = reg.score(X_test, y_test)
    assert r2 > 0.3


def test_regression_subsample_and_sample_weight():
    X, y = make_regression(n_samples=100, n_features=8, random_state=1)
    w = np.ones(len(y))
    w[:20] = 0.5
    reg = SEFRGradientBoostingRegressor(
        n_estimators=15,
        subsample=0.8,
        max_depth=2,
        min_samples_leaf=5,
        random_state=2,
    )
    reg.fit(X, y, sample_weight=w)
    assert reg.predict(X).shape == (100,)


def test_regression_multioutput_raises():
    X, y = make_regression(n_samples=40, n_features=4, n_targets=2, random_state=0)
    reg = SEFRGradientBoostingRegressor(n_estimators=3, random_state=0)
    with pytest.raises(ValueError, match="single-target|1d array"):
        reg.fit(X, y)


def test_fit_predict_smoke():
    X, y = make_classification(
        n_samples=200, n_features=10, random_state=42, flip_y=0.05
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    clf = SEFRGradientBoostingClassifier(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=10,
        random_state=0,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    assert pred.shape == y_test.shape
    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    score = clf.score(X_test, y_test)
    assert 0.5 <= score <= 1.0


def test_breast_cancer_with_scaler():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SEFRGradientBoostingClassifier(
                    n_estimators=50,
                    learning_rate=0.08,
                    max_depth=4,
                    min_samples_leaf=10,
                    random_state=2,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    assert pipe.score(X_test, y_test) > 0.85


def test_n_estimators_one():
    X, y = make_classification(n_samples=80, n_features=5, random_state=3)
    clf = SEFRGradientBoostingClassifier(
        n_estimators=1, max_depth=2, min_samples_leaf=5, random_state=0, use_cpp=False
    )
    clf.fit(X, y)
    assert len(clf.trees_) == 1


def test_subsample():
    X, y = make_classification(n_samples=150, n_features=8, random_state=4)
    clf = SEFRGradientBoostingClassifier(
        n_estimators=15,
        subsample=0.7,
        max_depth=2,
        min_samples_leaf=5,
        random_state=5,
    )
    clf.fit(X, y)
    assert clf.predict(X).shape == (150,)


def test_multiclass_fit_predict():
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_informative=8,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=0,
    )
    clf = SEFRGradientBoostingClassifier(
        n_estimators=20, max_depth=2, min_samples_leaf=5, random_state=0, use_cpp=False
    )
    clf.fit(X, y)
    assert clf.n_classes_ == 4
    # decision_function is (n, K) for multiclass; binary stays 1-D.
    df = clf.decision_function(X)
    assert df.shape == (X.shape[0], 4)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 4)
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = clf.predict(X)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))
    assert clf.trees_ and len(clf.trees_[0]) == 4


def test_binary_decision_function_is_1d():
    X, y = make_classification(
        n_samples=120, n_features=6, n_classes=2, random_state=0
    )
    clf = SEFRGradientBoostingClassifier(
        n_estimators=10, max_depth=2, min_samples_leaf=5, random_state=0
    )
    clf.fit(X, y)
    assert clf.n_classes_ == 2
    assert clf.decision_function(X).ndim == 1


def test_multiclass_ignores_scale_pos_weight():
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_informative=8,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=1,
    )
    a = SEFRGradientBoostingClassifier(
        n_estimators=15, max_depth=2, min_samples_leaf=5, random_state=2
    ).fit(X, y)
    b = SEFRGradientBoostingClassifier(
        n_estimators=15,
        max_depth=2,
        min_samples_leaf=5,
        scale_pos_weight=5.0,
        random_state=2,
    ).fit(X, y)
    # scale_pos_weight has no meaning for multiclass; predictions must be identical.
    assert_allclose(a.predict_proba(X), b.predict_proba(X))


def test_not_fitted():
    clf = SEFRGradientBoostingClassifier()
    X = np.zeros((3, 2))
    with pytest.raises(NotFittedError):
        clf.predict(X)


def test_classes_order_preserved():
    X, y = make_classification(n_samples=100, n_features=4, random_state=6)
    y_flipped = np.where(y == 0, 1, 0)
    clf = SEFRGradientBoostingClassifier(
        n_estimators=10, max_depth=2, min_samples_leaf=8, random_state=7
    )
    clf.fit(X, y_flipped)
    proba = clf.predict_proba(X[:5])
    assert proba.shape[1] == 2
    assert np.all((proba >= 0) & (proba <= 1))


def test_sample_weight():
    X, y = make_classification(n_samples=120, n_features=6, random_state=8)
    w = np.ones(len(y))
    w[y == 1] = 2.0
    clf = SEFRGradientBoostingClassifier(
        n_estimators=20, max_depth=2, min_samples_leaf=8, random_state=9
    )
    clf.fit(X, y, sample_weight=w)
    assert clf.score(X, y) >= 0.5


def test_class_weight_balanced():
    X, y = make_classification(
        n_samples=400, n_features=8, weights=[0.92, 0.08], random_state=10
    )
    clf = SEFRGradientBoostingClassifier(
        n_estimators=25,
        max_depth=3,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=11,
    )
    clf.fit(X, y)
    assert clf.score(X, y) >= 0.5


def _slow_tree_predict_row(tree, x: np.ndarray) -> float:
    node = tree.root_
    while not node.is_leaf:
        s = float(np.dot(x, node.coef) + node.intercept)
        node = node.left if s <= 0 else node.right
    return node.value


def test_vectorized_tree_predict_matches_rowwise():
    X, y = make_classification(n_samples=150, n_features=8, random_state=20)
    # Fit a single tree via booster internals would be heavy; build tree through classifier
    clf = SEFRGradientBoostingClassifier(
        n_estimators=1, max_depth=4, min_samples_leaf=5, random_state=21, use_cpp=False
    )
    clf.fit(X, y)
    tree = clf.trees_[0]
    vec = tree.predict(X)
    slow = np.array([_slow_tree_predict_row(tree, X[i]) for i in range(len(X))])
    assert_allclose(vec, slow, rtol=1e-15, atol=1e-14)


def test_high_dimensional_onehot_sparse_sefr_coef():
    """Tree splits must densify SEFR coef_ when stored as CSR (common after OHE)."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    rng = np.random.default_rng(0)
    n = 180
    data = {"num": rng.normal(size=n)}
    for i in range(18):
        data[f"cat{i}"] = rng.choice([f"a{j}" for j in range(12)], size=n)
    df = pd.DataFrame(data)
    y = rng.integers(0, 2, size=n)

    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), ["num"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [c for c in df.columns if c.startswith("cat")],
            ),
        ]
    )
    X = ct.fit_transform(df)
    assert X.shape[1] > 64

    clf = SEFRGradientBoostingClassifier(
        n_estimators=8,
        max_depth=4,
        min_samples_leaf=5,
        min_samples_split=4,
        random_state=0,
    )
    clf.fit(X, y)
    assert clf.predict(X).shape == (n,)
    assert clf.predict_proba(X).shape == (n, 2)


def test_auto_config_scales_with_sample_size():
    small = auto_boosting_config(200, 10)
    large = auto_boosting_config(20000, 10)
    assert small["max_depth"] < large["max_depth"]
    assert small["learning_rate"] < large["learning_rate"]
    assert small["min_samples_leaf"] < large["min_samples_leaf"]
    assert set(small) == set(AUTO_PARAM_NAMES)


def test_auto_config_samples_axis_candidates_when_wide():
    assert auto_boosting_config(1000, 20)["split_mode"] == "hybrid"
    assert auto_boosting_config(1000, 500)["split_mode"] == "hybrid_sampled"


@pytest.mark.parametrize("use_cpp", [False, True])
def test_defaults_are_resolved_from_data(use_cpp):
    X, y = make_classification(n_samples=600, n_features=8, random_state=0)
    clf = SEFRGradientBoostingClassifier(random_state=0, use_cpp=use_cpp).fit(X, y)
    expected = auto_boosting_config(600, 8)
    assert clf.auto_config_ == expected
    for name, value in expected.items():
        assert getattr(clf, f"{name}_") == value
        # Constructor params stay untouched so clone/get_params keep "auto".
        assert getattr(clf, name) == "auto"


def test_explicit_params_are_not_overridden():
    X, y = make_classification(n_samples=600, n_features=8, random_state=0)
    clf = SEFRGradientBoostingClassifier(
        n_estimators=7, max_depth=2, random_state=0, use_cpp=False
    ).fit(X, y)
    assert clf.n_estimators_ == 7
    assert clf.max_depth_ == 2
    assert len(clf.trees_) == 7
    assert "n_estimators" not in clf.auto_config_
    assert "max_depth" not in clf.auto_config_
    assert clf.auto_config_["subsample"] == auto_boosting_config(600, 8)["subsample"]


def test_untuned_model_is_accurate_out_of_the_box():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    clf = SEFRGradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    assert clf.score(X_test, y_test) > 0.9


def test_auto_regressor_fits_and_records_config():
    X, y = make_regression(n_samples=400, n_features=6, random_state=0)
    reg = SEFRGradientBoostingRegressor(random_state=0).fit(X, y)
    assert reg.auto_config_ == auto_boosting_config(400, 6)
    assert reg.score(X, y) > 0.5


def test_pickle_roundtrip_keeps_resolved_params():
    import pickle

    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    clf = SEFRGradientBoostingClassifier(random_state=0).fit(X, y)
    loaded = pickle.loads(pickle.dumps(clf))
    assert loaded.auto_config_ == clf.auto_config_
    assert_allclose(loaded.predict_proba(X), clf.predict_proba(X))


def test_scale_pos_weight_and_weighted_init():
    X, y = make_classification(n_samples=200, n_features=6, random_state=12)
    y_idx = y.astype(np.float64)
    sw = np.ones(len(y))
    ew_pos = 3.0
    sw = np.where(y == 1, sw * ew_pos, sw)
    pos_rate = np.dot(sw, y_idx) / sw.sum()
    expected_init = np.log(pos_rate / (1.0 - pos_rate))
    clf = SEFRGradientBoostingClassifier(
        n_estimators=5, max_depth=2, min_samples_leaf=10, random_state=13
    )
    clf.fit(X, y, sample_weight=np.ones(len(y)))
    assert abs(clf.init_score_ - expected_init) > 1e-3
    clf2 = SEFRGradientBoostingClassifier(
        n_estimators=5,
        max_depth=2,
        min_samples_leaf=10,
        scale_pos_weight=ew_pos,
        random_state=13,
    )
    clf2.fit(X, y, sample_weight=np.ones(len(y)))
    assert_allclose(clf2.init_score_, expected_init, rtol=1e-5)
