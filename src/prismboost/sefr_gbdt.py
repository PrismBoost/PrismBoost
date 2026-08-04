"""Gradient boosting with oblique splits from SEFR at each internal node."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from ._utils import SKLEARN_V1_6_OR_LATER, _fit_context, check_X_y, validate_data
from ._cpp_backend import CPP_AVAILABLE, SEFRBoostClassifierCore, SEFRBoostRegressorCore
from ._cpp_pickle import (
    cpp_estimator_getstate,
    cpp_estimator_setstate,
    load_estimator,
    save_estimator,
)
from .sefr import SEFR

__all__ = [
    "SEFRGradientBoostingClassifier",
    "SEFRGradientBoostingRegressor",
    "SEFRBoostClassifier",
    "SEFRBoostRegressor",
    "PrismBoostClassifier",
    "PrismBoostRegressor",
    "SPLIT_MODE_OPTIONS",
    "AUTO_PARAM_NAMES",
    "auto_boosting_config",
]

SPLIT_MODE_OPTIONS = ("sefr_only", "axis_fallback", "hybrid_sampled", "hybrid")

#: Parameters that accept ``"auto"`` and are then derived from the training data.
AUTO_PARAM_NAMES = (
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "subsample",
    "split_mode",
)


def auto_boosting_config(n_samples: int, n_features: int) -> dict:
    """Derive boosting hyperparameters from the training set shape.

    Used for every parameter left at ``"auto"``. The rules were calibrated
    against the per-dataset Optuna optima of 121 PMLB classification datasets:
    they target the median tuned setting per sample-size regime, so an untuned
    model starts near a reasonable configuration instead of a fixed one that
    under-fits large data and over-fits small data.

    Only the shape of ``X`` is used (never ``y``), so the result is identical
    across cross-validation folds of the same size and leaks no label
    information.

    Parameters
    ----------
    n_samples : int
        Number of training rows.

    n_features : int
        Number of training columns.

    Returns
    -------
    config : dict
        Values for every name in :data:`AUTO_PARAM_NAMES`.
    """
    n = max(1, int(n_samples))
    p = max(1, int(n_features))

    # Tuned optima keep the shrinkage budget (learning_rate * n_estimators)
    # near 10 on small data and near 20 from ~500 rows upward.
    small = n < 500
    learning_rate = 0.05 if small else 0.1
    min_samples_leaf = int(min(30, max(5, round(np.sqrt(n) / 2.0))))

    return {
        "n_estimators": 200,
        "learning_rate": learning_rate,
        "max_depth": 4 if small else 6,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": max(2, 2 * min_samples_leaf),
        "subsample": 1.0 if n < 100 else 0.8,
        # Scanning every axis-aligned candidate costs O(n_features) per node;
        # sample them once the frame is wide.
        "split_mode": "hybrid" if p <= 50 else "hybrid_sampled",
    }


def _is_auto(value) -> bool:
    return isinstance(value, str) and value == "auto"


def _resolve_boosting_params(estimator, n_samples: int, n_features: int, names) -> dict:
    """Resolve ``"auto"`` parameters and record them as fitted attributes."""
    auto = auto_boosting_config(n_samples, n_features)
    resolved = {}
    auto_used = {}
    for name in names:
        value = getattr(estimator, name)
        if _is_auto(value):
            value = auto[name]
            auto_used[name] = value
        resolved[name] = value
        setattr(estimator, f"{name}_", value)
    estimator.auto_config_ = auto_used
    return resolved


def _backfill_resolved_params(estimator) -> None:
    """Populate ``*_`` attributes for estimators pickled before auto defaults existed."""
    if hasattr(estimator, "n_estimators_"):
        return
    values = {}
    for name in AUTO_PARAM_NAMES:
        value = getattr(estimator, name, None)
        if value is None or _is_auto(value):
            return
        values[f"{name}_"] = value
    for attr, value in values.items():
        setattr(estimator, attr, value)
    estimator.auto_config_ = {}


def _cpp_random_seed(random_state) -> int:
    rng = check_random_state(random_state)
    return int(rng.randint(0, np.iinfo(np.uint32).max))


def _should_use_cpp(use_cpp: bool | None) -> bool:
    if use_cpp is False:
        return False
    return CPP_AVAILABLE


def _per_sample_class_weight(
    y_original: np.ndarray, classes: np.ndarray, class_weight
) -> np.ndarray:
    """Per-sample multipliers from sklearn-style class_weight (None / balanced / dict)."""
    if class_weight is None:
        return np.ones(y_original.shape[0], dtype=np.float64)
    cw = compute_class_weight(class_weight, classes=classes, y=y_original)
    w_by_class = {c: float(w) for c, w in zip(classes, cw)}
    return np.fromiter(
        (w_by_class[y] for y in y_original),
        dtype=np.float64,
        count=len(y_original),
    )


def _effective_fit_weights(
    y_idx: np.ndarray,
    y_original: np.ndarray,
    classes: np.ndarray,
    sample_weight: np.ndarray,
    class_weight,
    scale_pos_weight: Optional[float],
) -> np.ndarray:
    """Combine sample_weight, class_weight, and scale_pos_weight (positive class = classes_[1])."""
    sw = np.asarray(sample_weight, dtype=np.float64)
    cw = _per_sample_class_weight(y_original, classes, class_weight)
    ew = sw * cw
    spw = 1.0 if scale_pos_weight is None else float(scale_pos_weight)
    if spw != 1.0:
        ew = ew * np.where(y_idx == 1, spw, 1.0)
    return ew


# Bound the per-leaf Newton step in logit space (analogous to XGBoost's
# ``max_delta_step``). With saturated probabilities and extreme class_weight, the
# Hessian ``sum(w * p(1-p))`` can underflow relative to ``sum(w * r)`` and produce
# an enormous leaf value when a pure minority leaf is isolated. Typical legitimate
# values are O(1-10), so this cap leaves normal fits unchanged while preventing
# runaway updates.
_MAX_NEWTON_LEAF = 20.0


def _newton_leaf_value(
    residuals: np.ndarray,
    p: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    """Weighted log-loss Newton step: sum(w*r) / sum(w * p(1-p))."""
    h = p * (1.0 - p)
    h = np.maximum(h, 1e-10)
    num = np.sum(sample_weight * residuals)
    den = np.sum(sample_weight * h) + 1e-10
    value = num / den
    return float(np.clip(value, -_MAX_NEWTON_LEAF, _MAX_NEWTON_LEAF))


def _mse_leaf_value(residuals: np.ndarray, sample_weight: np.ndarray) -> float:
    """Weighted mean residual (Newton / negative-gradient step for squared loss)."""
    num = np.sum(sample_weight * residuals)
    den = np.sum(sample_weight) + 1e-10
    return float(num / den)


def _dense_sefr_coef(sefr: SEFR) -> np.ndarray:
    """Return a 1-D dense copy of ``SEFR.coef_`` (may be stored as CSR sparse)."""
    coef = sefr.coef_
    if hasattr(coef, "toarray"):
        return np.asarray(coef.toarray(), dtype=np.float64).ravel().copy()
    return np.asarray(coef, dtype=np.float64).ravel().copy()


def _sanitize_sefr_hyperplane(
    coef: np.ndarray, intercept: float
) -> tuple[np.ndarray, float]:
    """Stabilize linear split parameters from SEFR (avoid NaN/Inf and matmul overflow)."""
    c = np.asarray(coef, dtype=np.float64).ravel()
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    # With OHE + scaled numerics, keep coefficients in a range that keeps Xm @ c finite.
    c = np.clip(c, -1e4, 1e4)
    b = float(np.nan_to_num(intercept, nan=0.0, posinf=0.0, neginf=0.0))
    b = float(np.clip(b, -1e6, 1e6))
    return c, b


def _affine_hyperplane_scores(
    X: np.ndarray, coef: np.ndarray, intercept: float
) -> np.ndarray:
    """``X @ coef + intercept`` with finite inputs and no spurious matmul warnings."""
    Xb = np.asarray(X, dtype=np.float64, order="C")
    Xb = np.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.asarray(coef, dtype=np.float64).ravel()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        s = Xb @ c + float(intercept)
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


def _best_split_threshold(
    proj: np.ndarray,
    residuals: np.ndarray,
    sample_weight: np.ndarray,
    hessian: np.ndarray,
    min_samples_leaf: int,
) -> tuple[Optional[float], float]:
    """Best threshold ``t`` along the 1-D ``proj`` maximizing the GBDT split gain.

    Split rule is ``proj <= t`` (left) vs ``proj > t`` (right). The gain is the
    standard structure-score reduction ``G_L^2/H_L + G_R^2/H_R - G^2/H`` where
    ``G = sum(w * r)`` and ``H = sum(w * hessian)`` (``hessian`` is ones for
    squared error, ``p(1-p)`` for the logistic/multinomial Newton step). Keeps the
    SEFR oblique direction; only the cut location is optimized (instead of SEFR's
    class-balanced midpoint), which improves minority isolation on imbalanced nodes.

    Returns ``(t, gain)``; ``(None, -inf)`` when no split respects
    ``min_samples_leaf`` on both children or no positive-gain split exists.
    """
    n = proj.shape[0]
    if n < 2 * max(1, int(min_samples_leaf)):
        return None, -np.inf
    order = np.argsort(proj, kind="mergesort")
    ps = proj[order]
    gr = (sample_weight * residuals)[order]
    gh = (sample_weight * hessian)[order]

    g_total = float(gr.sum())
    h_total = float(gh.sum()) + 1e-12

    cum_g = np.cumsum(gr)[:-1]
    cum_h = np.cumsum(gh)[:-1]
    g_left = cum_g
    h_left = cum_h + 1e-12
    g_right = g_total - g_left
    h_right = h_total - h_left + 1e-12

    n_left = np.arange(1, n)
    n_right = n - n_left

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gain = g_left**2 / h_left + g_right**2 / h_right - g_total**2 / h_total

    distinct = ps[:-1] < ps[1:]
    valid = distinct & (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
    if not np.any(valid):
        return None, -np.inf

    gain = np.where(valid, gain, -np.inf)
    best = int(np.argmax(gain))
    best_gain = float(gain[best])
    if not np.isfinite(best_gain) or best_gain <= 0.0:
        return None, best_gain
    t = 0.5 * (float(ps[best]) + float(ps[best + 1]))
    return t, best_gain


def _axis_feature_subset(n_features: int, rng) -> np.ndarray:
    """Random ``sqrt(p)`` feature indices for ``hybrid_sampled`` axis search."""
    k = max(1, int(np.sqrt(n_features)))
    k = min(k, n_features)
    return rng.choice(n_features, size=k, replace=False)


def _best_axis_split(
    X: np.ndarray,
    residuals: np.ndarray,
    sample_weight: np.ndarray,
    hessian: np.ndarray,
    min_samples_leaf: int,
    *,
    feature_indices: Optional[np.ndarray] = None,
) -> tuple[Optional[int], float, float]:
    """Best single-feature (axis-aligned) split by the same GBDT gain as
    :func:`_best_split_threshold`, vectorized over all features.

    Crisp single-feature rules (e.g. thyroid TSH cutoffs) and pure feature
    interactions with no marginal signal are worst cases for SEFR's
    mean-difference oblique direction; offering an axis-aligned candidate and
    letting the gain comparison decide closes that gap without hurting datasets
    where the oblique split wins.

    Returns ``(feature_index, threshold, gain)``; ``(None, 0.0, -inf)`` when no
    valid positive-gain split exists.
    """
    feat_map: Optional[np.ndarray] = None
    if feature_indices is not None:
        feat_map = np.asarray(feature_indices, dtype=np.intp)
        if feat_map.size == 0:
            return None, 0.0, -np.inf
        X = X[:, feat_map]

    n, p = X.shape
    msl = max(1, int(min_samples_leaf))
    if n < 2 * msl:
        return None, 0.0, -np.inf

    order = np.argsort(X, axis=0, kind="mergesort")  # (n, p)
    Xs = np.take_along_axis(X, order, axis=0)
    gr = (sample_weight * residuals)[order]  # (n, p): per-column sorted grad
    gh = (sample_weight * hessian)[order]

    g_total = float(np.sum(sample_weight * residuals))
    h_total = float(np.sum(sample_weight * hessian)) + 1e-12

    g_left = np.cumsum(gr, axis=0)[:-1]  # (n-1, p)
    h_left = np.cumsum(gh, axis=0)[:-1] + 1e-12
    g_right = g_total - g_left
    h_right = h_total - h_left + 1e-12

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gain = (
            g_left**2 / h_left + g_right**2 / h_right - g_total**2 / h_total
        )

    n_left = np.arange(1, n)[:, None]
    distinct = Xs[:-1, :] < Xs[1:, :]
    valid = distinct & (n_left >= msl) & ((n - n_left) >= msl)
    if not np.any(valid):
        return None, 0.0, -np.inf

    gain = np.where(valid, gain, -np.inf)
    i, j = np.unravel_index(int(np.argmax(gain)), gain.shape)
    best_gain = float(gain[i, j])
    if not np.isfinite(best_gain) or best_gain <= 0.0:
        return None, 0.0, best_gain
    thr = 0.5 * (float(Xs[i, j]) + float(Xs[i + 1, j]))
    j_out = int(feat_map[j]) if feat_map is not None else int(j)
    return j_out, thr, best_gain


@dataclass
class _SEFRTreeNode:
    is_leaf: bool
    value: float = 0.0
    coef: Optional[np.ndarray] = None
    intercept: float = 0.0
    left: Optional["_SEFRTreeNode"] = None
    right: Optional["_SEFRTreeNode"] = None


class _SEFRTree:
    """One regression tree: internal nodes use linear SEFR splits on pseudo-residuals."""

    def __init__(self, root: _SEFRTreeNode):
        self.root_ = root

    @staticmethod
    def _grow(
        X: np.ndarray,
        residuals: np.ndarray,
        p: np.ndarray,
        sample_weight: np.ndarray,
        idx: np.ndarray,
        depth: int,
        max_depth: int,
        min_samples_leaf: int,
        min_samples_split: int,
        *,
        regression: bool,
        split_mode: str,
        rng,
    ) -> _SEFRTreeNode:
        n = idx.shape[0]
        r_n = residuals[idx]
        p_n = p[idx]
        w_n = sample_weight[idx]

        def leaf() -> _SEFRTreeNode:
            if regression:
                v = _mse_leaf_value(r_n, w_n)
            else:
                v = _newton_leaf_value(r_n, p_n, w_n)
            return _SEFRTreeNode(is_leaf=True, value=v)

        if depth >= max_depth or n < min_samples_split:
            return leaf()

        if np.all(r_n > 0) or np.all(r_n < 0):
            return leaf()

        y_bin = (r_n > 0).astype(int)
        if np.unique(y_bin).size < 2:
            return leaf()

        rw = np.abs(r_n) * w_n
        s = rw.sum()
        if s <= 1e-15:
            return leaf()
        rw = rw / s

        X_n = X[idx]
        if regression:
            hess = np.ones_like(r_n)
        else:
            hess = np.clip(p_n * (1.0 - p_n), 1e-10, None)

        # --- Oblique candidate: SEFR direction + gain-optimal cut location ---
        # Per-node min-max conditioning: SEFR's weight formula
        # (avg_pos - avg_neg) / (avg_pos + avg_neg) assumes non-negative features
        # and is sensitive to per-feature scale. Rescale the node's own feature
        # subset to [0, 1] before fitting SEFR, then map the resulting hyperplane
        # back to the original feature space so the split stays oblique in the
        # original coordinates (direction-preserving, identity intact).
        oblique_t = None
        oblique_gain = -np.inf
        coef = None
        proj = None
        lo = X_n.min(axis=0)
        hi = X_n.max(axis=0)
        feat_span = hi - lo
        nondegen = feat_span > 0.0
        if np.any(nondegen):
            X_fit = np.zeros_like(X_n)
            X_fit[:, nondegen] = (X_n[:, nondegen] - lo[nondegen]) / feat_span[nondegen]

            sefr = SEFR(kernel="linear")
            try:
                sefr.fit(X_fit, y_bin, sample_weight=rw)
            except ValueError:
                sefr = None

            if sefr is not None:
                coef_scaled = _dense_sefr_coef(sefr)
                # Map the scaled-space direction back to original space: a
                # coefficient on x_scaled_j = (x_j - lo_j) / rng_j corresponds to
                # coef_scaled_j / rng_j on x_j.
                coef = np.zeros_like(coef_scaled)
                coef[nondegen] = coef_scaled[nondegen] / feat_span[nondegen]
                intercept0 = float(np.asarray(sefr.intercept_).ravel()[0])
                coef, _ = _sanitize_sefr_hyperplane(coef, intercept0)
                # SEFR's intercept places the cut at a class-balanced midpoint,
                # which is biased on imbalanced nodes; pick the threshold that
                # maximizes the GBDT split gain on the (weighted) pseudo-residuals.
                proj = _affine_hyperplane_scores(X_n, coef, 0.0)
                oblique_t, oblique_gain = _best_split_threshold(
                    proj, r_n, w_n, hess, min_samples_leaf
                )

        # --- Axis-aligned candidate (optional by split_mode) ---
        axis_j: Optional[int] = None
        axis_thr = 0.0
        axis_gain = -np.inf
        if split_mode != "sefr_only":
            run_axis = split_mode in ("hybrid", "hybrid_sampled") or (
                split_mode == "axis_fallback"
                and (oblique_t is None or oblique_gain <= 0.0)
            )
            if run_axis:
                feat_idx = None
                if split_mode == "hybrid_sampled":
                    feat_idx = _axis_feature_subset(X_n.shape[1], rng)
                axis_j, axis_thr, axis_gain = _best_axis_split(
                    X_n,
                    r_n,
                    w_n,
                    hess,
                    min_samples_leaf,
                    feature_indices=feat_idx,
                )

        if split_mode == "sefr_only":
            use_axis = False
        elif split_mode == "axis_fallback":
            use_axis = axis_j is not None and oblique_t is None
        else:
            use_axis = axis_j is not None and (
                oblique_t is None or axis_gain > oblique_gain
            )
        if use_axis:
            coef = np.zeros(X_n.shape[1], dtype=np.float64)
            coef[axis_j] = 1.0
            t_star = axis_thr
            proj = X_n[:, axis_j].astype(np.float64, copy=False)
        elif oblique_t is not None:
            t_star = oblique_t
        else:
            return leaf()
        intercept = float(np.clip(-t_star, -1e6, 1e6))

        left_mask = proj <= t_star
        right_mask = ~left_mask
        n_left = int(np.sum(left_mask))
        n_right = int(np.sum(right_mask))

        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            return leaf()
        if n_left == 0 or n_right == 0:
            return leaf()

        idx_left = idx[left_mask]
        idx_right = idx[right_mask]

        node = _SEFRTreeNode(
            is_leaf=False,
            coef=coef,
            intercept=intercept,
        )
        node.left = _SEFRTree._grow(
            X,
            residuals,
            p,
            sample_weight,
            idx_left,
            depth + 1,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            regression=regression,
            split_mode=split_mode,
            rng=rng,
        )
        node.right = _SEFRTree._grow(
            X,
            residuals,
            p,
            sample_weight,
            idx_right,
            depth + 1,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            regression=regression,
            split_mode=split_mode,
            rng=rng,
        )
        return node

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        residuals: np.ndarray,
        p: np.ndarray,
        sample_weight: np.ndarray,
        max_depth: int,
        min_samples_leaf: int,
        min_samples_split: int,
        *,
        regression: bool = False,
        split_mode: str = "hybrid_sampled",
        rng=None,
    ) -> "_SEFRTree":
        if split_mode not in SPLIT_MODE_OPTIONS:
            raise ValueError(
                f"split_mode must be one of {SPLIT_MODE_OPTIONS}, got {split_mode!r}."
            )
        if rng is None:
            rng = np.random.RandomState(0)
        n_samples = X.shape[0]
        idx = np.arange(n_samples, dtype=np.intp)
        root = cls._grow(
            X,
            residuals,
            p,
            sample_weight,
            idx,
            depth=0,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            regression=regression,
            split_mode=split_mode,
            rng=rng,
        )
        return cls(root)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorized prediction: mask all samples through each node (batched matmul per node)."""
        # Internal helper, not a BaseEstimator — avoid check_is_fitted (sklearn 1.9+ get_tags).
        if not hasattr(self, "root_"):
            raise NotFittedError(
                "This _SEFRTree instance is not fitted yet. Call 'fit' before 'predict'."
            )
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float64)
        stack: list[tuple[_SEFRTreeNode, np.ndarray]] = [
            (self.root_, np.ones(n, dtype=bool))
        ]
        while stack:
            node, mask = stack.pop()
            if not np.any(mask):
                continue
            if node.is_leaf:
                out[mask] = node.value
                continue
            Xm = X[mask]
            s = _affine_hyperplane_scores(Xm, node.coef, node.intercept)
            idx = np.nonzero(mask)[0]
            left_sub = s <= 0.0
            right_sub = ~left_sub
            m_left = np.zeros(n, dtype=bool)
            m_right = np.zeros(n, dtype=bool)
            m_left[idx[left_sub]] = True
            m_right[idx[right_sub]] = True
            stack.append((node.right, m_right))
            stack.append((node.left, m_left))
        return out


class SEFRGradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """Gradient boosting (binary and multiclass) with SEFR oblique splits.

    Each boosting stage fits shallow tree(s). At every internal node, a linear
    SEFR model is fit to pseudo-residuals (sign as class, magnitude as weight),
    matching :class:`LinearBoostClassifier` gradient boosting. Leaves output a
    Newton step for the (multinomial) logistic loss.

    For **binary** targets the booster fits one tree per stage to the residual
    ``y - p`` (logistic loss); ``decision_function`` returns the scalar log-odds.
    For **multiclass** targets with ``K`` classes it maintains ``K`` score vectors
    and fits ``K`` trees per stage — one per class — to that class's residual
    ``Y_onehot_k - p_k`` under a softmax (multinomial deviance); each leaf uses the
    same Newton step with ``h = p_k (1 - p_k)``. ``decision_function`` then returns
    a ``(n_samples, K)`` score matrix.

    Boosting uses **effective weights** ``ew_i = sw_i * cw_i`` where ``sw`` is
    ``sample_weight`` (or ones) and ``cw`` comes from ``class_weight`` (sklearn
    ``\"balanced\"`` or a dict). For binary targets ``scale_pos_weight`` optionally
    multiplies weights for the positive class ``classes_[1]``; for multiclass it has
    no meaning and is ignored. SEFR node fitting uses sample weights ``|r| * ew``.

    Every capacity parameter below defaults to ``"auto"``: the value is derived
    from the training set shape in :meth:`fit` by :func:`auto_boosting_config`,
    the way CatBoost adapts its learning rate to the dataset. Passing an
    explicit value disables adaptation for that parameter only. Resolved values
    are exposed as ``n_estimators_``, ``learning_rate_``, ... and the subset that
    was derived is listed in ``auto_config_``.

    Parameters
    ----------
    n_estimators : int or "auto", default="auto"
        Number of boosting iterations (trees). ``"auto"`` uses 200.

    learning_rate : float or "auto", default="auto"
        Shrinkage applied to each tree's prediction. ``"auto"`` uses 0.05 below
        500 rows and 0.1 above, keeping ``learning_rate * n_estimators`` near
        the tuned optimum for that size.

    max_depth : int or "auto", default="auto"
        Maximum depth of each tree (root depth is 0). ``"auto"`` uses 4 below
        500 rows and 6 above.

    min_samples_leaf : int or "auto", default="auto"
        Minimum samples per child when splitting (by row count at node).
        ``"auto"`` uses ``sqrt(n_samples) / 2`` clipped to ``[5, 30]``.

    min_samples_split : int or "auto", default="auto"
        Minimum samples required to attempt a split at a node. ``"auto"`` uses
        twice the resolved ``min_samples_leaf``.

    subsample : float or "auto", default="auto"
        Fraction of rows used to fit each tree (stochastic boosting).
        ``"auto"`` uses 0.8, or 1.0 below 100 rows.

    class_weight : dict, 'balanced', or None, default=None
        Multipliers per class (sklearn-style), combined with ``sample_weight``.

    scale_pos_weight : float or None, default=None
        Binary only: if set, multiplies effective weights for the positive class
        ``classes_[1]`` (similar to XGBoost). ``None`` means ``1.0``. Ignored for
        multiclass targets.

    split_mode : {'auto', 'sefr_only', 'axis_fallback', 'hybrid_sampled', \
'hybrid'}, default='auto'
        How each internal node chooses between SEFR oblique splits and
        axis-aligned single-feature splits (see :class:`SEFRBoostClassifier`).
        ``"auto"`` uses ``'hybrid'`` up to 50 features and ``'hybrid_sampled'``
        beyond that, where scanning every axis-aligned candidate gets expensive.

    random_state : int, RandomState instance or None, default=None
        Random seed for subsampling and ``hybrid_sampled`` feature sampling.

    use_cpp : bool or None, default=None
        When ``True`` (default if the compiled ``_sefr_boost_core`` extension is
        installed), training and prediction run in the C++ backend for much faster
        fit/predict. Set ``False`` to force the pure-Python implementation.

    Attributes
    ----------
    auto_config_ : dict
        Parameters that were resolved from the data shape at ``fit`` time, mapped
        to the value used. Empty when every parameter was given explicitly.

    n_estimators_, learning_rate_, max_depth_, min_samples_leaf_, \
min_samples_split_, subsample_, split_mode_
        Values actually used for training, whether explicit or auto-derived.

    Notes
    -----
    Supports binary and multiclass classification. Multiclass cost scales with the
    number of classes ``K`` (``K`` trees per stage), as in native multinomial GBMs.
    Use ``sklearn.preprocessing.StandardScaler`` in a ``Pipeline`` if features need
    scaling.

    Standalone :class:`~.sefr.SEFR` applies a heuristic scale to scores inside
    ``predict_proba``; this booster uses raw log-odds from ``decision_function``
    for Newton updates, which is the correct separation for gradient boosting.
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left"), StrOptions({"auto"})],
        "learning_rate": [Interval(Real, 0.0, None, closed="left"), StrOptions({"auto"})],
        "max_depth": [Interval(Integral, 1, None, closed="left"), StrOptions({"auto"})],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left"), StrOptions({"auto"})],
        "min_samples_split": [Interval(Integral, 2, None, closed="left"), StrOptions({"auto"})],
        "subsample": [Interval(Real, 0.0, 1.0, closed="right"), StrOptions({"auto"})],
        "class_weight": [StrOptions({"balanced"}), dict, None],
        "scale_pos_weight": [Interval(Real, 0.0, None, closed="neither"), None],
        "split_mode": [StrOptions(set(SPLIT_MODE_OPTIONS) | {"auto"})],
        "random_state": ["random_state"],
        "use_cpp": ["boolean", None],
    }

    def __init__(
        self,
        *,
        n_estimators="auto",
        learning_rate="auto",
        max_depth="auto",
        min_samples_leaf="auto",
        min_samples_split="auto",
        subsample="auto",
        class_weight=None,
        scale_pos_weight=None,
        split_mode: str = "auto",
        random_state=None,
        use_cpp=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.split_mode = split_mode
        self.random_state = random_state
        self.use_cpp = use_cpp

    if SKLEARN_V1_6_OR_LATER:

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.target_tags.required = True
            tags.classifier_tags.multi_class = True
            return tags

    def _more_tags(self) -> dict:
        return {
            "requires_y": True,
            "_xfail_checks": {
                "check_sample_weight_equivalence_on_dense_data": (
                    "Tree structure can change when zero-weight samples are included vs omitted."
                ),
            },
        }

    def __getstate__(self):
        if hasattr(self, "_cpp_core_"):
            return cpp_estimator_getstate(
                self,
                core_cls=SEFRBoostClassifierCore,
                bytes_key="_cpp_core_bytes",
            )
        return self.__dict__.copy()

    def __setstate__(self, state):
        if "_cpp_core_bytes" in state:
            cpp_estimator_setstate(
                self,
                state,
                core_cls=SEFRBoostClassifierCore,
                bytes_key="_cpp_core_bytes",
            )
        else:
            self.__dict__.update(state)
        _backfill_resolved_params(self)

    def save(self, path):
        check_is_fitted(self, ["trees_", "_cpp_core_"], all_or_any=any)
        return save_estimator(self, path)

    @classmethod
    def load(cls, path):
        est = load_estimator(path)
        if not isinstance(est, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(est).__name__}")
        return est

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        if SKLEARN_V1_6_OR_LATER:
            X, y = validate_data(
                self,
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_all_finite=True,
            )
        else:
            X, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                force_all_finite=True,
                estimator=self,
            )
        self.n_features_in_ = X.shape[1]
        check_classification_targets(y)
        y_type = type_of_target(y)
        if y_type not in ("binary", "multiclass"):
            raise ValueError(
                "Only binary or multiclass classification is supported; "
                f"got target type {y_type!r}."
            )

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_classes_ = int(self.classes_.size)
        if self.n_classes_ < 2:
            raise ValueError(
                "Classification requires at least two classes in y; "
                f"got {self.n_classes_} class(es)."
            )

        n_samples = X.shape[0]
        y_original = np.asarray(y)
        _resolve_boosting_params(self, n_samples, self.n_features_in_, AUTO_PARAM_NAMES)

        if sample_weight is not None:
            sw = _check_sample_weight(sample_weight, X, dtype=np.float64)
        else:
            sw = np.ones(n_samples, dtype=np.float64)

        if _should_use_cpp(self.use_cpp):
            if self.n_classes_ == 2:
                ew = _effective_fit_weights(
                    y_idx,
                    y_original,
                    self.classes_,
                    sw,
                    self.class_weight,
                    self.scale_pos_weight,
                )
                w_sum = float(ew.sum()) + 1e-15
                pos_rate = float(
                    np.clip(np.dot(ew, y_idx.astype(np.float64)) / w_sum, 1e-10, 1.0 - 1e-10)
                )
                self.init_score_ = np.log(pos_rate / (1.0 - pos_rate))
            else:
                cw = _per_sample_class_weight(y_original, self.classes_, self.class_weight)
                ew = sw * cw
                w_sum = float(ew.sum()) + 1e-15
                K = self.n_classes_
                self.mc_leaf_scale_ = (K - 1.0) / K
                prior = np.zeros(K, dtype=np.float64)
                for k in range(K):
                    prior[k] = ew[y_idx == k].sum() / w_sum
                prior = np.clip(prior, 1e-10, 1.0 - 1e-10)
                self.init_score_ = np.log(prior)

            self._cpp_core_ = SEFRBoostClassifierCore(
                n_estimators=self.n_estimators_,
                learning_rate=self.learning_rate_,
                max_depth=self.max_depth_,
                min_samples_leaf=self.min_samples_leaf_,
                min_samples_split=self.min_samples_split_,
                subsample=self.subsample_,
                split_mode=self.split_mode_,
                random_state=_cpp_random_seed(self.random_state),
            )
            X_c = np.ascontiguousarray(X, dtype=np.float64)
            self._cpp_core_.fit(
                X_c,
                y_idx.astype(np.int64),
                sample_weight=ew,
            )
            self.trees_ = []
            return self

        rng = check_random_state(self.random_state)

        # Keep the binary path bit-exact (single tree per stage, scalar log-odds);
        # route to the K-tree softmax loop only when K > 2.
        if self.n_classes_ == 2:
            self._fit_binary(X, y_idx, y_original, sw, rng)
        else:
            self._fit_multiclass(X, y_idx, y_original, sw, rng)
        return self

    def _fit_binary(self, X, y_idx, y_original, sw, rng):
        n_samples = X.shape[0]
        y_binary = y_idx.astype(np.float64)

        ew = _effective_fit_weights(
            y_idx,
            y_original,
            self.classes_,
            sw,
            self.class_weight,
            self.scale_pos_weight,
        )
        w_sum = float(ew.sum()) + 1e-15
        pos_rate = float(np.clip(np.dot(ew, y_binary) / w_sum, 1e-10, 1.0 - 1e-10))
        self.init_score_ = np.log(pos_rate / (1.0 - pos_rate))
        F = np.full(n_samples, self.init_score_, dtype=np.float64)

        self.trees_: list[_SEFRTree] = []

        for _ in range(self.n_estimators_):
            p = 1.0 / (1.0 + np.exp(-F))
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
            residuals = y_binary - p

            if self.subsample_ < 1.0:
                n_sub = max(1, int(self.subsample_ * n_samples))
                sub_idx = rng.choice(n_samples, size=n_sub, replace=False)
                X_b = X[sub_idx]
                r_b = residuals[sub_idx]
                p_b = p[sub_idx]
                w_b = ew[sub_idx]
            else:
                X_b, r_b, p_b, w_b = X, residuals, p, ew

            tree = _SEFRTree.fit(
                X_b,
                r_b,
                p_b,
                w_b,
                max_depth=self.max_depth_,
                min_samples_leaf=self.min_samples_leaf_,
                min_samples_split=self.min_samples_split_,
                regression=False,
                split_mode=self.split_mode_,
                rng=rng,
            )
            self.trees_.append(tree)
            F = F + self.learning_rate_ * tree.predict(X)

        self.F_train_ = F

    def _fit_multiclass(self, X, y_idx, y_original, sw, rng):
        n_samples = X.shape[0]
        K = self.n_classes_
        # Friedman's multinomial Newton step scales the per-class leaf update by
        # (K-1)/K; the diagonal Hessian h = p_k (1 - p_k) overestimates curvature
        # for the coupled softmax, so this corrects the step magnitude. The same
        # factor is reapplied in ``_raw_F`` when reconstructing scores at predict
        # time. The binary path keeps the full (unscaled) Newton step.
        self.mc_leaf_scale_ = (K - 1.0) / K
        Y = np.eye(K, dtype=np.float64)[y_idx]  # one-hot, shape (n, K)

        # Multiclass effective weights: sample_weight * class_weight only.
        # scale_pos_weight is binary-specific (positive class = classes_[1]) and
        # has no meaning here, so it is dropped.
        cw = _per_sample_class_weight(y_original, self.classes_, self.class_weight)
        ew = sw * cw
        w_sum = float(ew.sum()) + 1e-15

        # Weighted class priors for softmax-invariant init scores (shift-invariant).
        prior = (ew[:, None] * Y).sum(axis=0) / w_sum  # shape (K,)
        prior = np.clip(prior, 1e-10, 1.0 - 1e-10)
        self.init_score_ = np.log(prior)  # shape (K,)
        F = np.tile(self.init_score_, (n_samples, 1))  # (n, K)

        # One list of K trees per boosting stage.
        self.trees_: list[list[_SEFRTree]] = []

        for _ in range(self.n_estimators_):
            # Softmax with max-subtraction for numerical stability.
            Fmax = F.max(axis=1, keepdims=True)
            expF = np.exp(F - Fmax)
            P = expF / expF.sum(axis=1, keepdims=True)
            P = np.clip(P, 1e-10, 1.0 - 1e-10)

            if self.subsample_ < 1.0:
                n_sub = max(1, int(self.subsample_ * n_samples))
                sub_idx = rng.choice(n_samples, size=n_sub, replace=False)
            else:
                sub_idx = slice(None)

            stage_trees: list[_SEFRTree] = []
            for k in range(K):
                residual_k = Y[:, k] - P[:, k]
                tree = _SEFRTree.fit(
                    X[sub_idx],
                    residual_k[sub_idx],
                    P[sub_idx, k],
                    ew[sub_idx],
                    max_depth=self.max_depth_,
                    min_samples_leaf=self.min_samples_leaf_,
                    min_samples_split=self.min_samples_split_,
                    regression=False,
                    split_mode=self.split_mode_,
                    rng=rng,
                )
                stage_trees.append(tree)
                F[:, k] += self.learning_rate_ * self.mc_leaf_scale_ * tree.predict(X)
            self.trees_.append(stage_trees)

        self.F_train_ = F

    def decision_function(self, X):
        check_is_fitted(self, ["trees_", "_cpp_core_"], all_or_any=any)
        if SKLEARN_V1_6_OR_LATER:
            X = validate_data(
                self,
                X,
                accept_sparse=False,
                dtype=np.float64,
                reset=False,
                ensure_all_finite=True,
            )
        else:
            X = validate_data(
                self,
                X,
                accept_sparse=False,
                dtype=np.float64,
                force_all_finite=True,
            )
        if hasattr(self, "_cpp_core_"):
            return self._cpp_core_.decision_function(
                np.ascontiguousarray(X, dtype=np.float64)
            )
        if self.n_classes_ == 2:
            F = np.full(X.shape[0], self.init_score_, dtype=np.float64)
            for tree in self.trees_:
                F = F + self.learning_rate_ * tree.predict(X)
            # F is log-odds for classes_[1] vs classes_[0] (y encoded as class index).
            return F
        return self._raw_F(X)

    def _raw_F(self, X):
        """Multiclass softmax scores ``(n_samples, K)`` (X already validated)."""
        n = X.shape[0]
        F = np.tile(self.init_score_, (n, 1))
        for stage in self.trees_:
            for k, tree in enumerate(stage):
                F[:, k] += self.learning_rate_ * self.mc_leaf_scale_ * tree.predict(X)
        return F

    def predict_proba(self, X):
        df = self.decision_function(X)
        if self.n_classes_ == 2:
            proba_pos = 1.0 / (1.0 + np.exp(-df))
            proba_pos = np.clip(proba_pos, 1e-10, 1.0 - 1e-10)
            return np.column_stack((1.0 - proba_pos, proba_pos))
        # df is the (n, K) score matrix; softmax with max-subtraction.
        Fmax = df.max(axis=1, keepdims=True)
        expF = np.exp(df - Fmax)
        return expF / expF.sum(axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.n_classes_ == 2:
            return self.classes_[(proba[:, 1] >= 0.5).astype(int)]
        return self.classes_[np.argmax(proba, axis=1)]


class SEFRGradientBoostingRegressor(RegressorMixin, BaseEstimator):
    """Gradient boosting regression with SEFR oblique splits (squared error loss).

    Matches the classification booster structure: each stage fits a shallow tree
    whose internal nodes use a linear SEFR split on pseudo-residuals ``y - F``,
    with sample weights ``|r| * sw`` at nodes. Leaf values are weighted mean
    residuals (the Newton step for squared loss). The initial prediction is the
    weighted mean of ``y``.

    Parameters are the same as :class:`SEFRGradientBoostingClassifier` except
    ``class_weight`` and ``scale_pos_weight`` are not used. Capacity parameters
    default to ``"auto"`` and are resolved from the training set shape by
    :func:`auto_boosting_config`; see ``auto_config_`` for what was derived.

    Notes
    -----
    Single-output regression only (``y`` one-dimensional after squeezing).
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left"), StrOptions({"auto"})],
        "learning_rate": [Interval(Real, 0.0, None, closed="left"), StrOptions({"auto"})],
        "max_depth": [Interval(Integral, 1, None, closed="left"), StrOptions({"auto"})],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left"), StrOptions({"auto"})],
        "min_samples_split": [Interval(Integral, 2, None, closed="left"), StrOptions({"auto"})],
        "subsample": [Interval(Real, 0.0, 1.0, closed="right"), StrOptions({"auto"})],
        "split_mode": [StrOptions(set(SPLIT_MODE_OPTIONS) | {"auto"})],
        "random_state": ["random_state"],
        "use_cpp": ["boolean", None],
    }

    def __init__(
        self,
        *,
        n_estimators="auto",
        learning_rate="auto",
        max_depth="auto",
        min_samples_leaf="auto",
        min_samples_split="auto",
        subsample="auto",
        split_mode: str = "auto",
        random_state=None,
        use_cpp=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.split_mode = split_mode
        self.random_state = random_state
        self.use_cpp = use_cpp

    if SKLEARN_V1_6_OR_LATER:

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.target_tags.required = True
            return tags

    def _more_tags(self) -> dict:
        return {
            "requires_y": True,
            "_xfail_checks": {
                "check_sample_weight_equivalence_on_dense_data": (
                    "Tree structure can change when zero-weight samples are included vs omitted."
                ),
                "check_regressors_train": (
                    "Default depth/leaf settings are conservative; R² on sklearn checker "
                    "data may not exceed 0.5 without stronger capacity."
                ),
            },
        }

    def __getstate__(self):
        if hasattr(self, "_cpp_core_"):
            return cpp_estimator_getstate(
                self,
                core_cls=SEFRBoostRegressorCore,
                bytes_key="_cpp_core_bytes",
            )
        return self.__dict__.copy()

    def __setstate__(self, state):
        if "_cpp_core_bytes" in state:
            cpp_estimator_setstate(
                self,
                state,
                core_cls=SEFRBoostRegressorCore,
                bytes_key="_cpp_core_bytes",
            )
        else:
            self.__dict__.update(state)
        _backfill_resolved_params(self)

    def save(self, path):
        check_is_fitted(self, ["trees_", "_cpp_core_"], all_or_any=any)
        return save_estimator(self, path)

    @classmethod
    def load(cls, path):
        est = load_estimator(path)
        if not isinstance(est, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(est).__name__}")
        return est

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        if SKLEARN_V1_6_OR_LATER:
            X, y = validate_data(
                self,
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_all_finite=True,
            )
        else:
            X, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                force_all_finite=True,
                estimator=self,
            )
        self.n_features_in_ = X.shape[1]
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 0:
            y = np.array([float(y)], dtype=np.float64)
        elif y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError(
                    "SEFRGradientBoostingRegressor only supports single-target regression."
                )
            y = y.ravel()
        elif y.ndim != 1:
            raise ValueError(
                "SEFRGradientBoostingRegressor only supports single-target regression."
            )
        n_samples = X.shape[0]
        if y.shape[0] != n_samples:
            raise ValueError("X and y must have the same number of samples.")
        _resolve_boosting_params(self, n_samples, self.n_features_in_, AUTO_PARAM_NAMES)

        if sample_weight is not None:
            sw = _check_sample_weight(sample_weight, X, dtype=np.float64)
        else:
            sw = np.ones(n_samples, dtype=np.float64)

        if _should_use_cpp(self.use_cpp):
            w_sum = float(sw.sum()) + 1e-15
            self.init_score_ = float(np.dot(sw, y) / w_sum)
            self._cpp_core_ = SEFRBoostRegressorCore(
                n_estimators=self.n_estimators_,
                learning_rate=self.learning_rate_,
                max_depth=self.max_depth_,
                min_samples_leaf=self.min_samples_leaf_,
                min_samples_split=self.min_samples_split_,
                subsample=self.subsample_,
                split_mode=self.split_mode_,
                random_state=_cpp_random_seed(self.random_state),
            )
            X_c = np.ascontiguousarray(X, dtype=np.float64)
            self._cpp_core_.fit(X_c, y, sample_weight=sw)
            self.trees_ = []
            return self

        w_sum = float(sw.sum()) + 1e-15
        self.init_score_ = float(np.dot(sw, y) / w_sum)
        F = np.full(n_samples, self.init_score_, dtype=np.float64)

        rng = check_random_state(self.random_state)
        self.trees_: list[_SEFRTree] = []

        for _ in range(self.n_estimators_):
            residuals = y - F

            if self.subsample_ < 1.0:
                n_sub = max(1, int(self.subsample_ * n_samples))
                sub_idx = rng.choice(n_samples, size=n_sub, replace=False)
                X_b = X[sub_idx]
                r_b = residuals[sub_idx]
                w_b = sw[sub_idx]
            else:
                X_b, r_b, w_b = X, residuals, sw

            p_dummy = np.ones(X_b.shape[0], dtype=np.float64)
            tree = _SEFRTree.fit(
                X_b,
                r_b,
                p_dummy,
                w_b,
                max_depth=self.max_depth_,
                min_samples_leaf=self.min_samples_leaf_,
                min_samples_split=self.min_samples_split_,
                regression=True,
                split_mode=self.split_mode_,
                rng=rng,
            )
            self.trees_.append(tree)
            F = F + self.learning_rate_ * tree.predict(X)

        self.F_train_ = F
        return self

    def predict(self, X):
        check_is_fitted(self, ["trees_", "_cpp_core_"], all_or_any=any)
        if SKLEARN_V1_6_OR_LATER:
            X = validate_data(
                self,
                X,
                accept_sparse=False,
                dtype=np.float64,
                reset=False,
                ensure_all_finite=True,
            )
        else:
            X = validate_data(
                self,
                X,
                accept_sparse=False,
                dtype=np.float64,
                force_all_finite=True,
            )
        if hasattr(self, "_cpp_core_"):
            return self._cpp_core_.predict(np.ascontiguousarray(X, dtype=np.float64))
        out = np.full(X.shape[0], self.init_score_, dtype=np.float64)
        for tree in self.trees_:
            out = out + self.learning_rate_ * tree.predict(X)
        return out


# Public aliases
SEFRBoostClassifier = SEFRGradientBoostingClassifier
SEFRBoostRegressor = SEFRGradientBoostingRegressor
PrismBoostClassifier = SEFRGradientBoostingClassifier
PrismBoostRegressor = SEFRGradientBoostingRegressor
