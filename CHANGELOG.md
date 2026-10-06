# Changelog

Notable changes per release. Dates are release dates on PyPI.

## 0.5.0 (2026-10-06)

### Added

- Wall-clock budget: `fit(..., time_limit=seconds)`. The boosting loop checks the elapsed time
  after each stage and stops, keeping the stages fitted so far, in the C++ core and the NumPy
  backend, for the classifier and the regressor. It composes with early stopping, ending at
  whichever comes first and keeping the best stage. On a 8,000 x 40 table with a 4,000-stage cap,
  an unbudgeted fit took 326 s against 3.1 s with `time_limit=3`.

  `time_limit` is a `fit` argument rather than a constructor parameter: it describes one call's
  budget, not the model, so it stays out of `get_params` and out of `clone`. The check happens
  between stages, so one stage on a very large table can still overshoot the budget; this bounds
  the loop rather than guaranteeing a deadline.

  Asked for by the TabArena maintainers (autogluon/tabarena#629), where every configuration gets
  a one-hour budget and 16 fits had run past it, the longest at 7,618 s.

### Removed from the known gaps

- The 0.4.0 entry listed the absence of a wall-clock budget as a known gap. This closes it. The
  fixed stopping criterion (log loss for classifiers, squared error for the regressor) remains.

## 0.4.1 (2026-10-01)

### Fixed

- The C++ backend crashed on Windows: the seed passed to the C++ core was drawn with NumPy's
  default integer type, C `long`, which is 32-bit on Windows, so every C++-backed fit raised
  `ValueError: high is out of bounds for int32`. Seeds are now drawn as 64-bit integers. On Linux
  and macOS `long` was already 64-bit, so seeds and results there are unchanged.

### Added

- Binary wheels for Linux (x86_64 and aarch64, manylinux and musllinux), macOS (Intel and Apple
  Silicon) and Windows (x64, and ARM64 for CPython 3.11+), for CPython 3.10 to 3.13. Earlier
  releases shipped one macOS wheel tagged for macOS 26, so every other platform compiled the C++
  core from the sdist. Each wheel is tested against its installed copy before release.

## 0.4.0 (2026-10-01)

### Added

- Early stopping. `fit` accepts `eval_set=(X_val, y_val)` and the estimators accept
  `early_stopping_rounds`, so a fit stops once the validation loss has not improved for that many
  stages and the ensemble is truncated to the best one. `n_estimators` becomes an upper bound,
  `best_iteration_` reports the stages kept and `validation_loss_` the loss after each fitted
  stage. Refitting with `n_estimators=best_iteration_` and the same `random_state` reproduces the
  early-stopped model. Passing `eval_set` alone records `validation_loss_` without stopping.
  Implemented in the C++ core and the NumPy backend, for the classifier and the regressor.
  Validation rows whose class was unseen in training are left out of the loss.

  The loss is fixed: log loss for classifiers, squared error for the regressor. There is no hook
  for a different criterion yet, so a caller that selects on another metric cannot drive the
  stopping decision.

### Known gaps

- No wall-clock budget. A fit runs to `n_estimators` or to early stopping, so a caller that needs
  to bound fit time has to bound the round count instead.
- Single-threaded: the C++ core links no OpenMP and the Python backend is NumPy-level, so there is
  no thread-count parameter.

## 0.3.0 (2026-09-30)

### Results change on upgrade

- Dropped `-ffast-math` from the C++ core. It implies `-ffinite-math-only`, which tells the
  compiler that NaN and infinity cannot occur, and the split search uses both as sentinels
  (`kSplitNone` is a quiet NaN, the best gain starts at `-infinity`). Under `-O3 -ffast-math`
  clang dropped the best-candidate bookkeeping in `best_split_threshold`, so trees collapsed to
  a single leaf and the booster predicted its initial score everywhere. Where it did not break
  outright it cost accuracy: 5-fold CV ROC-AUC on sonar 0.9053 to 0.9256, analcatdata_boxing1
  0.8600 to 0.8755, ionosphere 0.9662 to 0.9735. `-O3` is unchanged; the replacement flags are
  `-fno-math-errno -ffp-contract=fast`, and an 8000x30 fit over 50 rounds measured 2.519s before
  against 2.535s after. **Expect different numbers from 0.2.0 at identical hyperparameters and
  seeds.**

### Added

- `reg_lambda`, an L2 penalty on leaf weights, on both estimators and both backends. It enters
  the Newton step as `sum(w r) / (sum(w h) + reg_lambda)` and the split gain as
  `G^2 / (H + reg_lambda)`, matching XGBoost's parameter of the same name. Default `0.0`, so it
  changes nothing unless set. On a 3-class problem at a 2/10/88 split over 200 rounds, test log
  loss goes 0.399 at 0.0, 0.233 at 1.0, 0.207 at 5.0 and 0.188 at 20.0, against 0.437 for
  predicting the class prior. Neutral on ROC-AUC over 33 tuned PMLB datasets (13 wins, 16
  losses, median 0.0000): it acts on calibrated probabilities, not on ranking.

### Fixed

- Multiclass `predict_proba` now clips to `[1e-10, 1-1e-10]` and renormalizes, as the binary
  path already did. A saturated softmax previously returned exactly 0.0 and 1.0, which log loss
  scores as an infinite penalty on a single wrong row.
- `pandas` added to the `dev` extra, which `test_high_dimensional_onehot_sparse_sefr_coef`
  imports, and `is_classifier` in the SEFR test now takes an instance as scikit-learn 1.9
  requires. CI passes on Python 3.10 through 3.13.

## 0.2.0 (2026-08-04)

### Added

- `second_order` selects between Newton boosting (default) and the first-order gradient
  ablation. Not supported by the C++ backend, which implements the Newton criterion only.
- Every capacity parameter defaults to `"auto"` and is resolved from the training set shape at
  `fit` time, calibrated against the per-dataset Optuna optima of 121 PMLB datasets. Resolved
  values are exposed as `n_estimators_`, `learning_rate_` and so on, with the derived subset in
  `auto_config_`. Only the shape of `X` is used, never `y`.

### Changed

- PrismBoost is the primary name. `PrismBoostClassifier` and `PrismBoostRegressor` are the
  classes; `SEFRBoost*` and `SEFRGradientBoosting*` remain as aliases, and pickles written under
  the old module paths still load.
- To reproduce pre-0.2 behaviour, pass the old values explicitly: `n_estimators=100,
  learning_rate=0.1, max_depth=3, min_samples_leaf=10, min_samples_split=2, subsample=1.0,
  split_mode="hybrid_sampled"`.

## 0.1.2 (2026-07-26)

- First PyPI release under the PrismBoost organization: gradient boosting with SEFR oblique
  splits for classification and regression, with an optional C++ backend, an Optuna tuning
  example, and the PMLB benchmark results in the README.
