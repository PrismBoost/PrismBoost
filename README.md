# PrismBoost

[![PyPI](https://img.shields.io/pypi/v/prismboost.svg)](https://pypi.org/project/prismboost/)
[![Python](https://img.shields.io/pypi/pyversions/prismboost.svg)](https://pypi.org/project/prismboost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/PrismBoost/PrismBoost/blob/main/LICENSE)

**PrismBoost** is a gradient-boosting classifier/regressor that uses **SEFR oblique splits** at internal nodes (hyperplane splits instead of axis-aligned thresholds), with an optional fast C++ backend.

```python
from prismboost import PrismBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

clf = PrismBoostClassifier(n_estimators=100, max_depth=3, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

## Why oblique boosting?

Axis-aligned GBDTs approximate curved boundaries with staircases. PrismBoost fits **linear (oblique) splits**, so decision surfaces on non-linear problems are typically smoother.

<p align="center">
  <img src="docs/images/moons_surface.jpg" alt="PrismBoost vs XGBoost probability surfaces on moons" width="720"/>
</p>

<p align="center"><em>Predicted-probability surfaces on moons: PrismBoost (left) vs XGBoost (right).</em></p>

<p align="center">
  <img src="docs/images/decision_boundaries_grid.jpg" alt="Decision boundaries on six synthetic 2D datasets" width="720"/>
</p>

<p align="center"><em>Decision boundaries on six synthetic 2D datasets (rows) across classifiers (columns). Lower surface roughness <code>S</code> is smoother.</em></p>

<p align="center">
  <img src="docs/images/smoothness_bar.png" alt="Mean surface roughness by classifier" width="560"/>
</p>

## Benchmark highlights (PMLB)

Evaluated on **121** Penn Machine Learning Benchmark classification datasets against strong baselines (CatBoost, LightGBM, LightGBM-linear, XGBoost, SPORF, Random Forest, Logistic Regression). Hyperparameters are tuned with Optuna; scores are repeated stratified CV.

**Median scores** (higher is better for F1 / ROC-AUC; lower is better for inference latency):

| Model | Median macro-F1 | Median ROC-AUC | Median inference (ms/row) |
|-------|----------------:|---------------:|-------------------------:|
| **PrismBoost** | **0.903** | 0.976 | **0.056** |
| CatBoost | 0.892 | **0.976** | 0.184 |
| XGBoost | 0.875 | 0.970 | 0.615 |
| LightGBM-linear | 0.875 | 0.970 | 0.567 |
| LightGBM | 0.870 | 0.972 | 0.550 |
| Random Forest | 0.862 | 0.964 | 5.140 |
| SPORF | 0.856 | 0.970 | 10.058 |
| Logistic Regression | 0.822 | 0.942 | 0.053 |

**Average ranks** (1 = best; Friedman tests significant for macro-F1 and ROC-AUC):

| Model | Macro-F1 rank | ROC-AUC rank |
|-------|--------------:|-------------:|
| CatBoost | **3.33** | **3.48** |
| **PrismBoost** | **3.88** | 4.26 |
| LightGBM-linear | 3.99 | 4.26 |
| LightGBM | 4.10 | 4.10 |
| XGBoost | 4.31 | 4.24 |
| Logistic Regression | 5.31 | 5.66 |
| Random Forest | 5.48 | 5.31 |
| SPORF | 5.61 | 4.70 |

<p align="center">
  <img src="docs/images/cd_diagram.png" alt="Nemenyi critical-difference diagrams" width="720"/>
</p>

<p align="center"><em>Nemenyi critical-difference diagrams (α = 0.05). Models connected by a bar are not significantly different.</em></p>

On these data, PrismBoost is competitive with modern GBDTs on accuracy while remaining among the **fastest at inference** (second only to logistic regression; fastest non-linear model by median latency).

## Install

```bash
pip install prismboost
```

Requires Python 3.10–3.13. A C++17 compiler and CMake are used when building the optional native extension (included for common platforms via wheels / sdist build).

From source (editable / development):

```bash
pip install -e ".[dev]"
```

If the C++ extension fails to build, the package still works via the pure-Python backend.

Extras:

```bash
pip install "prismboost[examples]"   # Optuna for the tuning example
pip install "prismboost[dev]"        # pytest, ruff
```

## Public API

| Name | Description |
|------|-------------|
| `PrismBoostClassifier` | Primary classifier (sklearn-compatible) |
| `PrismBoostRegressor` | Primary regressor |
| `SEFRBoostClassifier` / `SEFRBoostRegressor` | Compatibility aliases |
| `SEFR` | Linear weak learner used inside oblique splits |

## Features

- Oblique tree splits from SEFR (closed-form linear separator)
- Binary and multiclass classification; regression
- Optional C++ backend for faster fit/predict
- sklearn estimator API (`fit`, `predict`, `predict_proba`, pipelines, pickling)
- Works with Optuna / GridSearchCV / RandomizedSearchCV

## Examples

```bash
pip install "prismboost[examples]"
python examples/quickstart.py
python examples/optuna_tuning.py --n-trials 20
```

- [`examples/quickstart.py`](examples/quickstart.py) — minimal fit / score
- [`examples/optuna_tuning.py`](examples/optuna_tuning.py) — Optuna CV search over trees, depth, learning rate, `split_mode`, and scaler

Docs mirror: [`docs/optuna_tuning.rst`](docs/optuna_tuning.rst).

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## License

This project is licensed under the [MIT License](https://github.com/PrismBoost/PrismBoost/blob/main/LICENSE).

Third-party note: `prismboost._utils` includes code derived from [wnb](https://github.com/msamogh/wnb) under the [BSD 3-Clause License](https://github.com/PrismBoost/PrismBoost/blob/main/src/prismboost/licenses/BSD-3-Clause.txt).

## Citation

If you use PrismBoost in academic work, please cite the accompanying paper (to be updated on publication).

## Authors

- Hamidreza Keshavarz
- Reza Rawassizadeh
