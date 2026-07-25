# PrismBoost

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

## Install

Requires Python 3.10–3.13, a C++17 compiler, and CMake (for the optional native extension).

```bash
pip install .
# or editable:
pip install -e ".[dev]"
```

If the C++ extension fails to build, the package still works via the pure-Python backend.

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

## Quick example

See [`examples/quickstart.py`](examples/quickstart.py).

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use PrismBoost in academic work, please cite the accompanying paper (to be updated on publication).

## Authors

- Hamidreza Keshavarz
- Reza Rawassizadeh
