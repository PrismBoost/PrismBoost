Quickstart
==========

Install
-------

From the repository root:

.. code-block:: bash

   pip install -e .

Or from PyPI:

.. code-block:: bash

   pip install prismboost

Minimal training example
------------------------

.. code-block:: python

   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.metrics import f1_score, roc_auc_score
   from prismboost import PrismBoostClassifier

   X, y = load_breast_cancer(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, stratify=y, random_state=42
   )

   model = Pipeline([
       ("scale", StandardScaler()),
       ("clf", PrismBoostClassifier(random_state=42)),
   ])

   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   y_prob = model.predict_proba(X_test)[:, 1]

   print("F1(weighted):", f1_score(y_test, y_pred, average="weighted"))
   print("ROC-AUC:", roc_auc_score(y_test, y_prob))

Capacity parameters (``n_estimators``, ``learning_rate``, ``max_depth``,
``min_samples_leaf``, ``min_samples_split``, ``subsample``, ``split_mode``)
default to ``"auto"`` and are derived from the training-set shape; see
:doc:`adaptive_defaults`. Pass any of them explicitly to pin it.

