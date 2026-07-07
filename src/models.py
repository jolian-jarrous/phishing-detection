"""Model factory for the five classifiers compared in the study.

Scale-sensitive models (Logistic Regression and MLP) are wrapped in a
``Pipeline`` with ``StandardScaler``. Tree-based models don't need scaling,
so they're used directly. This is the preprocessing step the original article
missed.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def build_models(include_xgboost: bool = True) -> dict:
    """Return a dict of {name: estimator} for every model used in the study.

    Set ``include_xgboost=False`` if the xgboost package isn't installed
    (useful for CI environments where we don't want to pin heavy deps).
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=300,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        except ImportError:
            pass

    return models
