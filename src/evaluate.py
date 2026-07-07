"""Cross-validation and metrics for the phishing classifiers."""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import (
    fbeta_score,
    make_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .models import RANDOM_STATE


def build_scoring() -> dict:
    """Return the metric suite as an sklearn scoring dict.

    Includes F2 (beta=2) because in phishing detection false negatives are
    more costly than false positives, so a metric that weights recall more
    heavily than precision is more aligned with what a defender cares about.
    """
    return {
        "accuracy":  "accuracy",
        "precision": "precision",
        "recall":    "recall",
        "f1":        "f1",
        "f2":        make_scorer(fbeta_score, beta=2),
        "roc_auc":   "roc_auc",
        "mcc":       make_scorer(matthews_corrcoef),
    }


def cross_validate_models(models: dict, X, y, n_splits: int = 5) -> pd.DataFrame:
    """Run stratified k-fold CV for every model, return a tidy dataframe.

    Columns: Model, Accuracy, Precision, Recall, F1, F2, ROC-AUC, MCC,
    plus ``Accuracy_std`` for the standard deviation across folds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = build_scoring()

    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({
            "Model": name,
            "Accuracy":     scores["test_accuracy"].mean(),
            "Accuracy_std": scores["test_accuracy"].std(),
            "Precision":    scores["test_precision"].mean(),
            "Recall":       scores["test_recall"].mean(),
            "F1":           scores["test_f1"].mean(),
            "F2":           scores["test_f2"].mean(),
            "ROC-AUC":      scores["test_roc_auc"].mean(),
            "MCC":          scores["test_mcc"].mean(),
        })
    return pd.DataFrame(rows)
