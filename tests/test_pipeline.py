"""Basic smoke tests. Run with:  python -m pytest tests/

These aren't full unit tests --- the project is a critical reproduction
study, not a library --- but they check the pipeline modules do the right
thing on a small synthetic dataframe.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.data import duplicate_summary, deduplicate, split_features_and_label
from src.features import add_url_complexity, spearman_correlations
from src.models import build_models


@pytest.fixture
def toy_df():
    # 5 rows, 3 URLs, 2 of them are duplicated feature vectors under different
    # Domain strings (row 0 and row 3 share features but not Domain), and rows
    # 1 and 4 are full-row duplicates (same Domain AND same features).
    return pd.DataFrame({
        "Domain":      ["a.com", "b.com", "c.com", "d.com", "b.com"],
        "Have_IP":     [0, 1, 0, 0, 1],
        "URL_Length":  [1, 0, 1, 1, 0],
        "URL_Depth":   [2, 1, 3, 2, 1],
        "Label":       [1, 0, 1, 1, 0],
    })


def test_duplicate_summary_distinguishes_the_two_kinds(toy_df):
    summary = duplicate_summary(toy_df)
    # rows 1 and 4 are identical everywhere -> 1 full-row duplicate
    assert summary["full_row_duplicates"] == 1
    # under features-only: rows 0/3 share features (label included), rows 1/4 share features
    # so there are 2 duplicated feature vectors
    assert summary["feature_vector_duplicates"] == 2
    assert summary["total_rows"] == 5


def test_deduplicate_on_features(toy_df):
    clean = deduplicate(toy_df, on_features_only=True)
    # 5 rows -> 3 unique feature vectors
    assert len(clean) == 3


def test_deduplicate_on_full_rows(toy_df):
    clean = deduplicate(toy_df, on_features_only=False)
    # 5 rows -> 4 unique full rows (the b.com dup collapses)
    assert len(clean) == 4


def test_split_features_and_label_drops_domain(toy_df):
    X, y = split_features_and_label(toy_df)
    assert "Domain" not in X.columns
    assert "Label" not in X.columns
    assert set(X.columns) == {"Have_IP", "URL_Length", "URL_Depth"}
    assert len(y) == len(toy_df)


def test_add_url_complexity(toy_df):
    X, _ = split_features_and_label(toy_df)
    X = add_url_complexity(X)
    assert "URL_Complexity" in X.columns
    # URL_Complexity = URL_Length * (URL_Depth + 1)
    expected = X["URL_Length"] * (X["URL_Depth"] + 1)
    np.testing.assert_array_equal(X["URL_Complexity"].values, expected.values)


def test_spearman_pairs_returned_sorted(toy_df):
    X, _ = split_features_and_label(toy_df)
    pairs = spearman_correlations(X, threshold=0.0)
    # sorted descending by rho
    assert list(pairs["rho"]) == sorted(pairs["rho"], reverse=True)


def test_build_models_returns_expected_keys():
    models = build_models(include_xgboost=False)
    assert set(models.keys()) == {
        "LogisticRegression", "DecisionTree", "RandomForest", "MLP",
    }
