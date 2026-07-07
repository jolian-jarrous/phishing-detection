"""Feature engineering utilities.

The original article uses 17 hand-crafted features straight out of the CSV.
This module adds one engineered feature (``URL_Complexity``) and provides a
Spearman correlation helper for the redundancy discussion.
"""
from __future__ import annotations
import pandas as pd


def add_url_complexity(X: pd.DataFrame) -> pd.DataFrame:
    """Add an interaction feature combining URL_Length and URL_Depth.

    Rationale: URL_Length and URL_Depth are individually the two strongest
    predictors, and correlate weakly with each other. An explicit interaction
    picks up signal that neither raw feature captures alone.

    ``+ 1`` on URL_Depth so a URL with depth 0 still contributes when
    URL_Length is 1.
    """
    X = X.copy()
    X["URL_Complexity"] = X["URL_Length"] * (X["URL_Depth"] + 1)
    return X


def spearman_correlations(X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Return pairs of features whose Spearman |rho| exceeds ``threshold``.

    Spearman (not Pearson) because the features are ordinal / binary rather
    than continuous-Gaussian. Kendall would also work; Spearman is fine at
    n = 10000.
    """
    corr = X.corr(method="spearman").abs()
    pairs = (
        corr.where(pd.notna(corr) & (corr < 1.0))
            .stack()
            .rename("rho")
            .reset_index()
    )
    pairs = pairs[pairs["rho"] >= threshold]
    pairs = pairs.rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    # deduplicate (a, b) vs (b, a)
    pairs["key"] = pairs.apply(
        lambda r: tuple(sorted([r["feature_a"], r["feature_b"]])), axis=1
    )
    pairs = pairs.drop_duplicates("key").drop(columns=["key"])
    return pairs.sort_values("rho", ascending=False).reset_index(drop=True)
