"""Data loading, cleaning, and duplicate analysis for the phishing dataset."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "urldata.csv"


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the phishing URL dataset from CSV.

    Returns the raw dataframe with the Domain column still attached. Use
    `split_features_and_label` when you actually want to feed data to a model.
    """
    return pd.read_csv(path)


def split_features_and_label(df: pd.DataFrame):
    """Return (X, y) with the Domain column dropped from X.

    The Domain column is the raw URL string and is only useful for error
    analysis (looking at *which* URLs the model gets wrong). It cannot be
    fed to a numerical model.
    """
    df = df.copy()
    if "Domain" in df.columns:
        df = df.drop(columns=["Domain"])
    y = df["Label"]
    X = df.drop(columns=["Label"])
    return X, y


def duplicate_summary(df: pd.DataFrame) -> dict:
    """Return a summary of the two different kinds of duplicates in the dataset.

    This is the distinction Amer's feedback asked for:

    - ``full_row_duplicates``:
        Rows that are identical across ALL columns, including the Domain
        (raw URL). If this number is > 0, it means the same URL literally
        appears more than once in the raw data.

    - ``feature_vector_duplicates``:
        Rows that are identical AFTER dropping the Domain column. This is
        different: two *different* URLs can produce the same 17-feature
        vector because the feature representation is coarse. This is the
        number that actually matters for the cross-validation critique,
        because it's the feature vector (not the URL string) that the
        model sees.
    """
    full_dup = int(df.duplicated().sum())

    feats_only = df.drop(columns=["Domain"]) if "Domain" in df.columns else df
    feat_dup = int(feats_only.duplicated().sum())

    unique_full = int((~df.duplicated()).sum())
    unique_feat = int((~feats_only.duplicated()).sum())

    return {
        "total_rows": len(df),
        "full_row_duplicates": full_dup,
        "unique_full_rows": unique_full,
        "feature_vector_duplicates": feat_dup,
        "unique_feature_vectors": unique_feat,
    }


def deduplicate(df: pd.DataFrame, on_features_only: bool = True) -> pd.DataFrame:
    """Drop duplicates from the dataframe.

    Parameters
    ----------
    df :
        The dataframe to clean.
    on_features_only :
        If True (default), drop rows whose numerical features + label are the
        same, ignoring the Domain column. This is what actually matters for
        modelling. If False, only drop rows that are identical everywhere
        including the Domain column.
    """
    if on_features_only and "Domain" in df.columns:
        subset = [c for c in df.columns if c != "Domain"]
        return df.drop_duplicates(subset=subset).reset_index(drop=True)
    return df.drop_duplicates().reset_index(drop=True)
