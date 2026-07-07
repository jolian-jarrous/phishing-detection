"""Reruns cross-validation on the de-duplicated dataset for the stress test.

This is the biggest finding of the project: after removing feature-vector
duplicates the models' MCC collapses toward chance.

Run from the repo root:

    python scripts/run_dedup_analysis.py

Outputs written to ``results/``:

- dedup_comparison.csv   : dirty vs clean CV metrics side-by-side.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_dataset, split_features_and_label, deduplicate
from src.features import add_url_complexity
from src.models import build_models, RANDOM_STATE
from src.evaluate import cross_validate_models


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    df = load_dataset()

    # ---- dirty: original dataset with duplicates ----
    X_dirty, y_dirty = split_features_and_label(df)
    X_dirty = add_url_complexity(X_dirty)
    X_tr_d, _, y_tr_d, _ = train_test_split(
        X_dirty, y_dirty, test_size=0.2, stratify=y_dirty, random_state=RANDOM_STATE
    )

    # ---- clean: drop feature-vector duplicates ----
    # Amer's feedback: dedupe on the feature vectors, not on the raw rows
    # including Domain, because the model sees features not URL strings.
    df_clean = deduplicate(df, on_features_only=True)
    print(f"Original rows: {len(df)}   De-duplicated rows: {len(df_clean)}")
    print(f"Class balance after de-dup:")
    print(df_clean["Label"].value_counts(normalize=True).round(3))

    X_clean, y_clean = split_features_and_label(df_clean)
    X_clean = add_url_complexity(X_clean)
    X_tr_c, _, y_tr_c, _ = train_test_split(
        X_clean, y_clean, test_size=0.2, stratify=y_clean, random_state=RANDOM_STATE
    )

    # ---- run both ----
    models = build_models()
    dirty_df = cross_validate_models(models, X_tr_d, y_tr_d)
    clean_df = cross_validate_models(models, X_tr_c, y_tr_c)

    combined = pd.DataFrame({
        "Model":            dirty_df["Model"],
        "Acc (dirty)":      dirty_df["Accuracy"].round(3),
        "Acc (clean)":      clean_df["Accuracy"].round(3),
        "F1 (dirty)":       dirty_df["F1"].round(3),
        "F1 (clean)":       clean_df["F1"].round(3),
        "MCC (dirty)":      dirty_df["MCC"].round(3),
        "MCC (clean)":      clean_df["MCC"].round(3),
    })
    combined.to_csv(results_dir / "dedup_comparison.csv", index=False)
    print("\nDedup comparison:")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
