"""Reproduces the main cross-validated analysis and saves results/ files.

Run from the repo root:

    python scripts/run_analysis.py

Outputs written to ``results/``:

- cv_results.csv         : 5-fold CV metrics for every model.
- feature_importance.csv : permutation feature importance from the best model.
- duplicate_summary.txt  : the two kinds of duplicate counts.
- plots/spearman.png     : feature correlation heatmap.
- plots/pca.png          : 2D PCA scatter, coloured by label.
- plots/feature_importance.png : permutation importance bar chart.
"""
from __future__ import annotations
import sys
from pathlib import Path

# so `import src.*` works when running from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.data import load_dataset, split_features_and_label, duplicate_summary
from src.features import add_url_complexity
from src.models import build_models, RANDOM_STATE
from src.evaluate import cross_validate_models
from src.plots import (
    save_correlation_heatmap,
    save_pca_scatter,
    save_feature_importance,
)


def main() -> None:
    results_dir = ROOT / "results"
    plots_dir = results_dir / "plots"
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # ---- data ----
    df = load_dataset()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # ---- duplicate analysis (Amer's feedback point) ----
    summary = duplicate_summary(df)
    (results_dir / "duplicate_summary.txt").write_text(
        "Duplicate analysis\n"
        "==================\n"
        f"Total rows in raw dataset:       {summary['total_rows']}\n\n"
        "Full-row duplicates (identical across ALL columns, including Domain):\n"
        f"  duplicated rows:      {summary['full_row_duplicates']}\n"
        f"  unique rows:          {summary['unique_full_rows']}\n\n"
        "Feature-vector duplicates (identical AFTER dropping Domain column):\n"
        f"  duplicated rows:      {summary['feature_vector_duplicates']}\n"
        f"  unique feature vectors: {summary['unique_feature_vectors']}\n\n"
        "The feature-vector count is what matters for cross-validation, because\n"
        "the model sees the numerical features not the Domain string. Different\n"
        "URLs mapping to the same feature vector still leak information across\n"
        "train/test folds.\n"
    )
    print(f"Full-row duplicates:       {summary['full_row_duplicates']}")
    print(f"Feature-vector duplicates: {summary['feature_vector_duplicates']}")

    # ---- features ----
    X, y = split_features_and_label(df)
    X = add_url_complexity(X)

    # ---- correlation and PCA plots ----
    save_correlation_heatmap(X, plots_dir / "spearman.png")
    save_pca_scatter(X, y, plots_dir / "pca.png")

    # ---- train/test split for held-out evaluation ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ---- cross-validation ----
    models = build_models()
    cv_df = cross_validate_models(models, X_tr, y_tr)
    cv_df.to_csv(results_dir / "cv_results.csv", index=False)
    print("\nCross-validation results:")
    print(cv_df.round(3).to_string(index=False))

    # ---- best model: fit on full training, do permutation importance ----
    best_name = cv_df.sort_values("F2", ascending=False).iloc[0]["Model"]
    print(f"\nBest model by F2: {best_name}")

    best_model = models[best_name]
    best_model.fit(X_tr, y_tr)
    result = permutation_importance(
        best_model, X_te, y_te,
        n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1, scoring="f1",
    )
    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(results_dir / "feature_importance.csv", index=False)
    save_feature_importance(imp_df, plots_dir / "feature_importance.png",
                            title=f"Permutation importance ({best_name})")


if __name__ == "__main__":
    main()
