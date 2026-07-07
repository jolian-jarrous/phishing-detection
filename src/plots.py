"""Plotting helpers. Kept thin on purpose --- matplotlib boilerplate belongs
here, not in the notebook."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_correlation_heatmap(X: pd.DataFrame, out_path: Path | str) -> None:
    corr = X.corr(method="spearman")
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        annot_kws={"size": 7}, cbar_kws={"shrink": 0.7},
    )
    plt.title("Spearman correlation between features")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_pca_scatter(X, y, out_path: Path | str) -> None:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], alpha=0.3, s=8, label="legit")
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], alpha=0.3, s=8, label="phish")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("First two PCA components, coloured by label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_feature_importance(imp_df: pd.DataFrame, out_path: Path | str,
                            title: str = "Permutation importance") -> None:
    plt.figure(figsize=(9, 6))
    sns.barplot(data=imp_df, y="feature", x="importance_mean", color="steelblue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
