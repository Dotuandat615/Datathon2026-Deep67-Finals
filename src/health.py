"""Marketplace health metrics for top-k recommendation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _topk(df: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    if k is None or "rank" not in df.columns:
        return df.copy()
    return df[df["rank"] <= k].copy()


def intra_list_diversity(df: pd.DataFrame, k: int | None = 10) -> pd.Series:
    """Average per-user uniqueness for category and district exposure."""
    recs = _topk(df, k)
    if recs.empty:
        return pd.Series(
            {
                "avg_unique_categories": 0.0,
                "avg_unique_districts": 0.0,
                "category_diversity_rate": 0.0,
                "district_diversity_rate": 0.0,
            }
        )

    denom = recs.groupby("user_id")["item_id"].count().replace(0, np.nan)
    cat = recs.groupby("user_id")["item_category"].nunique(dropna=True)
    district = recs.groupby("user_id")["item_district"].nunique(dropna=True)
    return pd.Series(
        {
            "avg_unique_categories": float(cat.mean()),
            "avg_unique_districts": float(district.mean()),
            "category_diversity_rate": float((cat / denom).mean()),
            "district_diversity_rate": float((district / denom).mean()),
        }
    )


def seller_gini(df: pd.DataFrame, seller_col: str = "item_seller_type") -> float:
    """Gini coefficient over seller or seller-type exposure counts."""
    if df.empty or seller_col not in df.columns:
        return 0.0
    counts = df[seller_col].fillna("unknown").value_counts().to_numpy(dtype=float)
    if counts.size == 0 or counts.sum() == 0:
        return 0.0
    counts = np.sort(counts)
    n = counts.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * counts) / (n * counts.sum())) - ((n + 1) / n))


def catalog_coverage(recommended_items, catalog_items) -> float:
    """Fraction of catalog items that receive at least one recommendation."""
    rec_set = set(pd.Series(recommended_items).dropna().astype(str))
    catalog_set = set(pd.Series(catalog_items).dropna().astype(str))
    if not catalog_set:
        return 0.0
    return float(len(rec_set & catalog_set) / len(catalog_set))


def item_age_distribution(df: pd.DataFrame, age_col: str = "item_age_at_cutoff") -> pd.Series:
    """Quantile summary for item age in recommendations."""
    if df.empty or age_col not in df.columns:
        return pd.Series({"age_p10": np.nan, "age_p25": np.nan, "age_p50": np.nan, "age_p75": np.nan, "age_p90": np.nan})
    age = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if age.empty:
        return pd.Series({"age_p10": np.nan, "age_p25": np.nan, "age_p50": np.nan, "age_p75": np.nan, "age_p90": np.nan})
    q = age.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return pd.Series(
        {
            "age_p10": float(q.loc[0.10]),
            "age_p25": float(q.loc[0.25]),
            "age_p50": float(q.loc[0.50]),
            "age_p75": float(q.loc[0.75]),
            "age_p90": float(q.loc[0.90]),
        }
    )


def freshness_penalty_by_position(
    df: pd.DataFrame,
    age_col: str = "item_age_at_cutoff",
    fresh_days: int = 30,
) -> pd.Series:
    """Position profile for fresh items; lower average rank is better."""
    if df.empty or age_col not in df.columns or "rank" not in df.columns:
        return pd.Series({"fresh_rate": 0.0, "avg_fresh_rank": np.nan, "fresh_top3_rate": 0.0})
    recs = df.copy()
    recs["_is_fresh"] = pd.to_numeric(recs[age_col], errors="coerce").le(fresh_days)
    fresh = recs[recs["_is_fresh"]]
    return pd.Series(
        {
            "fresh_rate": float(recs["_is_fresh"].mean()),
            "avg_fresh_rank": float(fresh["rank"].mean()) if not fresh.empty else np.nan,
            "fresh_top3_rate": float((fresh["rank"] <= 3).mean()) if not fresh.empty else 0.0,
        }
    )


def category_entropy(df: pd.DataFrame, category_col: str = "item_category") -> float:
    """Mean normalized entropy of categories within each user's recommendation list."""
    if df.empty or category_col not in df.columns:
        return 0.0

    def _entropy(values: pd.Series) -> float:
        p = values.dropna().value_counts(normalize=True)
        if len(p) <= 1:
            return 0.0
        return float(-(p * np.log(p)).sum() / np.log(len(p)))

    return float(df.groupby("user_id")[category_col].apply(_entropy).mean())

