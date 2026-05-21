"""Post-ranking utilities for marketplace-aware recommendation lists."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def rerank_mmr(
    df: pd.DataFrame,
    k: int = 10,
    score_col: str = "score",
    diversity_weight: float = 0.2,
    category_col: str = "item_category",
    district_col: str = "item_district",
) -> pd.DataFrame:
    """MMR-style rerank that discourages repeated category and district exposure."""
    if df.empty:
        return df.copy()
    required = {"user_id", "item_id", score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    outputs = []
    for user_id, group in df.groupby("user_id", sort=False):
        pool = group.sort_values(score_col, ascending=False).copy()
        selected = []
        cat_counts: defaultdict[object, int] = defaultdict(int)
        district_counts: defaultdict[object, int] = defaultdict(int)

        while len(selected) < k and not pool.empty:
            scored = pool.copy()
            cat_penalty = scored.get(category_col, pd.Series(index=scored.index, dtype=object)).map(cat_counts).fillna(0)
            district_penalty = scored.get(district_col, pd.Series(index=scored.index, dtype=object)).map(district_counts).fillna(0)
            scored["_mmr_score"] = scored[score_col].astype(float) - diversity_weight * (cat_penalty + 0.5 * district_penalty)
            pick_idx = scored["_mmr_score"].idxmax()
            picked = pool.loc[pick_idx].copy()
            selected.append(picked)
            cat_counts[picked.get(category_col)] += 1
            district_counts[picked.get(district_col)] += 1
            pool = pool.drop(index=pick_idx)

        if selected:
            user_out = pd.DataFrame(selected)
            user_out["rank"] = np.arange(1, len(user_out) + 1)
            user_out["user_id"] = user_id
            outputs.append(user_out)

    return pd.concat(outputs, ignore_index=True) if outputs else df.iloc[0:0].copy()


def rerank_topk(
    df: pd.DataFrame,
    k: int = 10,
    score_col: str = "score",
    freshness_weight: float = 0.0,
    seller_cap: int | None = None,
    age_col: str = "item_age_at_cutoff",
    seller_col: str = "item_seller_type",
) -> pd.DataFrame:
    """Rerank with optional freshness boost and per-user seller exposure cap."""
    if df.empty:
        return df.copy()
    recs = df.copy()
    age = pd.to_numeric(recs.get(age_col, 0), errors="coerce").fillna(999.0)
    recs["_adjusted_score"] = recs[score_col].astype(float) + freshness_weight * np.exp(-age / 30.0)

    outputs = []
    for user_id, group in recs.groupby("user_id", sort=False):
        ranked = group.sort_values("_adjusted_score", ascending=False)
        if seller_cap is not None:
            selected = []
            seller_counts: defaultdict[object, int] = defaultdict(int)
            leftovers = []
            for _, row in ranked.iterrows():
                seller = row.get(seller_col, "unknown")
                if seller_counts[seller] < seller_cap:
                    selected.append(row)
                    seller_counts[seller] += 1
                else:
                    leftovers.append(row)
                if len(selected) >= k:
                    break
            if len(selected) < k and leftovers:
                selected.extend(leftovers[: k - len(selected)])
            out = pd.DataFrame(selected)
        else:
            out = ranked.head(k).copy()
        out["rank"] = np.arange(1, len(out) + 1)
        out["user_id"] = user_id
        outputs.append(out.drop(columns=["_adjusted_score"], errors="ignore"))

    return pd.concat(outputs, ignore_index=True) if outputs else recs.iloc[0:0].copy()

