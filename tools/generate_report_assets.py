from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_recsys_5_assets"
OUT.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
    }
)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()


def draw_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    steps = [
        "Cleaned parquet\ninputs",
        "Leakage-safe\naggregates",
        "Hybrid recall\ncandidates",
        "Pairwise feature\njoin",
        "LambdaRank\nscoring",
        "Fairness +\ndiversity rules",
        "Top-10\nsubmission",
    ]
    x = np.linspace(0.05, 0.88, len(steps))
    colors = ["#244C5A", "#6A8D73", "#D39C45", "#7F6A93", "#2F6F9F", "#B45F4D", "#3E4C59"]
    for i, (xi, label) in enumerate(zip(x, steps)):
        ax.add_patch(
            plt.Rectangle(
                (xi, 0.38),
                0.11,
                0.28,
                transform=ax.transAxes,
                facecolor=colors[i],
                edgecolor="#222222",
                linewidth=0.8,
            )
        )
        ax.text(xi + 0.055, 0.52, label, ha="center", va="center", color="white", transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x[i + 1] - 0.005, 0.52),
                xytext=(xi + 0.115, 0.52),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"},
                xycoords=ax.transAxes,
            )
    ax.set_title("End-to-end recommender pipeline")
    savefig("fig01_pipeline.png")


def draw_timeline() -> None:
    dates = pd.to_datetime(["2026-02-08", "2026-03-09", "2026-03-10", "2026-04-09", "2026-04-10", "2026-05-07"])
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.axvspan(dates[0], dates[1], ymin=0.25, ymax=0.75, color="#6A8D73", alpha=0.85, label="training labels")
    ax.axvspan(dates[2], dates[3], ymin=0.25, ymax=0.75, color="#D39C45", alpha=0.85, label="validation labels")
    ax.axvspan(dates[4], dates[5], ymin=0.25, ymax=0.75, color="#2F6F9F", alpha=0.85, label="test target window")
    ax.axvline(dates[1], color="#222222", linestyle="--", linewidth=1.2)
    ax.text(dates[1], 0.84, "feature cutoff\n2026-03-09", ha="center", va="bottom")
    ax.axvline(dates[3], color="#222222", linestyle=":", linewidth=1.2)
    ax.text(dates[3], 0.84, "observed data limit\n2026-04-09", ha="center", va="bottom")
    ax.set_title("Time-based split used for leakage-safe validation")
    ax.legend(loc="lower center", ncol=3, frameon=True)
    savefig("fig02_time_split.png")


def main() -> None:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    draw_pipeline()
    draw_timeline()

    user_agg = pd.read_parquet(ROOT / "save_parquet" / "user_agg.parquet")
    item_cols = [
        "item_id",
        "item_category",
        "item_seller_type",
        "item_age_at_cutoff",
        "item_total_views",
        "item_total_contacts",
        "item_quality_score",
    ]
    item_agg = pd.read_parquet(ROOT / "save_parquet" / "item_agg.parquet", columns=item_cols)

    cand_counts = con.execute(
        f"""
        SELECT COUNT(*) AS candidates
        FROM read_parquet('{(ROOT / "save_parquet" / "candidates.parquet").as_posix()}')
        GROUP BY user_id
        """
    ).df()["candidates"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(cand_counts.clip(upper=600), bins=45, color="#2F6F9F", alpha=0.85)
    ax.axvline(cand_counts.median(), color="#B45F4D", linestyle="--", label=f"median = {cand_counts.median():.0f}")
    ax.set_xlabel("Candidates per user (clipped at 600)")
    ax.set_ylabel("User count")
    ax.set_title("Candidate pool size after recall expansion")
    ax.legend()
    savefig("fig03_candidate_pool_distribution.png")

    bins = [0, 5, 20, 100, np.inf]
    labels = ["cold (0-4)", "light (5-19)", "active (20-99)", "power (100+)"]
    seg = pd.cut(user_agg["user_total_views"].fillna(0), bins=bins, labels=labels, right=False)
    seg_counts = seg.value_counts().reindex(labels)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(seg_counts.index.astype(str), seg_counts.values, color=["#244C5A", "#6A8D73", "#D39C45", "#B45F4D"])
    ax.set_ylabel("Users")
    ax.set_title("Target users by historical activity level")
    ax.tick_params(axis="x", rotation=15)
    savefig("fig04_user_activity_segments.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    q = item_agg["item_quality_score"].dropna().clip(upper=item_agg["item_quality_score"].quantile(0.99))
    ax.hist(q, bins=50, color="#6A8D73", alpha=0.9)
    ax.set_xlabel("Item quality score (99th percentile clipped)")
    ax.set_ylabel("Listings")
    ax.set_title("Distribution of listing quality signal")
    savefig("fig05_item_quality_distribution.png")

    metrics = pd.DataFrame(
        {
            "metric": ["Recall@10", "NDCG@10", "Candidate recall"],
            "value": [0.3725, 0.2042, 0.0008],
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.3))
    bars = ax.bar(metrics["metric"], metrics["value"], color=["#2F6F9F", "#6A8D73", "#B45F4D"])
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("Score")
    ax.set_title("Offline validation metrics")
    for bar, value in zip(bars, metrics["value"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center")
    savefig("fig06_validation_metrics.png")

    sub_path = ROOT / "output" / "submission.csv"
    recs = pd.read_csv(sub_path, usecols=["user_id", "rank", "item_id"])
    recs = recs.merge(item_agg, on="item_id", how="left")

    seller_share = recs["item_seller_type"].fillna("unknown").value_counts(normalize=True).head(8).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(seller_share.index.astype(str), seller_share.values * 100, color="#7F6A93")
    ax.set_xlabel("Share of top-10 exposure (%)")
    ax.set_title("Exposure share by seller type in final recommendations")
    savefig("fig07_seller_type_exposure.png")

    diversity = recs.groupby("user_id")["item_category"].nunique(dropna=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(diversity, bins=np.arange(1, 13) - 0.5, color="#D39C45", alpha=0.9)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Distinct categories in top-10")
    ax.set_ylabel("Users")
    ax.set_title("Category diversity of final top-10 lists")
    savefig("fig08_category_diversity.png")

    exposure = recs["item_id"].value_counts()
    ranked = np.sort(exposure.values)
    cum_exposure = np.cumsum(ranked) / ranked.sum()
    cum_items = np.arange(1, len(ranked) + 1) / len(ranked)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot(cum_items, cum_exposure, color="#244C5A", linewidth=2)
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("Cumulative share of exposed items")
    ax.set_ylabel("Cumulative share of recommendation exposure")
    ax.set_title("Long-tail exposure Lorenz curve")
    savefig("fig09_lorenz_exposure.png")

    rec_age = recs["item_age_at_cutoff"].dropna().clip(upper=120)
    all_age = item_agg["item_age_at_cutoff"].dropna().clip(upper=120)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(all_age, bins=40, density=True, alpha=0.45, color="#999999", label="all listings")
    ax.hist(rec_age, bins=40, density=True, alpha=0.75, color="#2F6F9F", label="recommended listings")
    ax.set_xlabel("Listing age at feature cutoff, days (clipped at 120)")
    ax.set_ylabel("Density")
    ax.set_title("Freshness profile of recommended listings")
    ax.legend()
    savefig("fig10_freshness_profile.png")

    summary = {
        "target_users": int(pd.read_parquet(ROOT / "save_parquet" / "test_users.parquet").shape[0]),
        "user_agg_rows": int(user_agg.shape[0]),
        "item_agg_rows": int(item_agg.shape[0]),
        "candidate_rows": int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{(ROOT / 'save_parquet' / 'candidates.parquet').as_posix()}')").fetchone()[0]),
        "candidate_median_per_user": float(cand_counts.median()),
        "candidate_mean_per_user": float(cand_counts.mean()),
        "submission_rows": int(recs.shape[0]),
        "submission_users": int(recs["user_id"].nunique()),
        "submission_items_exposed": int(recs["item_id"].nunique()),
        "avg_distinct_categories_top10": float(diversity.mean()),
        "median_distinct_categories_top10": float(diversity.median()),
        "avg_recommended_item_age": float(rec_age.mean()),
        "avg_all_item_age": float(all_age.mean()),
    }
    pd.Series(summary, name="value").to_csv(OUT / "summary_metrics.csv")
    print(f"Wrote report assets to {OUT}")


if __name__ == "__main__":
    main()
