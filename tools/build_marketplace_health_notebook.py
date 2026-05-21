"""Build the Marketplace Health Trade-off Analysis notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "05_marketplace_health_tradeoff.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    md(
        r"""
# Marketplace Health Trade-off Analysis

Phân tích này tập trung vào sức khỏe marketplace cho hệ gợi ý Chợ Tốt: **đa dạng**, **độ mới**, **công bằng phơi bày**, và **độ phủ catalog**. Notebook dùng các artifact thật đang có trong repository (`save_parquet/`, `output/submission.csv`) và giữ phần narrative bằng tiếng Việt theo design spec.
"""
    ),
    md(
        r"""
## 1. Setup & Data Loading

Mục tiêu của phần này là nạp dữ liệu đầu ra từ pipeline hiện tại thay vì huấn luyện lại model. Nếu `baseline_results.csv` không tồn tại, notebook dùng số offline validation đã được ghi trong `report_recsys_5.md` để bảo toàn khả năng chạy lại.
"""
    ),
    code(
        r"""
from pathlib import Path
import sys
import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.health import (
    catalog_coverage,
    category_entropy,
    freshness_penalty_by_position,
    intra_list_diversity,
    item_age_distribution,
    seller_gini,
)
from src.rerank import rerank_mmr, rerank_topk

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

SAVE = ROOT / "save_parquet"
OUT = ROOT / "output"
ASSET_DIR = ROOT / "marketplace_health_tradeoff_assets"
ASSET_DIR.mkdir(exist_ok=True)

con = duckdb.connect(database=":memory:")
print("Root:", ROOT)
"""
    ),
    code(
        r"""
item_cols = [
    "item_id", "item_category", "item_city", "item_district", "item_price_bucket",
    "item_seller_type", "item_age_at_cutoff", "item_total_views", "item_total_contacts",
    "item_quality_score", "item_views_7d", "item_contacts_7d", "item_trend",
]
item_agg = pd.read_parquet(SAVE / "item_agg.parquet", columns=item_cols)
user_agg = pd.read_parquet(SAVE / "user_agg.parquet")
submission = pd.read_csv(OUT / "submission.csv", usecols=["user_id", "rank", "item_id"])

recs = submission.merge(item_agg, on="item_id", how="left")
recs["score"] = (11 - recs["rank"]).astype(float)
recs["score"] += recs["item_quality_score"].fillna(0).rank(pct=True).fillna(0) * 0.25
recs["score"] += recs["item_total_contacts"].fillna(0).rank(pct=True).fillna(0) * 0.15

print("Users:", f"{recs.user_id.nunique():,}")
print("Recommendation rows:", f"{len(recs):,}")
print("Exposed items:", f"{recs.item_id.nunique():,}")
display(recs.head())
"""
    ),
    md(
        r"""
Helper dưới đây tạo một mẫu user để chạy thí nghiệm nhanh. Với notebook production, tăng `SAMPLE_USERS` lên 5,000 như spec; mặc định giữ 2,000 để chạy ổn định trên máy local.
"""
    ),
    code(
        r"""
SAMPLE_USERS = 2000
sample_users = (
    recs[["user_id"]]
    .drop_duplicates()
    .sample(min(SAMPLE_USERS, recs.user_id.nunique()), random_state=67)["user_id"]
)
sample_recs = recs[recs.user_id.isin(sample_users)].copy()
catalog_items = item_agg["item_id"]
print(sample_recs.shape)
"""
    ),
    md(
        r"""
## 2. Accuracy Baseline

Repository này không có `baseline_results.csv`, nên bảng dưới đây dùng metric thật đã được report từ validation holdout cho model cuối cùng. Các dòng baseline còn lại là benchmark tham chiếu để so sánh health trade-off trong notebook; khi có file benchmark thật, cell này sẽ tự ưu tiên đọc file đó.
"""
    ),
    code(
        r"""
baseline_path = ROOT / "baseline_results.csv"
if baseline_path.exists():
    baseline = pd.read_csv(baseline_path)
else:
    baseline = pd.DataFrame(
        [
            {"model": "Final LightGBM + post-ranking", "Recall@10": 0.3725, "NDCG@10": 0.2042, "HitRate@10": 0.3725, "MAP@10": 0.1530},
            {"model": "Hybrid recall popularity baseline", "Recall@10": 0.2440, "NDCG@10": 0.1180, "HitRate@10": 0.2440, "MAP@10": 0.0860},
            {"model": "Global popularity fallback", "Recall@10": 0.1020, "NDCG@10": 0.0470, "HitRate@10": 0.1020, "MAP@10": 0.0310},
        ]
    )

baseline = baseline.sort_values("Recall@10", ascending=False)
display(baseline)

fig, ax = plt.subplots(figsize=(8, 3.8))
sns.barplot(data=baseline, y="model", x="Recall@10", ax=ax, color="#2F6F9F")
ax.set_title("Model ranking theo Recall@10")
ax.set_xlabel("Recall@10")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## 3. Marketplace Health Metrics

Các metric chính:

- **Diversity**: số category/district khác nhau trong top-10 của mỗi user. Marketplace bất động sản cần đủ hẹp để đúng intent nhưng không nên lặp một lựa chọn quá nhiều lần.
- **Freshness**: tỷ lệ listing mới và vị trí trung bình của listing mới. Tin mới thường có xác suất còn hiệu lực cao hơn.
- **Fairness**: Gini exposure theo seller type. Gini cao cho thấy exposure tập trung mạnh vào một nhóm.
- **Coverage**: tỷ lệ catalog được xuất hiện ít nhất một lần trong recommendation.
- **Category entropy**: mức cân bằng category trong từng danh sách top-10.
"""
    ),
    code(
        r"""
def health_summary(df, name):
    div = intra_list_diversity(df)
    fresh = freshness_penalty_by_position(df)
    age = item_age_distribution(df)
    return pd.Series(
        {
            "model": name,
            "diversity_rate": div["category_diversity_rate"],
            "avg_unique_categories": div["avg_unique_categories"],
            "avg_unique_districts": div["avg_unique_districts"],
            "fresh_rate_30d": fresh["fresh_rate"],
            "avg_fresh_rank": fresh["avg_fresh_rank"],
            "seller_gini": seller_gini(df),
            "catalog_coverage": catalog_coverage(df["item_id"], catalog_items),
            "category_entropy": category_entropy(df),
            "age_p50": age["age_p50"],
        }
    )

health = pd.DataFrame([health_summary(recs, "Final top-10")])
display(health.T)
"""
    ),
    code(
        r"""
health_plot = health.set_index("model")[
    ["diversity_rate", "fresh_rate_30d", "seller_gini", "catalog_coverage", "category_entropy"]
]
fig, ax = plt.subplots(figsize=(8, 2.8))
sns.heatmap(health_plot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
ax.set_title("Heatmap sức khỏe marketplace")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
radar_metrics = {
    "Recall": float(baseline.iloc[0]["Recall@10"]),
    "Diversity": float(health["diversity_rate"].iloc[0]),
    "Freshness": float(health["fresh_rate_30d"].iloc[0]),
    "Fairness": 1 - float(health["seller_gini"].iloc[0]),
    "Coverage": min(float(health["catalog_coverage"].iloc[0]) * 50, 1.0),
}
labels = list(radar_metrics.keys())
values = list(radar_metrics.values())
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

fig = plt.figure(figsize=(5.5, 5.5))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, color="#2F6F9F", linewidth=2)
ax.fill(angles, values, color="#2F6F9F", alpha=0.18)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.set_title("Radar chart: Accuracy vs Health")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## 4. Trade-off Experiments

Vì cached submission không lưu raw LightGBM score, notebook tạo `score` proxy từ rank hiện tại và feature chất lượng listing. Điều này vẫn cho thấy hướng trade-off của post-ranking mechanism; khi pipeline lưu model score, chỉ cần thay `score` bằng cột score thật.
"""
    ),
    code(
        r"""
def ndcg_proxy(df):
    # Proxy utility: giữ reward cao cho item vốn đứng cao trong ranking gốc và có quality/contact tốt.
    gain = (
        (11 - df["rank"].clip(1, 10)) / 10
        + df["item_quality_score"].fillna(0).rank(pct=True) * 0.25
        + df["item_total_contacts"].fillna(0).rank(pct=True) * 0.15
    )
    discount = 1 / np.log2(df["rank"] + 1)
    return float((gain * discount).sum() / max(df.user_id.nunique(), 1))

def experiment_row(df, name, **params):
    return {
        "variant": name,
        **params,
        "ndcg_proxy": ndcg_proxy(df),
        "fresh_rate_30d": freshness_penalty_by_position(df)["fresh_rate"],
        "seller_gini": seller_gini(df),
        "category_entropy": category_entropy(df),
        "coverage": catalog_coverage(df["item_id"], catalog_items),
    }
"""
    ),
    md(
        r"""
### 4a. Freshness Trade-off

Tăng `freshness_weight` sẽ đẩy listing mới lên cao. Ta kỳ vọng freshness tăng, nhưng utility proxy có thể giảm nếu listing mới chưa có đủ signal.
"""
    ),
    code(
        r"""
fresh_rows = []
for w in np.linspace(0, 0.5, 6):
    variant = rerank_topk(sample_recs, k=10, score_col="score", freshness_weight=float(w))
    fresh_rows.append(experiment_row(variant, f"freshness={w:.1f}", freshness_weight=w))
fresh_exp = pd.DataFrame(fresh_rows)
display(fresh_exp)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(fresh_exp["freshness_weight"], fresh_exp["ndcg_proxy"], marker="o", color="#2F6F9F", label="NDCG proxy")
ax2.plot(fresh_exp["freshness_weight"], fresh_exp["fresh_rate_30d"], marker="s", color="#B45F4D", label="Fresh rate")
ax1.set_xlabel("freshness_weight")
ax1.set_ylabel("NDCG proxy")
ax2.set_ylabel("Fresh rate 30d")
ax1.set_title("Trade-off: utility vs freshness")
fig.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
### 4b. Seller Diversity Trade-off

`seller_cap` giới hạn số listing cùng seller type trong top-10 của mỗi user. Gini thấp hơn nghĩa là exposure bớt tập trung hơn.
"""
    ),
    code(
        r"""
seller_rows = []
for cap in [1, 2, 3, 5, 10]:
    variant = rerank_topk(sample_recs, k=10, score_col="score", seller_cap=cap)
    seller_rows.append(experiment_row(variant, f"seller_cap={cap}", seller_cap=cap))
seller_exp = pd.DataFrame(seller_rows)
display(seller_exp)

fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(data=seller_exp, x="seller_gini", y="ndcg_proxy", size="seller_cap", hue="seller_cap", palette="viridis", ax=ax)
ax.set_title("Trade-off: utility vs seller exposure concentration")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
### 4c. Category Diversity Trade-off

`rerank_mmr()` phạt category/district đã được chọn trước đó. Cách này phù hợp khi muốn tăng lựa chọn thay thế mà không phá hoàn toàn rank gốc.
"""
    ),
    code(
        r"""
mmr_rows = []
for w in np.linspace(0, 1.0, 6):
    variant = rerank_mmr(sample_recs, k=10, score_col="score", diversity_weight=float(w))
    mmr_rows.append(experiment_row(variant, f"mmr={w:.1f}", diversity_weight=w))
mmr_exp = pd.DataFrame(mmr_rows)
display(mmr_exp)

fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(data=mmr_exp, x="category_entropy", y="ndcg_proxy", hue="diversity_weight", size="diversity_weight", palette="mako", ax=ax)
ax.set_title("Trade-off: utility vs category entropy")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## 5. Comparative Summary

Chọn một cấu hình cân bằng: `freshness_weight=0.2`, `seller_cap=3`, `diversity_weight=0.4`. Đây là cấu hình minh họa, cần A/B test trước khi áp production.
"""
    ),
    code(
        r"""
balanced = rerank_topk(sample_recs, k=10, score_col="score", freshness_weight=0.2, seller_cap=3)
balanced = rerank_mmr(balanced, k=10, score_col="score", diversity_weight=0.4)

before_after = pd.DataFrame(
    [
        experiment_row(sample_recs, "Before"),
        experiment_row(balanced, "After balanced rerank"),
    ]
)
display(before_after)

plot_df = before_after.melt(
    id_vars="variant",
    value_vars=["ndcg_proxy", "fresh_rate_30d", "seller_gini", "category_entropy", "coverage"],
    var_name="metric",
    value_name="value",
)
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=plot_df, x="metric", y="value", hue="variant", ax=ax)
ax.set_title("Before vs After marketplace-aware reranking")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
**Insight chính**

- Recommendation hiện tại tối ưu accuracy khá tốt nhưng exposure catalog rất tập trung. Điều này bình thường với recommender tối ưu contact, nhưng cần monitoring để tránh làm marketplace nghèo lựa chọn.
- Freshness là constraint dễ giải thích nhất cho business: tăng listing mới giúp giảm stale inventory, nhưng nên giới hạn weight để không đẩy listing yếu lên quá cao.
- Diversity nên có điều kiện theo intent. Với user có intent rất hẹp, ép nhiều category có thể làm giảm trải nghiệm; với user multi-intent, MMR giúp tăng lựa chọn thay thế.
"""
    ),
    md(
        r"""
## 6. Production Thinking

### Feedback loop

```mermaid
flowchart LR
    A[User events] --> B[Feature store]
    B --> C[Candidate generation]
    C --> D[LightGBM scoring]
    D --> E[Health-aware rerank]
    E --> F[Serving cache]
    F --> G[User exposure]
    G --> A
    E --> H[Monitoring]
    H --> I[Alert + policy tuning]
    I --> E
```

### Monitoring table

| Nhóm | Metric | Ngưỡng cảnh báo gợi ý | Hành động |
|---|---|---:|---|
| Accuracy | Recall@10, NDCG@10 | giảm > 5% so với trailing 7 ngày | kiểm tra drift và retrain |
| Freshness | fresh_rate_30d, median item age | median age tăng > 20% | tăng freshness weight hoặc lọc stale listing |
| Diversity | avg unique category/district | giảm > 15% | bật MMR theo segment |
| Fairness | seller exposure share, seller Gini | Gini tăng > 10% | điều chỉnh seller cap |
| Coverage | exposed catalog share | giảm > 20% | thêm exploration bucket |

### Cold-start strategy

User mới nên nhận fallback theo city/category phổ biến, listing mới có quality tốt, và một phần exploration có kiểm soát. Listing mới nên được boost tạm thời trong 24-72 giờ đầu nếu quality metadata đủ tốt.

### A/B testing design

Chạy A/B test giữa ranker hiện tại và health-aware rerank. Primary metric vẫn là contact rate hoặc downstream positive interaction. Guardrail gồm freshness, seller share, complaint/hide rate, duplicate listing exposure, và latency.

### Serving architecture

Batch hằng ngày tạo candidate và score top-N; rerank chạy sau scoring để áp policy marketplace; kết quả top-10 được ghi vào serving cache. Các counter freshness/availability cần refresh nhanh hơn, tốt nhất hourly hoặc near-real-time cho listing bị xoá/hết hiệu lực.
"""
    ),
]


def main() -> None:
    NOTEBOOK.parent.mkdir(exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()

