# Datathon2026-Deep67-Finals

## Project overview
This repository contains a full recommender system pipeline for the Datathon 2026 finals. The pipeline ingests cleaned parquet datasets, builds user and item aggregates, generates recall candidates from multiple channels, trains a learning-to-rank model with time-based validation, and exports a submission file in the required format.

The main implementation lives in the notebook [recsys_5_fixed.ipynb](notebooks/recsys_5_fixed.ipynb). It follows a leakage-safe workflow (all features computed up to a feature cutoff date) and uses a multi-stage ranking stack optimized for recall and final ranking quality.

## Repository structure
- [notebooks/recsys_5_fixed.ipynb](notebooks/recsys_5_fixed.ipynb): End-to-end pipeline (feature engineering, candidate generation, training, scoring, submission).
- [notebooks/eda_datathon.ipynb](notebooks/eda_datathon.ipynb): Exploratory analysis notebook.
- [notebooks/recsys_5_market_health_compare.ipynb](notebooks/recsys_5_market_health_compare.ipynb): Market-health comparison notebook.
- [notebooks/recsys_5_market_health_compare_2.ipynb](notebooks/recsys_5_market_health_compare_2.ipynb): Alternate market-health comparison notebook.
- [notebooks/05_marketplace_health_tradeoff.ipynb](notebooks/05_marketplace_health_tradeoff.ipynb): Marketplace health trade-off notebook.
- datathon_2026_processed/: Cleaned parquet inputs and test users.
- marketplace_health_tradeoff_assets/: Assets generated for the marketplace health tradeoff notebook.
- output/: Generated `submission.csv` and provided `sample_submission.csv`.
- report_recsys_5_assets/: Report artifacts and summary metrics.
- save_parquet/: Optional cached parquet intermediates and DuckDB temp files.
- src/: Shared Python helpers for health and reranking logic.
- tools/: Notebook and report generation utilities.

### Folder structure map
```
Datathon2026-Deep67-Finals/
├─ README.md
├─ datathon_2026_processed/
│  ├─ train_clean/
│  │  ├─ dim_listing/
│  │  │  └─ dim_listing_cleaned.parquet
│  │  ├─ fact_listing_snapshot/
│  │  │  └─ fact_listing_snapshot_cleaned.parquet
│  │  ├─ fact_user_ad_interactions/
│  │  │  └─ *.parquet
│  │  └─ fact_user_events/
│  │     └─ **/*.parquet
│  └─ test/
│     └─ test_users.parquet
├─ marketplace_health_tradeoff_assets/
├─ notebooks/
│  ├─ 05_marketplace_health_tradeoff.ipynb
│  ├─ eda_datathon.ipynb
│  ├─ recsys_5_fixed.ipynb
│  ├─ recsys_5_market_health_compare.ipynb
│  ├─ recsys_5_market_health_compare_2.ipynb
│  └─ outputs/
├─ output/
│  ├─ sample_submission.csv
│  └─ submission.csv
├─ report_recsys_5_assets/
└─ save_parquet/
	└─ duckdb_temp/
├─ src/
│  ├─ __init__.py
│  ├─ health.py
│  └─ rerank.py
└─ tools/
	├─ build_marketplace_health_notebook.py
	└─ generate_report_assets.py
```

### Pipeline graph (data flow and file locations)
```
datathon_2026_processed/
	├─ train_clean/dim_listing/*.parquet -----------┐
	├─ train_clean/fact_listing_snapshot/*.parquet -┼─> [recsys_5_fixed.ipynb]
	├─ train_clean/fact_user_ad_interactions/*.parquet
	├─ train_clean/fact_user_events/**/*.parquet ----┘
	└─ test/test_users.parquet ----------------------┐
																									 │
	[recsys_5_fixed.ipynb] builds:                   │
		- user aggregates                              │
		- item aggregates                              │
		- user preferences                             │
		- recall candidates (pop, history, i2i, ALS)   │
		- pair features + hard negatives               │
		- LightGBM ranker                              │
		- test scoring + post-ranking                  │
																									 │
	writes outputs:                                  │
		- save_parquet/*.parquet (optional cache) <----┘
		- output/submission.csv (final)
```

## Data inputs
The notebook expects the cleaned dataset layout under `datathon_2026_processed/`:
- `train_clean/dim_listing/dim_listing_cleaned.parquet`
- `train_clean/fact_listing_snapshot/fact_listing_snapshot_cleaned.parquet`
- `train_clean/fact_user_ad_interactions/*.parquet`
- `train_clean/fact_user_events/**/**/*.parquet`
- `test/test_users.parquet`

Paths are configured at the top of the notebook (Windows absolute paths are currently used). If you move the dataset, update `BASE_DATA_DIR` in the first code cell of [notebooks/recsys_5_fixed.ipynb](notebooks/recsys_5_fixed.ipynb).

## End-to-end pipeline (full workflow)
1. **Environment and configuration**
	- Define time windows: feature cutoff, training window, validation window, and prediction window.
	- Configure DuckDB settings and optional caching paths.
	- Register parquet datasets as DuckDB views.

2. **Target user pool**
	- Build the target pool by combining all test users with a sampled set of training users.
	- All downstream feature aggregation is restricted to this user pool.

3. **User aggregates (up to feature cutoff)**
	- Total views, positive events, positive rate, average dwell time, last active days, login type.
	- Optional ad-interaction aggregates (ad views, ad leads).

4. **Item aggregates (up to feature cutoff)**
	- Cumulative views and contacts.
	- 7-day trends and recent contact rates.
	- Listing metadata (category, city, district, price bucket, beds, baths, area, images).
	- Item age at cutoff and optional lead velocity feature (if contact interactions data exists).

5. **User preference profiling**
	- Determine each user's top category, preferred city, preferred district, and price bucket based on positive events and pageviews.

6. **Candidate generation (recall stage)**
	- Global popularity (top items overall).
	- Top items per category, city, district, and price bucket.
	- User history (recent pageviews).
	- Item-to-item co-occurrence within session windows.
	- ALS collaborative filtering (optional if `implicit` is installed).
	- Exclude previously positive interactions and deduplicate.
	- Pad to guarantee minimum candidates per user and apply a final recall boost to stabilize pool size.

7. **Pair feature engineering**
	- Join user, item, and user-item aggregates.
	- Build match features between user preferences and item attributes.
	- Add freshness and recency weights.

8. **Training and validation (time-based split)**
	- Train positives: events within training window.
	- Validation positives: events within validation window.
	- Construct hard negatives from viewed-but-not-positive items.
	- Downsample soft negatives and cap per-user counts for memory safety.
	- Train LightGBM LambdaRank and evaluate Recall@10 and NDCG@10.

9. **Scoring and post-processing**
	- Score candidates for test users in chunks to control memory.
	- Apply post-ranking rules: freshness boost, category diversity penalty, and private-agent fairness.
	- Apply cold-user fallback with global or city-level top items.

10. **Submission export**
	- Align with `output/sample_submission.csv`.
	- Save final `output/submission.csv` with columns `ID`, `user_id`, `rank`, `item_id`.

## How to run
1. Open [notebooks/recsys_5_fixed.ipynb](notebooks/recsys_5_fixed.ipynb) in VS Code.
2. Update `BASE_DATA_DIR` if your dataset path differs.
3. Run cells top-to-bottom.
4. The final submission is written to `output/submission.csv`.

## Notes and recommendations
- Enable caching by setting `USE_CACHE = True` after the first full run.
- ALS candidate generation requires `implicit` (install with `pip install implicit`).
- LightGBM training requires `lightgbm` and `scikit-learn` (install with `pip install lightgbm scikit-learn`).
- For memory-constrained machines, keep the sampling caps and validation downsampling enabled.