"""
precompute_events.py
====================
Pre-aggregate tất cả heavy computations từ clean_fact_user_events (500 files, ~24GB)
bằng DuckDB một lần duy nhất → lưu vào data/aggregated/ dưới dạng parquet nhỏ.

Notebook diagnostic_analysis.ipynb sẽ chỉ load các file nhỏ này thay vì re-compute.

Chạy: python precompute_events.py
Thời gian ước tính: 5–15 phút tuỳ máy (chạy 1 lần duy nhất)
"""

import duckdb
import os
import sys
import time
from pathlib import Path

# ── Config ──
BASE       = Path("f:/PYTHON/Datathon2026-Deep67-Finals/data/processed")
EVENTS     = str(BASE / "clean_fact_user_events/*.parquet")
OUT_DIR    = Path("f:/PYTHON/Datathon2026-Deep67-Finals/data/aggregated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HARD_CONTACT_EVENTS = ("'view_phone','contact_chat','contact_zalo','contact_sms'")
PAGEVIEW_EVENT      = "pageview"

# ── DuckDB connection: tuning cho tốc độ ──
con = duckdb.connect()
# Dùng tối đa CPU cores và 4GB RAM buffer
con.execute("SET threads TO 8")
con.execute("SET memory_limit = '4GB'")
con.execute("SET temp_directory = 'f:/PYTHON/Datathon2026-Deep67-Finals/tmp'")
con.execute("SET max_temp_directory_size = '30GiB'")
con.execute("SET preserve_insertion_order = false")  # faster GROUP BY
Path("f:/PYTHON/Datathon2026-Deep67-Finals/tmp").mkdir(exist_ok=True)

def run(label, sql, out_name):
    """Chạy query và save kết quả ra parquet."""
    t0 = time.time()
    print(f"\n[{label}] Running...", flush=True)
    out_path = str(OUT_DIR / f"{out_name}.parquet")
    con.execute(f"COPY ({sql}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    elapsed = time.time() - t0
    n = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
    print(f"  -> {n:,} rows | {os.path.getsize(out_path)/1024:.0f} KB | {elapsed:.1f}s", flush=True)

# ─────────────────────────────────────────────
# A. events_by_type
# ─────────────────────────────────────────────
run("A", f"""
    SELECT event_type, COUNT(*) AS n_events
    FROM read_parquet('{EVENTS}')
    GROUP BY event_type
    ORDER BY n_events DESC
""", "events_by_type")

# ─────────────────────────────────────────────
# B. events_by_category_type
# ─────────────────────────────────────────────
run("B", f"""
    SELECT category, event_type, COUNT(*) AS n_events
    FROM read_parquet('{EVENTS}')
    WHERE category IS NOT NULL
    GROUP BY category, event_type
""", "events_by_category_type")

# ─────────────────────────────────────────────
# C. events_by_login_type
# ─────────────────────────────────────────────
run("C", f"""
    SELECT is_login, event_type, COUNT(*) AS n_events
    FROM read_parquet('{EVENTS}')
    GROUP BY is_login, event_type
""", "events_by_login_type")

# ─────────────────────────────────────────────
# D. events_by_hour_type
# ─────────────────────────────────────────────
run("D", f"""
    SELECT EXTRACT(HOUR FROM event_ts) AS hour, event_type, COUNT(*) AS n_events
    FROM read_parquet('{EVENTS}')
    WHERE event_ts IS NOT NULL
    GROUP BY hour, event_type
    ORDER BY hour
""", "events_by_hour_type")

# ─────────────────────────────────────────────
# E. events_by_dow_type
# ─────────────────────────────────────────────
run("E", f"""
    SELECT DAYOFWEEK(date) AS day_of_week, event_type, COUNT(*) AS n_events
    FROM read_parquet('{EVENTS}')
    WHERE date IS NOT NULL
    GROUP BY day_of_week, event_type
    ORDER BY day_of_week
""", "events_by_dow_type")

# ─────────────────────────────────────────────
# F. events_by_date_type
# ─────────────────────────────────────────────
run("F", f"""
    SELECT date, event_type, COUNT(*) AS n_events
    FROM read_parquet('{EVENTS}')
    WHERE date IS NOT NULL
    GROUP BY date, event_type
    ORDER BY date
""", "events_by_date_type")

# ─────────────────────────────────────────────
# G. events_by_city_type
# ─────────────────────────────────────────────
run("G", f"""
    SELECT city_name, event_type, COUNT(*) AS n_events
    FROM read_parquet('{EVENTS}')
    WHERE city_name IS NOT NULL
    GROUP BY city_name, event_type
""", "events_by_city_type")

# ─────────────────────────────────────────────
# H. events_by_device_type
# ─────────────────────────────────────────────
run("H", f"""
    SELECT device, event_type, COUNT(*) AS n_events
    FROM read_parquet('{EVENTS}')
    WHERE device IS NOT NULL
    GROUP BY device, event_type
""", "events_by_device_type")

# ─────────────────────────────────────────────
# I. session_depth
# ~149M unique sessions -> too large for full row-level output.
# Instead: save a DISTRIBUTION table (n_events bucket -> count/share)
# and key summary stats. This is what the notebook actually uses.
# ─────────────────────────────────────────────
run("I_summary", f"""
    WITH sess AS (
        SELECT
            session_id,
            COUNT(*) AS n_events,
            SUM(CASE WHEN event_type = '{PAGEVIEW_EVENT}' THEN 1 ELSE 0 END) AS n_pageviews,
            SUM(CASE WHEN event_type IN ({HARD_CONTACT_EVENTS}) THEN 1 ELSE 0 END) AS n_hard_contacts
        FROM read_parquet('{EVENTS}')
        WHERE session_id IS NOT NULL
        GROUP BY session_id
    )
    SELECT
        CASE
            WHEN n_events = 1  THEN '1'
            WHEN n_events <= 3 THEN '2-3'
            WHEN n_events <= 5 THEN '4-5'
            WHEN n_events <= 10 THEN '6-10'
            WHEN n_events <= 50 THEN '11-50'
            ELSE '50+'
        END AS depth_bucket,
        COUNT(*) AS n_sessions,
        SUM(CASE WHEN n_hard_contacts > 0 THEN 1 ELSE 0 END) AS sessions_with_contact,
        AVG(n_events) AS avg_events,
        AVG(n_pageviews) AS avg_pageviews,
        MEDIAN(n_events) AS median_events
    FROM sess
    GROUP BY depth_bucket
""", "session_depth_dist")

# Also save overall stats for funnel
run("I_stats", f"""
    SELECT
        COUNT(DISTINCT session_id) AS total_sessions,
        SUM(CASE WHEN is_contact = 1 THEN 1 ELSE 0 END) AS sessions_with_any_contact,
        AVG(CASE WHEN event_type = '{PAGEVIEW_EVENT}' THEN 1 ELSE 0 END) AS pv_rate
    FROM read_parquet('{EVENTS}')
    WHERE session_id IS NOT NULL
""", "session_stats")

# ─────────────────────────────────────────────
# J. dwell_time_sample (2% sample với filter)
# ─────────────────────────────────────────────
run("J", f"""
    SELECT item_id, category, event_type, dwell_time_sec, is_login, device, date
    FROM read_parquet('{EVENTS}')
    WHERE dwell_time_sec IS NOT NULL
      AND dwell_time_sec > 0
      AND dwell_time_sec <= 3600
      AND random() < 0.02
""", "dwell_time_sample")

# ─────────────────────────────────────────────
# K. surface_position_device
# ─────────────────────────────────────────────
run("K", f"""
    SELECT
        surface,
        CASE
            WHEN position = 1 THEN '1'
            WHEN position <= 3 THEN '2-3'
            WHEN position <= 5 THEN '4-5'
            WHEN position <= 10 THEN '6-10'
            WHEN position <= 20 THEN '11-20'
            WHEN position <= 50 THEN '21-50'
            WHEN position <= 100 THEN '51-100'
            ELSE '100+'
        END AS position_bucket,
        device,
        event_type,
        COUNT(*) AS n_events
    FROM read_parquet('{EVENTS}')
    WHERE surface IS NOT NULL AND position IS NOT NULL AND device IS NOT NULL
    GROUP BY surface, position_bucket, device, event_type
""", "surface_position_device")

con.close()
print("\n" + "="*60)
print("PRECOMPUTE COMPLETE")
print(f"Saved to: {OUT_DIR}")
files = sorted(OUT_DIR.glob("*.parquet"))
total_kb = sum(f.stat().st_size for f in files) / 1024
print(f"Files: {len(files)} | Total size: {total_kb:.0f} KB ({total_kb/1024:.1f} MB)")
print("="*60)
