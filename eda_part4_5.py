"""Phần 4-5: Hiệu suất tin đăng + Hành vi người dùng"""
import duckdb, matplotlib.pyplot as plt, matplotlib.ticker as mticker
import seaborn as sns, numpy as np, os, sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'figure.facecolor': '#0f0f1a', 'axes.facecolor': '#1a1a2e',
    'text.color': '#e0e0e0', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#b0b0b0', 'ytick.color': '#b0b0b0',
    'axes.edgecolor': '#333355', 'grid.color': '#2a2a4a', 'grid.alpha': 0.5,
})

OUTPUT_DIR = r"f:\PYTHON\Datathon2026-Deep67-Finals\outputs\eda"
BASE = r"f:/PYTHON/Datathon2026-Deep67-Finals/data/processed"
DIM = f"'{BASE}/dim_listing_cleaned/dim_listing_cleaned.parquet'"
SNAP = f"'{BASE}/fact_listing_snapshot_cleaned/fact_listing_snapshot_cleaned.parquet'"
EVENTS = f"'{BASE}/clean_fact_user_events/*.parquet'"
INTER = f"'{BASE}/fact_user_ad_interactions/*.parquet'"
CATEGORY_MAP = {1010:'Phòng trọ/Studio',1020:'Căn hộ/Chung cư',1030:'Nhà ở',1040:'Đất',1050:'Dự án mở bán'}
PALETTE = ['#6C63FF','#FF6584','#43E8D8','#FFD93D','#FF8C42','#A78BFA','#34D399','#F472B6','#60A5FA','#FBBF24']
con = duckdb.connect()

def save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png")); plt.close(fig); print(f"  ✓ {name}.png")
def fmt_num(x, _=None):
    if x >= 1e6: return f'{x/1e6:.1f}M'
    if x >= 1e3: return f'{x/1e3:.0f}K'
    return f'{x:.0f}'
def map_cat(df, col='category'):
    df[col] = df[col].map(CATEGORY_MAP).fillna(df[col].astype(str)); return df

# ============================================================
# PHẦN 4: HIỆU SUẤT TIN ĐĂNG (SNAPSHOT)
# ============================================================
print("=" * 50)
print("PHẦN 4: Hiệu suất tin đăng (Snapshot)")
print("=" * 50)

# 15. Phân bố views_24h (log scale)
df = con.sql(f"SELECT views_24h FROM {SNAP} WHERE views_24h > 0 AND views_24h <= 500 USING SAMPLE 500000").df()
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['views_24h'], bins=100, color=PALETTE[0], edgecolor='none', alpha=0.85)
ax.set_yscale('log')
ax.set_title('Phân bố lượt xem trong 24h (log scale)', fontweight='bold', pad=15)
ax.set_xlabel('Views 24h'); ax.set_ylabel('Tần suất (log)')
ax.axvline(df['views_24h'].median(), color='#FFD93D', ls='--', lw=2, label=f"Median: {df['views_24h'].median():.0f}")
ax.legend(facecolor='#1a1a2e', edgecolor='#333355')
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '15_views_24h_distribution')

# 16. Phân bố contacts_24h
df = con.sql(f"SELECT contacts_24h FROM {SNAP} WHERE contacts_24h >= 0 AND contacts_24h <= 20 USING SAMPLE 500000").df()
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['contacts_24h'], bins=21, color=PALETTE[1], edgecolor='none', alpha=0.85)
ax.set_title('Phân bố số liên hệ trong 24h', fontweight='bold', pad=15)
ax.set_xlabel('Contacts 24h'); ax.set_ylabel('Tần suất')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '16_contacts_24h_distribution')

# 17. CTR theo tuổi tin đăng
df = con.sql(f"""
    SELECT CAST(listing_age_days AS INT) as age, AVG(ctr_24h) as avg_ctr
    FROM {SNAP} WHERE listing_age_days BETWEEN 0 AND 120
    GROUP BY CAST(listing_age_days AS INT) ORDER BY age
""").df()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['age'], df['avg_ctr'], color=PALETTE[2], lw=2)
ax.fill_between(df['age'], df['avg_ctr'], alpha=0.15, color=PALETTE[2])
ax.set_title('CTR trung bình theo tuổi tin đăng', fontweight='bold', pad=15)
ax.set_xlabel('Tuổi tin đăng (ngày)'); ax.set_ylabel('CTR trung bình')
ax.grid(alpha=0.3); ax.set_axisbelow(True)
save(fig, '17_ctr_by_listing_age')

# 18. Views theo ngày trong tuần
df = con.sql(f"""
    SELECT day_of_week, AVG(views_24h) as avg_views, AVG(contacts_24h) as avg_contacts
    FROM {SNAP} GROUP BY day_of_week ORDER BY day_of_week
""").df()
days_vi = ['Thứ 2','Thứ 3','Thứ 4','Thứ 5','Thứ 6','Thứ 7','CN']
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(days_vi, df['avg_views'], color=PALETTE[0], edgecolor='none', width=0.5, label='Avg Views', alpha=0.8)
ax2 = ax1.twinx()
ax2.plot(days_vi, df['avg_contacts'], color=PALETTE[1], marker='o', lw=2.5, label='Avg Contacts')
ax1.set_title('Lượt xem & liên hệ trung bình theo ngày trong tuần', fontweight='bold', pad=15)
ax1.set_ylabel('Avg Views 24h', color=PALETTE[0])
ax2.set_ylabel('Avg Contacts 24h', color=PALETTE[1])
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper right', facecolor='#1a1a2e', edgecolor='#333355')
ax1.grid(axis='y', alpha=0.3); ax1.set_axisbelow(True)
save(fig, '18_views_contacts_by_weekday')

# 19. Xu hướng views theo tháng
df = con.sql(f"""
    SELECT DATE_TRUNC('week', date) as week, SUM(views_24h) as total_views, SUM(contacts_24h) as total_contacts
    FROM {SNAP} GROUP BY week ORDER BY week
""").df()
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.fill_between(df['week'], df['total_views'], alpha=0.3, color=PALETTE[0])
ax1.plot(df['week'], df['total_views'], color=PALETTE[0], lw=2, label='Total Views')
ax2 = ax1.twinx()
ax2.plot(df['week'], df['total_contacts'], color=PALETTE[1], lw=2, label='Total Contacts')
ax1.set_title('Xu hướng lượt xem & liên hệ theo tuần', fontweight='bold', pad=15)
ax1.set_ylabel('Total Views', color=PALETTE[0])
ax2.set_ylabel('Total Contacts', color=PALETTE[1])
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper right', facecolor='#1a1a2e', edgecolor='#333355')
ax1.grid(alpha=0.3); ax1.set_axisbelow(True)
plt.xticks(rotation=30)
save(fig, '19_weekly_views_contacts_trend')

# 20. New listing vs Old listing
df = con.sql(f"""
    SELECT is_new_listing, AVG(views_24h) as avg_v, AVG(contacts_24h) as avg_c, AVG(ctr_24h) as avg_ctr
    FROM {SNAP} GROUP BY is_new_listing
""").df()
df['label'] = df['is_new_listing'].map({1:'Tin mới', 0:'Tin cũ'})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (metric, title, pal) in enumerate([('avg_v','Avg Views/24h',0),('avg_c','Avg Contacts/24h',1),('avg_ctr','Avg CTR',2)]):
    axes[i].bar(df['label'], df[metric], color=[PALETTE[pal], PALETTE[pal+3]], edgecolor='none', width=0.4)
    for j, v in enumerate(df[metric]):
        axes[i].text(j, v*1.02, f'{v:.3f}' if 'ctr' in metric else f'{v:.2f}', ha='center', color='white', fontsize=11)
    axes[i].set_title(title, fontweight='bold')
    axes[i].grid(axis='y', alpha=0.3); axes[i].set_axisbelow(True)
fig.suptitle('So sánh hiệu suất: Tin mới vs Tin cũ', fontweight='bold', fontsize=16, y=1.03)
fig.tight_layout()
save(fig, '20_new_vs_old_listing_performance')

# ============================================================
# PHẦN 5: HÀNH VI NGƯỜI DÙNG (EVENTS) - EXPANDED
# ============================================================
print("\n" + "=" * 50)
print("PHẦN 5: Hành vi người dùng (Events) - Expanded")
print("=" * 50)

# 21. Phân bố event_type
df = con.sql(f"SELECT event_type, COUNT(*) as cnt FROM {EVENTS} GROUP BY event_type ORDER BY cnt DESC").df()
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(df['event_type'], df['cnt'], color=PALETTE[:len(df)], edgecolor='none', width=0.5)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02, fmt_num(b.get_height()), ha='center', color='white', fontsize=10)
ax.set_title('Phân bố loại sự kiện (Event Type)', fontweight='bold', pad=15)
ax.set_ylabel('Số lượng events'); ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '21_event_type_distribution')

# 22. Phân bố thiết bị
df = con.sql(f"SELECT device, COUNT(*) as cnt FROM {EVENTS} GROUP BY device ORDER BY cnt DESC").df()
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(df['cnt'], labels=df['device'], autopct='%1.1f%%',
    colors=PALETTE[:len(df)], startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.4, edgecolor='#0f0f1a', linewidth=2))
for t in autotexts: t.set_fontsize(12); t.set_color('white'); t.set_fontweight('bold')
for t in texts: t.set_fontsize(12)
ax.set_title('Phân bố thiết bị truy cập', fontweight='bold', pad=20)
save(fig, '22_device_distribution')

# 23. Login vs Non-login theo event_type
df = con.sql(f"""
    SELECT event_type, is_login, COUNT(*) as cnt FROM {EVENTS}
    GROUP BY event_type, is_login ORDER BY cnt DESC
""").df()
fig, ax = plt.subplots(figsize=(12, 7))
events = df.groupby('event_type')['cnt'].sum().sort_values(ascending=False).index.tolist()
x = np.arange(len(events))
for i, login in enumerate(['login','non-login']):
    sub = df[df['is_login']==login].set_index('event_type').reindex(events).fillna(0)
    ax.bar(x + i*0.35, sub['cnt'], 0.35, label='Đã đăng nhập' if login=='login' else 'Chưa đăng nhập',
           color=PALETTE[i], edgecolor='none')
ax.set_xticks(x+0.175); ax.set_xticklabels(events, rotation=20, ha='right')
ax.set_title('Login vs Non-login theo loại sự kiện', fontweight='bold', pad=15)
ax.set_ylabel('Số lượng'); ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.legend(facecolor='#1a1a2e', edgecolor='#333355')
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '23_login_vs_nonlogin_by_event')

# 24. Phân bố dwell_time (lọc outliers)
df = con.sql(f"""
    SELECT dwell_time_sec FROM {EVENTS}
    WHERE dwell_time_sec IS NOT NULL AND dwell_time_sec BETWEEN 1 AND 600
    USING SAMPLE 500000
""").df()
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['dwell_time_sec'], bins=100, color=PALETTE[3], edgecolor='none', alpha=0.85)
ax.axvline(df['dwell_time_sec'].median(), color='#FF6584', ls='--', lw=2, label=f"Median: {df['dwell_time_sec'].median():.0f}s")
ax.set_title('Phân bố thời gian xem trang (Dwell Time)', fontweight='bold', pad=15)
ax.set_xlabel('Thời gian (giây)'); ax.set_ylabel('Tần suất')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.legend(facecolor='#1a1a2e', edgecolor='#333355')
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '24_dwell_time_distribution')

# 25. Xu hướng events theo ngày
df = con.sql(f"SELECT date, COUNT(*) as cnt FROM {EVENTS} GROUP BY date ORDER BY date").df()
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['date'], df['cnt'], color=PALETTE[2], lw=1.5)
ax.fill_between(df['date'], df['cnt'], alpha=0.15, color=PALETTE[2])
ax.set_title('Xu hướng số lượng events theo ngày', fontweight='bold', pad=15)
ax.set_ylabel('Số events'); ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(alpha=0.3); ax.set_axisbelow(True)
plt.xticks(rotation=30)
save(fig, '25_daily_events_trend')

# 26. Giờ cao điểm
df = con.sql(f"SELECT EXTRACT(HOUR FROM event_ts) as hour, COUNT(*) as cnt FROM {EVENTS} GROUP BY hour ORDER BY hour").df()
fig, ax = plt.subplots(figsize=(14, 6))
colors = [PALETTE[0] if 8<=h<=22 else '#333355' for h in df['hour']]
ax.bar(df['hour'].astype(int), df['cnt'], color=colors, edgecolor='none')
ax.set_title('Phân bố events theo giờ trong ngày', fontweight='bold', pad=15)
ax.set_xlabel('Giờ'); ax.set_ylabel('Số events')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.set_xticks(range(24))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '26_hourly_events')

# 27. Ngày trong tuần
df = con.sql(f"SELECT EXTRACT(DOW FROM date) as dow, COUNT(*) as cnt FROM {EVENTS} GROUP BY dow ORDER BY dow").df()
days_vi = ['CN','Thứ 2','Thứ 3','Thứ 4','Thứ 5','Thứ 6','Thứ 7']
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(days_vi, df['cnt'], color=PALETTE[4], edgecolor='none', width=0.5)
for i, v in enumerate(df['cnt']):
    ax.text(i, v*1.02, fmt_num(v), ha='center', color='white', fontsize=10)
ax.set_title('Phân bố events theo ngày trong tuần', fontweight='bold', pad=15)
ax.set_ylabel('Số events'); ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '27_weekday_events')

# ---- EXPANDED: Cross-table charts cho Phần 5 ----

# 28. Events theo category (join dim_listing)
df = con.sql(f"""
    SELECT d.category, COUNT(*) as cnt
    FROM {EVENTS} e JOIN {DIM} d ON e.item_id = d.item_id
    WHERE d.category IN (1010,1020,1030,1040,1050)
    GROUP BY d.category ORDER BY cnt DESC
""").df()
df = map_cat(df)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(df['category'], df['cnt'], color=PALETTE[:len(df)], edgecolor='none', width=0.5)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02, fmt_num(b.get_height()), ha='center', color='white')
ax.set_title('Số lượng events theo danh mục BĐS (demand)', fontweight='bold', pad=15)
ax.set_ylabel('Số events'); ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '28_events_by_category')

# 29. Dwell time trung bình theo category
df = con.sql(f"""
    SELECT d.category, AVG(e.dwell_time_sec) as avg_dwell, MEDIAN(e.dwell_time_sec) as med_dwell
    FROM {EVENTS} e JOIN {DIM} d ON e.item_id = d.item_id
    WHERE e.dwell_time_sec BETWEEN 1 AND 3600 AND d.category IN (1010,1020,1030,1040,1050)
    GROUP BY d.category
""").df()
df = map_cat(df)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df))
ax.bar(x - 0.175, df['avg_dwell'], 0.35, label='Trung bình', color=PALETTE[0], edgecolor='none')
ax.bar(x + 0.175, df['med_dwell'], 0.35, label='Trung vị', color=PALETTE[2], edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels(df['category'], rotation=15, ha='right')
ax.set_title('Dwell Time trung bình theo danh mục BĐS', fontweight='bold', pad=15)
ax.set_ylabel('Thời gian (giây)')
ax.legend(facecolor='#1a1a2e', edgecolor='#333355')
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '29_dwell_time_by_category')

# 30. Device × Category heatmap
df = con.sql(f"""
    SELECT e.device, d.category, COUNT(*) as cnt
    FROM {EVENTS} e JOIN {DIM} d ON e.item_id = d.item_id
    WHERE d.category IN (1010,1020,1030,1040,1050)
    GROUP BY e.device, d.category
""").df()
df = map_cat(df)
pivot = df.pivot_table(index='device', columns='category', values='cnt', fill_value=0)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt=',.0f', cmap='magma', ax=ax, linewidths=0.5, linecolor='#333355')
ax.set_title('Heatmap: Thiết bị × Danh mục BĐS', fontweight='bold', pad=15)
save(fig, '30_device_category_heatmap')

# 31. Login rate theo price_bucket
df = con.sql(f"""
    SELECT d.price_bucket,
           COUNT(CASE WHEN e.is_login='login' THEN 1 END)*100.0/COUNT(*) as login_rate,
           COUNT(*) as total
    FROM {EVENTS} e JOIN {DIM} d ON e.item_id = d.item_id
    WHERE d.price_bucket IS NOT NULL AND d.ad_type='sell'
    GROUP BY d.price_bucket HAVING total > 10000
    ORDER BY login_rate DESC
""").df()
sell_order = ['<500M','500M–800M','1B–1.5B','1.5B–2B','2B–3B','3B–5B','5B–7B','7B–10B','10B–15B','15B–20B','>20B']
df = df.set_index('price_bucket').reindex(sell_order).dropna().reset_index()
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(df['price_bucket'][::-1], df['login_rate'][::-1], color=PALETTE[5], edgecolor='none', height=0.6)
for i, v in enumerate(df['login_rate'][::-1]):
    ax.text(v+0.3, i, f'{v:.1f}%', va='center', color='white', fontsize=10)
ax.set_title('Tỷ lệ đăng nhập theo mức giá (Mua bán)', fontweight='bold', pad=15)
ax.set_xlabel('Tỷ lệ login (%)'); ax.grid(axis='x', alpha=0.3); ax.set_axisbelow(True)
save(fig, '31_login_rate_by_price_bucket')

# 32. Session depth distribution
df = con.sql(f"""
    SELECT events_per_session, COUNT(*) as cnt FROM (
        SELECT session_id, COUNT(*) as events_per_session FROM {EVENTS}
        WHERE session_id IS NOT NULL
        GROUP BY session_id
    ) WHERE events_per_session <= 50
    GROUP BY events_per_session ORDER BY events_per_session
""").df()
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(df['events_per_session'], df['cnt'], color=PALETTE[6], edgecolor='none')
ax.set_title('Phân bố độ sâu phiên (Events per Session)', fontweight='bold', pad=15)
ax.set_xlabel('Số events trong 1 session'); ax.set_ylabel('Số sessions')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '32_session_depth_distribution')

# 33. Event type × Device heatmap
df = con.sql(f"""
    SELECT event_type, device, COUNT(*) as cnt FROM {EVENTS}
    GROUP BY event_type, device
""").df()
pivot = df.pivot_table(index='event_type', columns='device', values='cnt', fill_value=0)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(pivot, annot=True, fmt=',.0f', cmap='magma', ax=ax, linewidths=0.5, linecolor='#333355')
ax.set_title('Heatmap: Loại sự kiện × Thiết bị', fontweight='bold', pad=15)
save(fig, '33_event_type_device_heatmap')

print("\n✅ Phần 4-5 hoàn thành (19 charts)")
con.close()
