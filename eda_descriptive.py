"""
Descriptive EDA - Chợ Tốt Bất Động Sản
Sử dụng DuckDB để query dữ liệu lớn, Matplotlib/Seaborn để vẽ biểu đồ.
"""
import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os
import warnings
import sys, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# SETUP
# ============================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor': '#1a1a2e',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0',
    'axes.edgecolor': '#333355',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.5,
})

OUTPUT_DIR = r"f:\PYTHON\Datathon2026-Deep67-Finals\outputs\eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = r"f:/PYTHON/Datathon2026-Deep67-Finals/data/processed"
DIM = f"'{BASE}/dim_listing_cleaned/dim_listing_cleaned.parquet'"
SNAP = f"'{BASE}/fact_listing_snapshot_cleaned/fact_listing_snapshot_cleaned.parquet'"
INTER = f"'{BASE}/fact_user_ad_interactions/*.parquet'"
EVENTS = f"'{BASE}/clean_fact_user_events/*.parquet'"

CATEGORY_MAP = {
    1010: 'Phòng trọ/Studio',
    1020: 'Căn hộ/Chung cư',
    1030: 'Nhà ở',
    1040: 'Đất',
    1050: 'Dự án mở bán',
}

PALETTE = ['#6C63FF', '#FF6584', '#43E8D8', '#FFD93D', '#FF8C42',
           '#A78BFA', '#34D399', '#F472B6', '#60A5FA', '#FBBF24']

con = duckdb.connect()

def save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
    plt.close(fig)
    print(f"  ✓ {name}.png")

def map_cat(df, col='category'):
    df[col] = df[col].map(CATEGORY_MAP).fillna(df[col].astype(str))
    return df

def fmt_num(x, _=None):
    if x >= 1e6: return f'{x/1e6:.1f}M'
    if x >= 1e3: return f'{x/1e3:.0f}K'
    return f'{x:.0f}'

# ============================================================
# PHẦN 1: TỔNG QUAN THỊ TRƯỜNG
# ============================================================
print("=" * 50)
print("PHẦN 1: Tổng quan thị trường")
print("=" * 50)

# 1. Phân bố danh mục BĐS
df = con.sql(f"SELECT category, COUNT(*) as cnt FROM {DIM} GROUP BY category ORDER BY cnt DESC").df()
df = map_cat(df)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(df['category'], df['cnt'], color=PALETTE[:len(df)], edgecolor='none', width=0.6)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20000,
            fmt_num(b.get_height()), ha='center', va='bottom', fontsize=11, color='white')
ax.set_title('Phân bố tin đăng theo danh mục BĐS', fontweight='bold', pad=15)
ax.set_ylabel('Số lượng tin đăng')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
save(fig, '01_category_distribution')

# 2. Phân bố giá theo loại (sell vs let)
df = con.sql(f"""
    SELECT price_bucket, ad_type, COUNT(*) as cnt FROM {DIM}
    WHERE price_bucket IS NOT NULL
    GROUP BY price_bucket, ad_type ORDER BY cnt DESC
""").df()
sell_order = ['<500M','500M–800M','1B–1.5B','1.5B–2B','2B–3B','3B–5B','5B–7B','7B–10B','10B–15B','15B–20B','>20B']
let_order = ['<2M/tháng','2M–3M/tháng','3M–5M/tháng','5M–7M/tháng','7M–10M/tháng','10M–15M/tháng','15M–20M/tháng','20M–30M/tháng','>30M/tháng']
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for i, (ad, order, color) in enumerate([('sell', sell_order, '#6C63FF'), ('let', let_order, '#FF6584')]):
    sub = df[df['ad_type'] == ad].set_index('price_bucket').reindex(order).fillna(0)
    axes[i].barh(sub.index, sub['cnt'], color=color, edgecolor='none', height=0.6)
    axes[i].set_title(f'Phân bố giá - {"Mua bán" if ad=="sell" else "Cho thuê"}', fontweight='bold')
    axes[i].set_xlabel('Số lượng tin đăng')
    axes[i].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
    axes[i].invert_yaxis()
    axes[i].grid(axis='x', alpha=0.3)
    axes[i].set_axisbelow(True)
fig.suptitle('Phân bố giá theo loại hình giao dịch', fontweight='bold', fontsize=16, y=1.02)
fig.tight_layout()
save(fig, '02_price_distribution_by_type')

# 3. Phân bố loại người bán
df = con.sql(f"SELECT seller_type, COUNT(*) as cnt FROM {DIM} GROUP BY seller_type").df()
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#6C63FF', '#FF6584']
wedges, texts, autotexts = ax.pie(df['cnt'], labels=df['seller_type'].map({'agent':'Môi giới','private':'Cá nhân'}),
    autopct='%1.1f%%', colors=colors, startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.4, edgecolor='#0f0f1a', linewidth=2))
for t in autotexts: t.set_fontsize(13); t.set_color('white'); t.set_fontweight('bold')
for t in texts: t.set_fontsize(13)
ax.set_title('Tỷ lệ loại người bán', fontweight='bold', pad=20)
save(fig, '03_seller_type_distribution')

# 4. Phân bố trạng thái tin đăng
df = con.sql(f"SELECT ad_status, COUNT(*) as cnt FROM {DIM} GROUP BY ad_status ORDER BY cnt DESC").df()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(df['ad_status'], df['cnt'], color=PALETTE[:len(df)], edgecolor='none', width=0.5)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 10000,
            fmt_num(b.get_height()), ha='center', va='bottom', fontsize=11, color='white')
ax.set_title('Phân bố trạng thái tin đăng', fontweight='bold', pad=15)
ax.set_ylabel('Số lượng')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '04_ad_status_distribution')

# 5. Phân bố tình trạng pháp lý
df = con.sql(f"""
    SELECT legal_status, COUNT(*) as cnt FROM {DIM}
    WHERE legal_status NOT IN ('4','6')
    GROUP BY legal_status ORDER BY cnt DESC
""").df()
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(df['legal_status'][::-1], df['cnt'][::-1], color=PALETTE[0], edgecolor='none', height=0.6)
for i, (v, n) in enumerate(zip(df['cnt'][::-1], df['legal_status'][::-1])):
    ax.text(v + 10000, i, fmt_num(v), va='center', fontsize=10, color='white')
ax.set_title('Phân bố tình trạng pháp lý', fontweight='bold', pad=15)
ax.set_xlabel('Số lượng tin đăng')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='x', alpha=0.3); ax.set_axisbelow(True)
save(fig, '05_legal_status_distribution')

# ============================================================
# PHẦN 2: PHÂN TÍCH ĐỊA LÝ
# ============================================================
print("\n" + "=" * 50)
print("PHẦN 2: Phân tích Địa lý")
print("=" * 50)

# 6. Top 15 tỉnh/thành
df = con.sql(f"SELECT city_name, COUNT(*) as cnt FROM {DIM} GROUP BY city_name ORDER BY cnt DESC LIMIT 15").df()
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(df['city_name'][::-1], df['cnt'][::-1], color=PALETTE[0], edgecolor='none', height=0.6)
for i, v in enumerate(df['cnt'][::-1]):
    ax.text(v + 5000, i, fmt_num(v), va='center', fontsize=10, color='white')
ax.set_title('Top 15 tỉnh/thành phố theo số lượng tin đăng', fontweight='bold', pad=15)
ax.set_xlabel('Số lượng tin đăng')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='x', alpha=0.3); ax.set_axisbelow(True)
save(fig, '06_top15_cities')

# 7. HCM vs HN vs DN - So sánh category
df = con.sql(f"""
    SELECT city_name, category, COUNT(*) as cnt FROM {DIM}
    WHERE city_name IN ('Tp Hồ Chí Minh','Hà Nội','Đà Nẵng')
    GROUP BY city_name, category
""").df()
df = map_cat(df)
fig, ax = plt.subplots(figsize=(12, 7))
cities = ['Tp Hồ Chí Minh','Hà Nội','Đà Nẵng']
cats = list(CATEGORY_MAP.values())
x = np.arange(len(cities))
w = 0.15
for i, cat in enumerate(cats):
    vals = [df[(df['city_name']==c) & (df['category']==cat)]['cnt'].sum() for c in cities]
    ax.bar(x + i*w, vals, w, label=cat, color=PALETTE[i], edgecolor='none')
ax.set_xticks(x + w*2)
ax.set_xticklabels(cities)
ax.set_title('So sánh danh mục BĐS: HCM vs Hà Nội vs Đà Nẵng', fontweight='bold', pad=15)
ax.set_ylabel('Số lượng tin đăng')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.legend(fontsize=9, loc='upper right', facecolor='#1a1a2e', edgecolor='#333355')
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '07_hcm_hn_dn_category_comparison')

# 8. Heatmap tỉnh × danh mục
df = con.sql(f"""
    SELECT city_name, category, COUNT(*) as cnt FROM {DIM}
    GROUP BY city_name, category
""").df()
df = map_cat(df)
top_cities = con.sql(f"SELECT city_name FROM {DIM} GROUP BY city_name ORDER BY COUNT(*) DESC LIMIT 10").df()['city_name'].tolist()
df = df[df['city_name'].isin(top_cities)]
pivot = df.pivot_table(index='city_name', columns='category', values='cnt', fill_value=0)
pivot = pivot.reindex(top_cities)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt=',.0f', cmap='magma', ax=ax,
            linewidths=0.5, linecolor='#333355', cbar_kws={'label': 'Số lượng'})
ax.set_title('Heatmap: Top 10 tỉnh/thành × Danh mục BĐS', fontweight='bold', pad=15)
ax.set_ylabel(''); ax.set_xlabel('')
save(fig, '08_city_category_heatmap')

# 9. Top 20 quận/huyện
df = con.sql(f"SELECT district_name, city_name, COUNT(*) as cnt FROM {DIM} GROUP BY district_name, city_name ORDER BY cnt DESC LIMIT 20").df()
df['label'] = df['district_name'] + ' (' + df['city_name'].str.slice(0,6) + ')'
fig, ax = plt.subplots(figsize=(12, 9))
ax.barh(df['label'][::-1], df['cnt'][::-1], color=PALETTE[2], edgecolor='none', height=0.6)
for i, v in enumerate(df['cnt'][::-1]):
    ax.text(v + 1000, i, fmt_num(v), va='center', fontsize=9, color='white')
ax.set_title('Top 20 quận/huyện phổ biến nhất', fontweight='bold', pad=15)
ax.set_xlabel('Số lượng tin đăng')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='x', alpha=0.3); ax.set_axisbelow(True)
save(fig, '09_top20_districts')

# ============================================================
# PHẦN 3: ĐẶC ĐIỂM TIN ĐĂNG
# ============================================================
print("\n" + "=" * 50)
print("PHẦN 3: Đặc điểm tin đăng")
print("=" * 50)

# 10. Phân bố diện tích theo ad_type
df = con.sql(f"""
    SELECT ad_type, area_sqm FROM {DIM}
    WHERE area_sqm BETWEEN 10 AND 1000
""").df()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for i, (at, color, title) in enumerate([('sell','#6C63FF','Mua bán'), ('let','#FF6584','Cho thuê')]):
    sub = df[df['ad_type']==at]['area_sqm']
    axes[i].hist(sub, bins=80, color=color, edgecolor='none', alpha=0.85)
    axes[i].set_title(f'Phân bố diện tích - {title}', fontweight='bold')
    axes[i].set_xlabel('Diện tích (m²)')
    axes[i].set_ylabel('Số lượng')
    axes[i].axvline(sub.median(), color='#FFD93D', ls='--', lw=2, label=f'Median: {sub.median():.0f}m²')
    axes[i].legend(facecolor='#1a1a2e', edgecolor='#333355')
    axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
    axes[i].grid(axis='y', alpha=0.3); axes[i].set_axisbelow(True)
fig.tight_layout()
save(fig, '10_area_distribution_by_type')

# 11. Phân bố số phòng ngủ
df = con.sql(f"SELECT bedrooms, COUNT(*) as cnt FROM {DIM} WHERE bedrooms BETWEEN 0 AND 10 GROUP BY bedrooms ORDER BY bedrooms").df()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df['bedrooms'].astype(int).astype(str), df['cnt'], color=PALETTE[4], edgecolor='none', width=0.6)
for i, v in enumerate(df['cnt']):
    ax.text(i, v + 10000, fmt_num(v), ha='center', fontsize=10, color='white')
ax.set_title('Phân bố số phòng ngủ', fontweight='bold', pad=15)
ax.set_xlabel('Số phòng ngủ'); ax.set_ylabel('Số lượng tin đăng')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '11_bedrooms_distribution')

# 12. Phân bố số ảnh
df = con.sql(f"SELECT images_count, COUNT(*) as cnt FROM {DIM} WHERE images_count <= 30 GROUP BY images_count ORDER BY images_count").df()
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(df['images_count'].astype(int), df['cnt'], color=PALETTE[5], edgecolor='none')
ax.set_title('Phân bố số ảnh trong tin đăng', fontweight='bold', pad=15)
ax.set_xlabel('Số ảnh'); ax.set_ylabel('Số lượng tin đăng')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '12_images_count_distribution')

# 13. Nội thất vs Giá
df = con.sql(f"""
    SELECT furnishing, ad_type, COUNT(*) as cnt FROM {DIM}
    WHERE furnishing != 'Unknown'
    GROUP BY furnishing, ad_type ORDER BY cnt DESC
""").df()
fig, ax = plt.subplots(figsize=(12, 7))
furnish_order = ['Bàn giao thô','Hoàn thiện cơ bản','Nhà trống','Nội thất đầy đủ','Nội thất cao cấp']
x = np.arange(len(furnish_order))
for i, at in enumerate(['sell','let']):
    sub = df[df['ad_type']==at].set_index('furnishing').reindex(furnish_order).fillna(0)
    ax.bar(x + i*0.35, sub['cnt'], 0.35, label='Mua bán' if at=='sell' else 'Cho thuê',
           color=PALETTE[i], edgecolor='none')
ax.set_xticks(x + 0.175)
ax.set_xticklabels(furnish_order, rotation=15, ha='right')
ax.set_title('Tình trạng nội thất theo loại giao dịch', fontweight='bold', pad=15)
ax.set_ylabel('Số lượng tin đăng')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.legend(facecolor='#1a1a2e', edgecolor='#333355')
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '13_furnishing_by_type')

# 14. Loại nhà vs Diện tích
df = con.sql(f"""
    SELECT house_type, area_sqm FROM {DIM}
    WHERE house_type != 'Unknown' AND area_sqm BETWEEN 10 AND 500
""").df()
fig, ax = plt.subplots(figsize=(12, 7))
order = ['Nhà ngõ, hẻm','Nhà mặt phố, mặt tiền','Nhà phố liền kề','Nhà biệt thự']
data = [df[df['house_type']==h]['area_sqm'].values for h in order]
bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False,
                medianprops=dict(color='#FFD93D', linewidth=2))
for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.7); patch.set_edgecolor('white')
ax.set_title('Phân bố diện tích theo loại nhà', fontweight='bold', pad=15)
ax.set_ylabel('Diện tích (m²)')
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '14_house_type_vs_area')

print("\n✅ Phần 1-3 hoàn thành (14 charts)")
