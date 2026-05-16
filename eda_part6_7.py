"""Phần 6-7: Tương tác User-Ad (Expanded) + Cross-analysis"""
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
INTER = f"'{BASE}/fact_user_ad_interactions/*.parquet'"
EVENTS = f"'{BASE}/clean_fact_user_events/*.parquet'"
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
# PHẦN 6: TƯƠNG TÁC USER-AD - EXPANDED
# ============================================================
print("=" * 50)
print("PHẦN 6: Tương tác User-Ad (Expanded)")
print("=" * 50)

# 34. Funnel chuyển đổi tổng thể
df = con.sql(f"""
    SELECT
        COUNT(*) as total_interactions,
        SUM(CASE WHEN adview_count > 0 THEN 1 ELSE 0 END) as has_adview,
        SUM(CASE WHEN lead_count > 0 THEN 1 ELSE 0 END) as has_lead,
        SUM(CASE WHEN chat_message_count > 0 THEN 1 ELSE 0 END) as has_chat,
        SUM(CASE WHEN purchased THEN 1 ELSE 0 END) as has_purchase
    FROM {INTER}
""").fetchone()
labels = ['Tương tác','Xem tin','Có Lead','Có Chat','Mua']
values = list(df)
fig, ax = plt.subplots(figsize=(12, 7))
colors_funnel = PALETTE[:5]
for i, (lbl, val) in enumerate(zip(labels, values)):
    width = val / values[0]
    ax.barh(len(labels)-1-i, width, height=0.6, left=(1-width)/2, color=colors_funnel[i], edgecolor='none')
    ax.text(0.5, len(labels)-1-i, f'{lbl}\n{fmt_num(val)} ({val/values[0]*100:.1f}%)',
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.set_xlim(0, 1); ax.axis('off')
ax.set_title('Funnel chuyển đổi tương tác', fontweight='bold', pad=20, fontsize=16)
save(fig, '34_conversion_funnel')

# 35. Tỷ lệ purchased theo category
df = con.sql(f"""
    SELECT category, COUNT(*) as total,
           SUM(CASE WHEN purchased THEN 1 ELSE 0 END)*100.0/COUNT(*) as purchase_rate
    FROM {INTER} GROUP BY category ORDER BY purchase_rate DESC
""").df()
df = map_cat(df)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(df['category'], df['purchase_rate'], color=PALETTE[:len(df)], edgecolor='none', width=0.5)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{b.get_height():.2f}%', ha='center', color='white')
ax.set_title('Tỷ lệ mua (Purchase Rate) theo danh mục BĐS', fontweight='bold', pad=15)
ax.set_ylabel('Purchase Rate (%)'); ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '35_purchase_rate_by_category')

# 36. Phân bố adview_count
df = con.sql(f"SELECT adview_count FROM {INTER} WHERE adview_count BETWEEN 1 AND 30 USING SAMPLE 500000").df()
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['adview_count'], bins=30, color=PALETTE[0], edgecolor='none', alpha=0.85)
ax.set_title('Phân bố số lần xem tin (Adview Count)', fontweight='bold', pad=15)
ax.set_xlabel('Adview Count'); ax.set_ylabel('Tần suất')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '36_adview_count_distribution')

# 37. Phân bố chat_message_count
df = con.sql(f"SELECT chat_message_count FROM {INTER} WHERE chat_message_count BETWEEN 1 AND 30 USING SAMPLE 300000").df()
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['chat_message_count'], bins=30, color=PALETTE[1], edgecolor='none', alpha=0.85)
ax.set_title('Phân bố số tin nhắn chat', fontweight='bold', pad=15)
ax.set_xlabel('Chat Message Count'); ax.set_ylabel('Tần suất')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '37_chat_message_distribution')

# 38. Xu hướng tương tác theo tuần
df = con.sql(f"""
    SELECT DATE_TRUNC('week', date) as week,
           SUM(adview_count) as views, SUM(lead_count) as leads,
           SUM(chat_message_count) as chats
    FROM {INTER} GROUP BY week ORDER BY week
""").df()
fig, ax = plt.subplots(figsize=(14, 6))
for i, (col, lbl) in enumerate([('views','Adviews'),('leads','Leads'),('chats','Chats')]):
    ax.plot(df['week'], df[col], color=PALETTE[i], lw=2, label=lbl, marker='o', markersize=3)
ax.set_title('Xu hướng tương tác theo tuần', fontweight='bold', pad=15)
ax.set_ylabel('Tổng số'); ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
ax.legend(facecolor='#1a1a2e', edgecolor='#333355')
ax.grid(alpha=0.3); ax.set_axisbelow(True); plt.xticks(rotation=30)
save(fig, '38_weekly_interaction_trends')

# ---- EXPANDED: Cross-table charts cho Phần 6 ----

# 39. Conversion funnel theo category (join dim_listing)
df = con.sql(f"""
    SELECT d.category,
           COUNT(*) as total,
           SUM(CASE WHEN i.adview_count > 0 THEN 1 ELSE 0 END)*100.0/COUNT(*) as view_rate,
           SUM(CASE WHEN i.lead_count > 0 THEN 1 ELSE 0 END)*100.0/COUNT(*) as lead_rate,
           SUM(CASE WHEN i.chat_message_count > 0 THEN 1 ELSE 0 END)*100.0/COUNT(*) as chat_rate,
           SUM(CASE WHEN i.purchased THEN 1 ELSE 0 END)*100.0/COUNT(*) as purchase_rate
    FROM {INTER} i JOIN {DIM} d ON i.item_id = d.item_id
    WHERE d.category IN (1010,1020,1030,1040,1050)
    GROUP BY d.category
""").df()
df = map_cat(df)
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(df))
w = 0.18
for i, (col, lbl) in enumerate([('view_rate','Xem tin'),('lead_rate','Lead'),('chat_rate','Chat'),('purchase_rate','Mua')]):
    ax.bar(x + i*w, df[col], w, label=lbl, color=PALETTE[i], edgecolor='none')
ax.set_xticks(x + 1.5*w); ax.set_xticklabels(df['category'], rotation=15, ha='right')
ax.set_title('Tỷ lệ chuyển đổi theo danh mục BĐS', fontweight='bold', pad=15)
ax.set_ylabel('Tỷ lệ (%)'); ax.legend(facecolor='#1a1a2e', edgecolor='#333355')
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '39_conversion_by_category')

# 40. Purchase rate theo seller_type
df = con.sql(f"""
    SELECT d.seller_type,
           SUM(CASE WHEN i.purchased THEN 1 ELSE 0 END)*100.0/COUNT(*) as purchase_rate,
           AVG(i.adview_count) as avg_views, AVG(i.chat_message_count) as avg_chats,
           COUNT(*) as total
    FROM {INTER} i JOIN {DIM} d ON i.item_id = d.item_id
    GROUP BY d.seller_type
""").df()
df['label'] = df['seller_type'].map({'agent':'Môi giới','private':'Cá nhân'})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = [('purchase_rate','Tỷ lệ mua (%)',1),('avg_views','Avg Adviews',0),('avg_chats','Avg Chats',2)]
for i, (m, title, ci) in enumerate(metrics):
    axes[i].bar(df['label'], df[m], color=[PALETTE[ci], PALETTE[ci+3]], edgecolor='none', width=0.4)
    for j, v in enumerate(df[m]):
        axes[i].text(j, v*1.02, f'{v:.2f}', ha='center', color='white', fontsize=11)
    axes[i].set_title(title, fontweight='bold')
    axes[i].grid(axis='y', alpha=0.3); axes[i].set_axisbelow(True)
fig.suptitle('So sánh tương tác: Môi giới vs Cá nhân', fontweight='bold', fontsize=16, y=1.03)
fig.tight_layout()
save(fig, '40_interaction_by_seller_type')

# 41. Purchase rate theo city (top 10)
df = con.sql(f"""
    SELECT d.city_name,
           SUM(CASE WHEN i.purchased THEN 1 ELSE 0 END)*100.0/COUNT(*) as purchase_rate,
           COUNT(*) as total
    FROM {INTER} i JOIN {DIM} d ON i.item_id = d.item_id
    GROUP BY d.city_name HAVING total > 5000
    ORDER BY purchase_rate DESC LIMIT 15
""").df()
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(df['city_name'][::-1], df['purchase_rate'][::-1], color=PALETTE[6], edgecolor='none', height=0.6)
for i, v in enumerate(df['purchase_rate'][::-1]):
    ax.text(v+0.02, i, f'{v:.2f}%', va='center', color='white', fontsize=10)
ax.set_title('Tỷ lệ mua theo tỉnh/thành (Top 15)', fontweight='bold', pad=15)
ax.set_xlabel('Purchase Rate (%)'); ax.grid(axis='x', alpha=0.3); ax.set_axisbelow(True)
save(fig, '41_purchase_rate_by_city')

# 42. Purchase rate theo price_bucket (sell only)
df = con.sql(f"""
    SELECT d.price_bucket,
           SUM(CASE WHEN i.purchased THEN 1 ELSE 0 END)*100.0/COUNT(*) as purchase_rate,
           COUNT(*) as total
    FROM {INTER} i JOIN {DIM} d ON i.item_id = d.item_id
    WHERE d.ad_type='sell' AND d.price_bucket IS NOT NULL
    GROUP BY d.price_bucket HAVING total > 1000
""").df()
sell_order = ['<500M','500M–800M','1B–1.5B','1.5B–2B','2B–3B','3B–5B','5B–7B','7B–10B','10B–15B','15B–20B','>20B']
df = df.set_index('price_bucket').reindex(sell_order).dropna().reset_index()
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(df['price_bucket'][::-1], df['purchase_rate'][::-1], color=PALETTE[3], edgecolor='none', height=0.6)
for i, v in enumerate(df['purchase_rate'][::-1]):
    ax.text(v+0.02, i, f'{v:.2f}%', va='center', color='white', fontsize=10)
ax.set_title('Tỷ lệ mua theo mức giá (Mua bán)', fontweight='bold', pad=15)
ax.set_xlabel('Purchase Rate (%)'); ax.grid(axis='x', alpha=0.3); ax.set_axisbelow(True)
save(fig, '42_purchase_rate_by_price_bucket')

# 43. Chat intensity theo furnishing
df = con.sql(f"""
    SELECT d.furnishing, AVG(i.chat_message_count) as avg_chat, AVG(i.lead_count) as avg_lead
    FROM {INTER} i JOIN {DIM} d ON i.item_id = d.item_id
    WHERE d.furnishing != 'Unknown'
    GROUP BY d.furnishing
""").df()
furnish_order = ['Bàn giao thô','Hoàn thiện cơ bản','Nhà trống','Nội thất đầy đủ','Nội thất cao cấp']
df = df.set_index('furnishing').reindex(furnish_order).reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df))
ax.bar(x-0.175, df['avg_chat'], 0.35, label='Avg Chat', color=PALETTE[1], edgecolor='none')
ax.bar(x+0.175, df['avg_lead'], 0.35, label='Avg Lead', color=PALETTE[2], edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels(df['furnishing'], rotation=15, ha='right')
ax.set_title('Mức độ tương tác theo tình trạng nội thất', fontweight='bold', pad=15)
ax.set_ylabel('Trung bình'); ax.legend(facecolor='#1a1a2e', edgecolor='#333355')
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '43_chat_lead_by_furnishing')

# 44. Conversion theo listing_age (join snapshot)
df = con.sql(f"""
    SELECT
        CASE WHEN s.listing_age_days <= 7 THEN '0-7 ngày'
             WHEN s.listing_age_days <= 14 THEN '8-14 ngày'
             WHEN s.listing_age_days <= 30 THEN '15-30 ngày'
             WHEN s.listing_age_days <= 60 THEN '31-60 ngày'
             WHEN s.listing_age_days <= 90 THEN '61-90 ngày'
             ELSE '>90 ngày' END as age_group,
        AVG(s.views_24h) as avg_views, AVG(s.contacts_24h) as avg_contacts, AVG(s.ctr_24h) as avg_ctr
    FROM {SNAP} s
    WHERE s.listing_age_days >= 0
    GROUP BY age_group
""").df()
order = ['0-7 ngày','8-14 ngày','15-30 ngày','31-60 ngày','61-90 ngày','>90 ngày']
df = df.set_index('age_group').reindex(order).reset_index()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (col, title, ci) in enumerate([('avg_views','Avg Views/24h',0),('avg_contacts','Avg Contacts/24h',1),('avg_ctr','Avg CTR',2)]):
    axes[i].bar(df['age_group'], df[col], color=PALETTE[ci], edgecolor='none', width=0.5)
    axes[i].set_title(title, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].grid(axis='y', alpha=0.3); axes[i].set_axisbelow(True)
fig.suptitle('Hiệu suất tin đăng theo tuổi listing', fontweight='bold', fontsize=16, y=1.03)
fig.tight_layout()
save(fig, '44_performance_by_listing_age')

# ============================================================
# PHẦN 7: CROSS-ANALYSIS
# ============================================================
print("\n" + "=" * 50)
print("PHẦN 7: Cross-analysis")
print("=" * 50)

# 45. Supply vs Demand by category
supply = con.sql(f"SELECT category, COUNT(*) as supply FROM {DIM} WHERE category IN (1010,1020,1030,1040,1050) GROUP BY category").df()
demand = con.sql(f"SELECT category, COUNT(*) as demand FROM {EVENTS} WHERE category IN (1010,1020,1030,1040,1050) GROUP BY category").df()
df = supply.merge(demand, on='category')
df = map_cat(df)
df['demand_per_supply'] = df['demand'] / df['supply']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(df))
axes[0].bar(x-0.175, df['supply'], 0.35, label='Cung (Tin đăng)', color=PALETTE[0], edgecolor='none')
axes[0].bar(x+0.175, df['demand'], 0.35, label='Cầu (Events)', color=PALETTE[1], edgecolor='none')
axes[0].set_xticks(x); axes[0].set_xticklabels(df['category'], rotation=15, ha='right')
axes[0].set_title('Cung vs Cầu theo danh mục', fontweight='bold')
axes[0].set_ylabel('Số lượng'); axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_num))
axes[0].legend(facecolor='#1a1a2e', edgecolor='#333355')
axes[0].grid(axis='y', alpha=0.3); axes[0].set_axisbelow(True)

axes[1].bar(df['category'], df['demand_per_supply'], color=PALETTE[2:2+len(df)], edgecolor='none', width=0.5)
for i, v in enumerate(df['demand_per_supply']):
    axes[1].text(i, v*1.02, f'{v:.1f}', ha='center', color='white', fontsize=11)
axes[1].set_title('Tỷ lệ Cầu/Cung (Events per Listing)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)
axes[1].grid(axis='y', alpha=0.3); axes[1].set_axisbelow(True)
fig.tight_layout()
save(fig, '45_supply_vs_demand_by_category')

# 46. Performance theo seller_type (join dim + snapshot)
df = con.sql(f"""
    SELECT d.seller_type, AVG(s.views_24h) as avg_v, AVG(s.contacts_24h) as avg_c, AVG(s.ctr_24h) as avg_ctr
    FROM {SNAP} s JOIN {DIM} d ON s.item_id = d.item_id
    GROUP BY d.seller_type
""").df()
df['label'] = df['seller_type'].map({'agent':'Môi giới','private':'Cá nhân'})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (m, title, ci) in enumerate([('avg_v','Avg Views/24h',0),('avg_c','Avg Contacts/24h',1),('avg_ctr','Avg CTR',2)]):
    axes[i].bar(df['label'], df[m], color=[PALETTE[ci], PALETTE[ci+3]], edgecolor='none', width=0.4)
    for j, v in enumerate(df[m]):
        axes[i].text(j, v*1.02, f'{v:.3f}' if 'ctr' in m else f'{v:.2f}', ha='center', color='white', fontsize=11)
    axes[i].set_title(title, fontweight='bold')
    axes[i].grid(axis='y', alpha=0.3); axes[i].set_axisbelow(True)
fig.suptitle('Hiệu suất tin đăng: Môi giới vs Cá nhân', fontweight='bold', fontsize=16, y=1.03)
fig.tight_layout()
save(fig, '46_performance_by_seller_type')

# 47. Performance theo city (top 10)
df = con.sql(f"""
    SELECT d.city_name, AVG(s.views_24h) as avg_v, AVG(s.contacts_24h) as avg_c, AVG(s.ctr_24h) as avg_ctr
    FROM {SNAP} s JOIN {DIM} d ON s.item_id = d.item_id
    GROUP BY d.city_name ORDER BY avg_v DESC LIMIT 10
""").df()
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(df))
ax.bar(x-0.15, df['avg_v'], 0.3, label='Avg Views', color=PALETTE[0], edgecolor='none')
ax2 = ax.twinx()
ax2.bar(x+0.15, df['avg_c'], 0.3, label='Avg Contacts', color=PALETTE[1], edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels(df['city_name'], rotation=30, ha='right')
ax.set_title('Avg Views & Contacts theo tỉnh/thành (Top 10)', fontweight='bold', pad=15)
ax.set_ylabel('Avg Views', color=PALETTE[0]); ax2.set_ylabel('Avg Contacts', color=PALETTE[1])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc='upper right', facecolor='#1a1a2e', edgecolor='#333355')
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
save(fig, '47_performance_by_city')

# 48. Views/Contacts by category (join dim + snapshot)
df = con.sql(f"""
    SELECT d.category, AVG(s.views_24h) as avg_v, AVG(s.contacts_24h) as avg_c, AVG(s.ctr_24h) as avg_ctr
    FROM {SNAP} s JOIN {DIM} d ON s.item_id = d.item_id
    WHERE d.category IN (1010,1020,1030,1040,1050)
    GROUP BY d.category
""").df()
df = map_cat(df)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (m, title, ci) in enumerate([('avg_v','Avg Views/24h',0),('avg_c','Avg Contacts/24h',1),('avg_ctr','Avg CTR',2)]):
    axes[i].bar(df['category'], df[m], color=PALETTE[:len(df)], edgecolor='none', width=0.5)
    for j, v in enumerate(df[m]):
        axes[i].text(j, v*1.02, f'{v:.4f}' if 'ctr' in m else f'{v:.2f}', ha='center', color='white', fontsize=10)
    axes[i].set_title(title, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=15)
    axes[i].grid(axis='y', alpha=0.3); axes[i].set_axisbelow(True)
fig.suptitle('Hiệu suất tin đăng theo danh mục BĐS', fontweight='bold', fontsize=16, y=1.03)
fig.tight_layout()
save(fig, '48_performance_by_category')

print("\n✅ PHẦN 6-7 hoàn thành (15 charts)")
print("=" * 50)
print("TỔNG CỘNG: 48 biểu đồ đã được tạo tại outputs/eda/")
print("=" * 50)
con.close()
