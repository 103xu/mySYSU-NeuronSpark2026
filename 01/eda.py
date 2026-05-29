"""Step 1: 数据读取与EDA分析"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "136b479a-05b3-426d-b95d-1530d094f5be"

# ============================================================
# 1. 数据读取
# ============================================================
print("=" * 60)
print("1. 数据读取")
print("=" * 60)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")
weather = pd.read_csv(f"{DATA_DIR}/weather.csv")
menu = pd.read_csv(f"{DATA_DIR}/menu.csv")

print(f"train.csv:    {train.shape}, 日期范围: {train['date'].min()} ~ {train['date'].max()}")
print(f"test.csv:     {test.shape},  日期范围: {test['date'].min()} ~ {test['date'].max()}")
print(f"calendar.csv: {calendar.shape}, 日期范围: {calendar['date'].min()} ~ {calendar['date'].max()}")
print(f"weather.csv:  {weather.shape}, 日期范围: {weather['date'].min()} ~ {weather['date'].max()}")
print(f"menu.csv:     {menu.shape}, 日期范围: {menu['date'].min()} ~ {menu['date'].max()}")

# ============================================================
# 2. 基础信息
# ============================================================
print("\n" + "=" * 60)
print("2. 基础信息")
print("=" * 60)

print("\n[train 数据类型]")
print(train.dtypes)
print("\n[train 缺失值]")
print(train.isnull().sum())
print("\n[train 描述统计]")
print(train.describe())

print("\n[唯一值统计]")
print(f"  日期数(train): {train['date'].nunique()}")
print(f"  日期数(test):  {test['date'].nunique()}")
print(f"  饭点: {sorted(train['meal'].unique())}")
print(f"  区域: {sorted(train['canteen_area'].unique())}")

# ============================================================
# 3. 目标变量分布分析
# ============================================================
print("\n" + "=" * 60)
print("3. 目标变量 volume 分布分析")
print("=" * 60)

print(f"\nvolume 总体统计:")
print(f"  mean={train['volume'].mean():.2f}, std={train['volume'].std():.2f}")
print(f"  min={train['volume'].min():.2f}, max={train['volume'].max():.2f}")
print(f"  median={train['volume'].median():.2f}")

# 按饭点统计
print("\n按饭点统计 volume:")
for meal in ['breakfast', 'lunch', 'dinner']:
    sub = train[train['meal'] == meal]['volume']
    print(f"  {meal:10s}: mean={sub.mean():7.2f}, std={sub.std():7.2f}, min={sub.min():7.2f}, max={sub.max():7.2f}")

# 按区域统计
print("\n按区域统计 volume:")
for area in sorted(train['canteen_area'].unique()):
    sub = train[train['canteen_area'] == area]['volume']
    print(f"  {area}: mean={sub.mean():7.2f}, std={sub.std():7.2f}, min={sub.min():7.2f}, max={sub.max():7.2f}")

# 交叉统计
print("\n按饭点×区域交叉统计 (mean volume):")
pivot = train.pivot_table(values='volume', index='canteen_area', columns='meal', aggfunc='mean')
print(pivot.round(2))

# ============================================================
# 4. Calendar 特征分析
# ============================================================
print("\n" + "=" * 60)
print("4. Calendar 特征分析")
print("=" * 60)

print(f"\ncalendar 列: {list(calendar.columns)}")
print(f"calendar 缺失值:\n{calendar.isnull().sum()}")
print(f"\nis_holiday 分布:\n{calendar['is_holiday'].value_counts()}")
print(f"\nis_exam_week 分布:\n{calendar['is_exam_week'].value_counts()}")
print(f"\nis_makeup_day 分布:\n{calendar['is_makeup_day'].value_counts()}")
print(f"\ncampus_event_level 分布:\n{calendar['campus_event_level'].value_counts()}")
print(f"\nis_weekend 分布:\n{calendar['is_weekend'].value_counts()}")

# 节假日 volume 对比
train_cal = train.merge(calendar, on='date', how='left')
print("\n节假日 vs 非节假日 volume 对比:")
print(train_cal.groupby('is_holiday')['volume'].describe())

# 考试周 volume 对比
print("\n考试周 vs 非考试周 volume 对比:")
print(train_cal.groupby('is_exam_week')['volume'].describe())

# 周末 volume 对比
print("\n周末 vs 工作日 volume 对比:")
print(train_cal.groupby('is_weekend')['volume'].describe())

# ============================================================
# 5. Weather 特征分析
# ============================================================
print("\n" + "=" * 60)
print("5. Weather 特征分析")
print("=" * 60)

print(f"\nweather 列: {list(weather.columns)}")
print(f"weather 缺失值:\n{weather.isnull().sum()}")
print(f"\nweather 类别分布:\n{weather['weather'].value_counts()}")
print(f"\nrain_level 分布:\n{weather['rain_level'].value_counts()}")

train_w = train.merge(weather, on=['date', 'meal'], how='left')
print("\n天气类别 vs volume:")
print(train_w.groupby('weather')['volume'].describe())

# ============================================================
# 6. Menu 特征分析
# ============================================================
print("\n" + "=" * 60)
print("6. Menu 特征分析")
print("=" * 60)

print(f"\nmenu 列: {list(menu.columns)}")
print(f"menu 缺失值:\n{menu.isnull().sum()}")
print(f"\nmenu_type 分布:\n{menu['menu_type'].value_counts()}")
print(f"\nmenu_popularity 分布:\n{menu['menu_popularity'].value_counts()}")
print(f"\nis_promotion 分布:\n{menu['is_promotion'].value_counts()}")

train_m = train.merge(menu, on=['date', 'meal', 'canteen_area'], how='left')
print("\n菜单类型 vs volume:")
print(train_m.groupby('menu_type')['volume'].describe())

print("\n促销 vs volume:")
print(train_m.groupby('is_promotion')['volume'].describe())

# ============================================================
# 7. 时间序列可视化
# ============================================================
print("\n" + "=" * 60)
print("7. 时间序列趋势可视化")
print("=" * 60)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
areas = sorted(train['canteen_area'].unique())

for idx, area in enumerate(areas):
    ax = axes[idx // 2][idx % 2]
    for meal in ['breakfast', 'lunch', 'dinner']:
        sub = train[(train['canteen_area'] == area) & (train['meal'] == meal)].copy()
        sub = sub.sort_values('date')
        ax.plot(range(len(sub)), sub['volume'].values, alpha=0.7, label=meal, linewidth=0.8)
    ax.set_title(f'Area {area} - Volume over time')
    ax.legend()
    ax.set_xlabel('Time index')
    ax.set_ylabel('Volume')

plt.tight_layout()
plt.savefig('eda_time_series.png', dpi=100)
plt.close()
print("时间序列图已保存: eda_time_series.png")

# ============================================================
# 8. 异常值检测
# ============================================================
print("\n" + "=" * 60)
print("8. 异常值检测 (3-sigma)")
print("=" * 60)

for area in sorted(train['canteen_area'].unique()):
    for meal in ['breakfast', 'lunch', 'dinner']:
        sub = train[(train['canteen_area'] == area) & (train['meal'] == meal)]['volume']
        mean, std = sub.mean(), sub.std()
        anomalies = sub[(sub < mean - 3*std) | (sub > mean + 3*std)]
        if len(anomalies) > 0:
            print(f"  {area} {meal}: {len(anomalies)} 异常值 (mean={mean:.1f}, std={std:.1f})")

# ============================================================
# 9. 关键发现汇总
# ============================================================
print("\n" + "=" * 60)
print("9. EDA 关键发现")
print("=" * 60)

print(f"""
数据规模:
  - 训练集: {len(train)} 行, {train['date'].nunique()} 天
  - 测试集: {len(test)} 行, {test['date'].nunique()} 天

目标变量:
  - 均值={train['volume'].mean():.2f}, 标准差={train['volume'].std():.2f}
  - lunch > dinner > breakfast  (午餐量最大)
  - A04 / A01 区域量最大, A05 最小

关键特征:
  - 节假日影响明显 (is_holiday)
  - 考试周影响 volume
  - 周末 volume 与工作日不同
  - 天气 (weather, temperature, rain) 可能影响
  - 菜单类型和热度影响 volume
  - 促销活动 (is_promotion) 可能提升 volume

时间序列特征:
  - 存在明显的星期周期性
  - 不同饭点的模式不同
  - 存在长期趋势 (如学期初 vs 学期末)
""")

print("EDA 完成!")
