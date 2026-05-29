"""
train_2.py — CatBoost 训练脚本
基于参考代码重构，修复天气合并、添加时间序列验证、特征工程
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
import time

DATA_DIR = "136b479a-05b3-426d-b95d-1530d094f5be"

# =========================
# 读取数据
# =========================
print("=" * 60)
print("读取数据")
print("=" * 60)

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
calendar_df = pd.read_csv(f"{DATA_DIR}/calendar.csv")
weather_df = pd.read_csv(f"{DATA_DIR}/weather.csv")
menu_df = pd.read_csv(f"{DATA_DIR}/menu.csv")

# 日期处理
train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])
calendar_df["date"] = pd.to_datetime(calendar_df["date"])
weather_df["date"] = pd.to_datetime(weather_df["date"])
menu_df["date"] = pd.to_datetime(menu_df["date"])

# 排序
train_df = train_df.sort_values(["canteen_area", "meal", "date"]).reset_index(drop=True)

print(f"train: {train_df.shape}, test: {test_df.shape}")
print(f"日期范围 — train: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
print(f"日期范围 — test:  {test_df['date'].min().date()} ~ {test_df['date'].max().date()}")

# =========================
# 合并附加表
# =========================
print("\n" + "=" * 60)
print("合并特征表")
print("=" * 60)


def merge_features(df):
    # calendar: 按 date
    df = df.merge(calendar_df, on="date", how="left")
    # weather: 按 date + meal（参考代码只按 date 合并是 bug，修正为 date+meal）
    df = df.merge(weather_df, on=["date", "meal"], how="left")
    # menu: 按 date + meal + canteen_area
    df = df.merge(menu_df, on=["date", "meal", "canteen_area"], how="left")
    return df


train_df = merge_features(train_df)
test_df = merge_features(test_df)
print(f"合并后 train: {train_df.shape}, test: {test_df.shape}")

# =========================
# 时间特征
# =========================
print("\n" + "=" * 60)
print("时间特征工程")
print("=" * 60)


def build_time_features(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear

    # 周期编码
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


train_df = build_time_features(train_df)
test_df = build_time_features(test_df)
print(f"时间特征后 train: {train_df.shape}")

# =========================
# 缺失值处理
# =========================
train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)

# =========================
# 特征列 & 类别特征
# =========================
exclude_cols = ["date", "volume"]
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

cat_features = [col for col in feature_cols if str(train_df[col].dtype) in ("object", "string", "str")]
print(f"\n  数值特征: {len(feature_cols) - len(cat_features)}")
print(f"  类别特征: {len(cat_features)} → {cat_features}")

# =========================
# 时间序列切分（严格按时间顺序）
# =========================
print("\n" + "=" * 60)
print("时间序列验证集划分 (前470天训练, 后90天验证)")
print("=" * 60)

all_dates = sorted(train_df["date"].unique())
split_idx = len(all_dates) - 90
train_dates = all_dates[:split_idx]
val_dates = all_dates[split_idx:]

train_mask = train_df["date"].isin(train_dates)
val_mask = train_df["date"].isin(val_dates)

X_train = train_df[train_mask][feature_cols]
y_train = train_df[train_mask]["volume"]
X_val = train_df[val_mask][feature_cols]
y_val = train_df[val_mask]["volume"]

print(f"  训练: {X_train.shape}, 验证: {X_val.shape}")
print(f"  训练日期: {train_dates[0].date()} ~ {train_dates[-1].date()}")
print(f"  验证日期: {val_dates[0].date()} ~ {val_dates[-1].date()}")

# =========================
# 训练 CatBoost
# =========================
print("\n" + "=" * 60)
print("训练 CatBoost")
print("=" * 60)

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    loss_function="MAE",
    eval_metric="MAE",
    early_stopping_rounds=200,
    cat_features=cat_features,
    verbose=100,
    thread_count=-1,
)

t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
)
elapsed = time.time() - t0

print(f"\n  训练完成, 耗时: {elapsed:.1f}s")
print(f"  最佳迭代: {model.get_best_iteration()}")

# =========================
# 验证集评估
# =========================
print("\n" + "=" * 60)
print("验证集评估")
print("=" * 60)

y_pred = model.predict(X_val)
y_pred = np.maximum(y_pred, 0)
mae = mean_absolute_error(y_val, y_pred)
print(f"  Overall MAE: {mae:.4f}")

# 按饭点
for meal in ["breakfast", "lunch", "dinner"]:
    m = train_df[val_mask]["meal"] == meal
    if m.sum() > 0:
        mae_m = mean_absolute_error(y_val[m], y_pred[m])
        print(f"  {meal:10s}: MAE={mae_m:.2f}")

# 按区域
for area in sorted(train_df["canteen_area"].unique()):
    m = train_df[val_mask]["canteen_area"] == area
    if m.sum() > 0:
        mae_a = mean_absolute_error(y_val[m], y_pred[m])
        print(f"  {area}: MAE={mae_a:.2f}")

# =========================
# 特征重要性
# =========================
print("\n" + "=" * 60)
print("Top 15 特征重要性")
print("=" * 60)

importance = model.get_feature_importance()
imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance,
}).sort_values("importance", ascending=False)

for i, row in imp_df.head(15).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:10.2f}")

# =========================
# 全量训练最终模型
# =========================
print("\n" + "=" * 60)
print("全量训练最终模型 (560天)")
print("=" * 60)

X_full = train_df[feature_cols]
y_full = train_df["volume"]

final_model = CatBoostRegressor(
    iterations=model.get_best_iteration() + 50,  # 数据多了，稍微多加一些迭代
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    loss_function="MAE",
    cat_features=cat_features,
    verbose=100,
    thread_count=-1,
)

final_model.fit(X_full, y_full)
print(f"  全量训练完成")

# =========================
# 保存模型
# =========================
model_path = "catboost_model.cbm"
final_model.save_model(model_path)
print(f"\n模型已保存: {model_path}")
print(f"  验证 MAE: {mae:.4f}")
print(f"  最佳迭代: {model.get_best_iteration()}")
print("完成!")
