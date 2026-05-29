"""
predict_2.py — CatBoost 推理脚本
加载模型，生成 results.csv
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

DATA_DIR = "136b479a-05b3-426d-b95d-1530d094f5be"

# =========================
# 读取数据
# =========================
print("读取数据...")
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


# =========================
# 合并附加表（与训练时完全一致）
# =========================
def merge_features(df):
    df = df.merge(calendar_df, on="date", how="left")
    df = df.merge(weather_df, on=["date", "meal"], how="left")
    df = df.merge(menu_df, on=["date", "meal", "canteen_area"], how="left")
    return df


# 需要 train 来对齐列结构，但只取列名
train_df = merge_features(train_df)
test_df = merge_features(test_df)


# =========================
# 时间特征
# =========================
def build_time_features(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


train_df = build_time_features(train_df)
test_df = build_time_features(test_df)

# =========================
# 缺失值
# =========================
train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)

# =========================
# 特征列
# =========================
exclude_cols = ["date", "volume"]
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

# =========================
# 加载模型
# =========================
print("加载模型...")
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")
print(f"  模型已加载")

# =========================
# 预测
# =========================
print("推理中...")
pred = model.predict(test_df[feature_cols])
pred = np.maximum(pred, 0)

# =========================
# 生成提交文件
# =========================
submit_df = test_df[["date", "meal", "canteen_area"]].copy()
submit_df["volume"] = pred
submit_df["date"] = submit_df["date"].astype(str)

submit_df.to_csv("results.csv", index=False)

print(f"results.csv 已生成: {len(submit_df)} 行")
print(f"volume 范围: [{submit_df['volume'].min():.2f}, {submit_df['volume'].max():.2f}]")
print(submit_df.head())
