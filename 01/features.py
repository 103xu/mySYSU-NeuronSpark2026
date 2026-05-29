"""Step 2: 特征工程 v2 — 基于 volume_pool 的动态特征计算，支持迭代预测"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "136b479a-05b3-426d-b95d-1530d094f5be"


def load_raw_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")
    weather = pd.read_csv(f"{DATA_DIR}/weather.csv")
    menu = pd.read_csv(f"{DATA_DIR}/menu.csv")
    return train, test, calendar, weather, menu


def merge_static_features(df, calendar, weather, menu):
    """合并外部静态特征表（calendar/weather/menu）"""
    df = df.merge(calendar, on='date', how='left')
    df = df.merge(weather, on=['date', 'meal'], how='left')
    df = df.merge(menu, on=['date', 'meal', 'canteen_area'], how='left')
    return df


def encode_categorical(df, encoders=None, fit=True):
    cat_cols = ['meal', 'canteen_area', 'weather', 'menu_type']
    if encoders is None:
        encoders = {}
    for col in cat_cols:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                df[col + '_enc'] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                le = encoders[col]
                known = set(le.classes_)
                df[col + '_enc'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known else -1
                )
    return df, encoders


def add_time_features(df):
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    df['year'] = df['date_dt'].dt.year
    df['month'] = df['date_dt'].dt.month
    df['day'] = df['date_dt'].dt.day
    df['day_of_year'] = df['date_dt'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    meal_order = {'breakfast': 0, 'lunch': 1, 'dinner': 2}
    df['meal_order'] = df['meal'].map(meal_order)
    df['semester_week_sin'] = np.sin(2 * np.pi * df['semester_week'] / 20)
    df['semester_week_cos'] = np.cos(2 * np.pi * df['semester_week'] / 20)
    df = df.drop(columns=['date_dt'])
    return df


def add_interaction_features(df):
    df = df.copy()
    df['weather_meal_interact'] = df['weather_enc'] * 10 + df['meal_order']
    df['area_meal_interact'] = df['canteen_area_enc'] * 10 + df['meal_order']
    df['popularity_promo'] = df['menu_popularity'] * (1 + df['is_promotion'])
    df['exam_meal'] = df['is_exam_week'] * df['meal_order']
    df['holiday_meal'] = df['is_holiday'] * df['meal_order']
    df['weekend_meal'] = df['is_weekend'] * df['meal_order']
    df['event_meal'] = df['campus_event_level'] * df['meal_order']
    return df


def build_static_features(train_raw, test_raw, calendar, weather, menu):
    """构建静态特征（不依赖历史 volume，始终可用）"""
    train = merge_static_features(train_raw, calendar, weather, menu)
    test = merge_static_features(test_raw, calendar, weather, menu)
    train, encoders = encode_categorical(train, encoders=None, fit=True)
    test, _ = encode_categorical(test, encoders=encoders, fit=False)
    train = add_time_features(train)
    test = add_time_features(test)
    train = add_interaction_features(train)
    test = add_interaction_features(test)
    return train, test, encoders


def compute_lag_from_pool(df_rows, volume_pool, lag_days=[1, 7, 14, 21, 28]):
    """
    从 volume_pool 计算滞后特征。
    volume_pool: DataFrame with columns [date, meal, canteen_area, volume]
    包含训练集原始数据 + 已预测的测试集数据
    """
    pool = volume_pool.copy()
    pool['date_dt'] = pd.to_datetime(pool['date'])

    result = df_rows.copy()
    result['date_dt'] = pd.to_datetime(result['date'])

    for lag in lag_days:
        lag_pool = pool.copy()
        lag_pool['date_dt'] = lag_pool['date_dt'] + pd.Timedelta(days=lag)
        lag_pool = lag_pool.rename(columns={'volume': f'volume_lag_{lag}d'})

        result = result.merge(
            lag_pool[['date_dt', 'meal', 'canteen_area', f'volume_lag_{lag}d']],
            on=['date_dt', 'meal', 'canteen_area'], how='left'
        )

    result = result.drop(columns=['date_dt'])
    return result


def compute_rolling_from_pool(df_rows, volume_pool, windows=[3, 7, 14, 30]):
    """
    从 volume_pool 计算滚动窗口特征。
    对每个目标日期，取 volume_pool 中 <= 该日期的最近窗口统计量。
    """
    pool = volume_pool.copy()
    pool['date'] = pd.to_datetime(pool['date'])
    pool = pool.sort_values(['canteen_area', 'meal', 'date'])

    result = df_rows.copy()
    result['date_dt'] = pd.to_datetime(result['date'])

    for area in pool['canteen_area'].unique():
        for meal_type in pool['meal'].unique():
            mask_pool = (pool['canteen_area'] == area) & (pool['meal'] == meal_type)
            mask_res = (result['canteen_area'] == area) & (result['meal'] == meal_type)

            pool_sub = pool[mask_pool].set_index('date')['volume'].sort_index()
            res_dates = result.loc[mask_res, 'date_dt']

            if len(pool_sub) == 0:
                continue

            for w in windows:
                roll_mean = pool_sub.rolling(window=w, min_periods=1).mean()
                roll_std = pool_sub.rolling(window=w, min_periods=1).std().fillna(0)
                roll_min = pool_sub.rolling(window=w, min_periods=1).min()
                roll_max = pool_sub.rolling(window=w, min_periods=1).max()

                for col_name, roll_series in [
                    (f'rolling_mean_{w}d', roll_mean),
                    (f'rolling_std_{w}d', roll_std),
                    (f'rolling_min_{w}d', roll_min),
                    (f'rolling_max_{w}d', roll_max),
                ]:
                    vals = []
                    for d in res_dates:
                        past = roll_series[roll_series.index <= d]
                        if len(past) > 0:
                            vals.append(past.iloc[-1])
                        else:
                            vals.append(np.nan)
                    result.loc[mask_res, col_name] = vals

    result = result.drop(columns=['date_dt'])
    return result


def compute_full_features(df_rows, volume_pool):
    """对一组行计算完整的动态特征（lag + rolling）"""
    df = compute_lag_from_pool(df_rows, volume_pool)
    df = compute_rolling_from_pool(df, volume_pool)
    return df


def get_feature_columns(train_feat):
    """获取模型特征列（排除非特征列）"""
    exclude = ['date', 'volume', 'weather', 'menu_type', 'meal', 'canteen_area']
    cols = [c for c in train_feat.columns if c not in exclude]
    cols = train_feat[cols].select_dtypes(include=[np.number]).columns.tolist()
    return cols


if __name__ == "__main__":
    train_raw, test_raw, calendar, weather, menu = load_raw_data()
    train_static, test_static, encoders = build_static_features(train_raw, test_raw, calendar, weather, menu)

    # 在训练集上计算动态特征（使用 train_raw 作为 volume_pool）
    train_full = compute_full_features(train_static, train_raw[['date', 'meal', 'canteen_area', 'volume']])
    print(f"训练集完整特征: {train_full.shape}")
    print(f"特征列数: {len(get_feature_columns(train_full))}")
