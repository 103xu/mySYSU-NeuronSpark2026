"""Step 3-9 v4: 无 lag 模型 — 仅用静态+滚动特征，避免迭代误差累积"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import time
warnings.filterwarnings('ignore')

from features import (
    load_raw_data, build_static_features,
    compute_full_features, get_feature_columns,
    compute_rolling_from_pool
)

SEED = 42
np.random.seed(SEED)


def compute_features_no_lag(df_rows, volume_pool):
    """仅计算滚动特征（不使用 lag），所有特征始终可用"""
    return compute_rolling_from_pool(df_rows, volume_pool)


def compute_metrics(y_true, y_pred, prefix=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    print(f"  {prefix}MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def main():
    print("=" * 60)
    print("NS-2026-01 v4 — 无 Lag 特征模型")
    print("=" * 60)

    # 1. 加载数据
    train_raw, test_raw, calendar, weather, menu = load_raw_data()

    # 2. 构建静态特征
    print("\n构建静态特征...")
    train_static, test_static, encoders = build_static_features(
        train_raw, test_raw, calendar, weather, menu
    )

    # 3. 构建训练集特征（只用 rolling，无 lag）
    volume_pool = train_raw[['date', 'meal', 'canteen_area', 'volume']].copy()
    train_full = compute_features_no_lag(train_static, volume_pool)

    # 获取特征列（排除 lag 列）
    all_feature_cols = get_feature_columns(train_full)
    feature_cols = [c for c in all_feature_cols if 'lag' not in c]

    # 也加入静态特征中不在 lag/rolling 里的特征
    static_num_cols = train_static.select_dtypes(include=[np.number]).columns.tolist()
    for col in static_num_cols:
        if col not in feature_cols and col in train_full.columns:
            feature_cols.append(col)

    # 合并，去重，排除非特征列
    feature_cols = list(set(feature_cols))
    feature_cols = [c for c in feature_cols if c not in ['volume', 'date', 'meal', 'canteen_area', 'weather', 'menu_type']]

    print(f"无 Lag 特征数: {len(feature_cols)}")

    # ================================================================
    # 时间序列验证：前470天训练，后90天验证
    # ================================================================
    print("\n" + "=" * 60)
    print("时间序列验证 (前470天训练, 后90天验证)")
    print("=" * 60)

    all_dates = sorted(train_raw['date'].unique())
    split_idx = len(all_dates) - 90
    train_dates = all_dates[:split_idx]
    val_dates = all_dates[split_idx:]

    train_mask = train_full['date'].isin(train_dates)
    val_mask = train_full['date'].isin(val_dates)

    X_train = train_full[train_mask][feature_cols].fillna(0)
    y_train = train_full[train_mask]['volume'].values
    X_val = train_full[val_mask][feature_cols].fillna(0)
    y_val = train_full[val_mask]['volume'].values

    print(f"  训练: {X_train.shape}, 验证: {X_val.shape}")

    # ================================================================
    # Baseline
    # ================================================================
    print("\n" + "=" * 60)
    print("Baseline (无 lag)")
    print("=" * 60)

    # 全局均值
    global_mean = y_train.mean()
    print(f"  全局均值 MAE: {mean_absolute_error(y_val, np.full_like(y_val, global_mean)):.2f}")

    # 分组均值
    val_df = train_full[val_mask][['date', 'meal', 'canteen_area', 'volume']].copy()
    train_grouped = train_full[train_mask].groupby(['meal', 'canteen_area'])['volume'].mean()
    y_pred_grp = []
    for _, row in val_df.iterrows():
        y_pred_grp.append(train_grouped.get((row['meal'], row['canteen_area']), global_mean))
    print(f"  分组均值 MAE: {mean_absolute_error(y_val, np.array(y_pred_grp)):.2f}")

    # 同 weekday 均值
    val_df_full = train_full[val_mask].copy()
    train_wday = train_full[train_mask].groupby(['weekday', 'meal', 'canteen_area'])['volume'].mean()
    y_pred_wday = []
    for _, row in val_df_full.iterrows():
        y_pred_wday.append(train_wday.get((row['weekday'], row['meal'], row['canteen_area']), global_mean))
    print(f"  同星期均值 MAE: {mean_absolute_error(y_val, np.array(y_pred_wday)):.2f}")

    # ================================================================
    # 模型训练
    # ================================================================
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    configs = [
        ('XGB-depth5-300r', 'xgboost',
         {'max_depth': 5, 'learning_rate': 0.05, 'min_child_weight': 10,
          'reg_lambda': 5.0, 'reg_alpha': 1.0, 'subsample': 0.8, 'colsample_bytree': 0.8}, 300),
        ('XGB-depth6-500r', 'xgboost',
         {'max_depth': 6, 'learning_rate': 0.03, 'min_child_weight': 5,
          'reg_lambda': 1.0, 'reg_alpha': 0.5, 'subsample': 0.8, 'colsample_bytree': 0.8}, 500),
        ('XGB-depth4-200r', 'xgboost',
         {'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 20,
          'reg_lambda': 10.0, 'reg_alpha': 2.0, 'subsample': 0.7, 'colsample_bytree': 0.7}, 200),
        ('LGB-leaves31-300r', 'lightgbm',
         {'num_leaves': 31, 'learning_rate': 0.05, 'min_data_in_leaf': 30,
          'lambda_l1': 1.0, 'lambda_l2': 5.0, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
          'bagging_freq': 5}, 300),
        ('LGB-leaves63-500r', 'lightgbm',
         {'num_leaves': 63, 'learning_rate': 0.03, 'min_data_in_leaf': 20,
          'lambda_l1': 0.5, 'lambda_l2': 1.0, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5}, 500),
    ]

    experiments = []
    for name, model_type, params, n_rounds in configs:
        print(f"\n--- {name} ---")
        if model_type == 'xgboost':
            default_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'verbosity': 0,
                'random_state': SEED,
                'n_jobs': -1,
            }
            default_params.update(params)
            model = xgb.train(
                default_params,
                xgb.DMatrix(X_train, label=y_train),
                num_boost_round=n_rounds,
                verbose_eval=False,
            )
            y_pred = model.predict(xgb.DMatrix(X_val))
        else:
            default_params = {
                'objective': 'regression',
                'metric': 'mae',
                'verbose': -1,
                'random_state': SEED,
                'n_jobs': -1,
            }
            default_params.update(params)
            model = lgb.train(
                default_params,
                lgb.Dataset(X_train, label=y_train),
                num_boost_round=n_rounds,
            )
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        mae_val = mean_absolute_error(y_val, y_pred)
        print(f"  Val MAE: {mae_val:.4f}")
        experiments.append((name, mae_val, model, model_type, params, n_rounds))

    # 汇总
    print("\n实验汇总:")
    for name, mae, _, _, _, rounds in sorted(experiments, key=lambda x: x[1]):
        print(f"  {name:25s}: MAE={mae:.2f} (rounds={rounds})")

    best = min(experiments, key=lambda x: x[1])
    best_name, best_val_mae, _, best_type, best_params, best_rounds = best
    print(f"\n最佳: {best_name}, Val MAE={best_val_mae:.2f}")

    # ================================================================
    # 全量训练并预测测试集
    # ================================================================
    print("\n" + "=" * 60)
    print("最终模型：全量560天训练 + 预测测试集")
    print("=" * 60)

    X_full = train_full[feature_cols].fillna(0)
    y_full = train_full['volume'].values

    if best_type == 'xgboost':
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'verbosity': 0,
            'random_state': SEED,
            'n_jobs': -1,
        }
        default_params.update(best_params)
        final_model = xgb.train(
            default_params,
            xgb.DMatrix(X_full, label=y_full),
            num_boost_round=best_rounds,
            verbose_eval=100,
        )
    else:
        default_params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbose': -1,
            'random_state': SEED,
            'n_jobs': -1,
        }
        default_params.update(best_params)
        final_model = lgb.train(
            default_params,
            lgb.Dataset(X_full, label=y_full),
            num_boost_round=best_rounds,
            callbacks=[lgb.log_evaluation(100)],
        )

    # 测试集特征（无需迭代预测！所有特征始终可用）
    test_full = compute_features_no_lag(test_static, volume_pool)
    X_test = test_full[feature_cols].fillna(0)

    if best_type == 'xgboost':
        y_test_pred = final_model.predict(xgb.DMatrix(X_test))
    else:
        y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)

    # 生成提交文件
    submission = test_raw[['date', 'meal', 'canteen_area']].copy()
    submission['volume'] = np.clip(y_test_pred, 0, None)

    submission.to_csv('results.csv', index=False)
    print(f"\nresults.csv: {len(submission)} 行")
    print(f"volume: [{submission['volume'].min():.2f}, {submission['volume'].max():.2f}]")

    assert list(submission.columns) == ['date', 'meal', 'canteen_area', 'volume']
    assert len(submission) == 1620
    assert submission['volume'].notna().all()
    assert (submission['volume'] >= 0).all()

    # 特征重要性
    print("\n特征重要性 Top 15:")
    if best_type == 'xgboost':
        imp = final_model.get_score(importance_type='gain')
        for feat, score in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {feat:30s}: {score:10.2f}")
    else:
        imp_arr = final_model.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({'f': feature_cols, 'imp': imp_arr}).nlargest(15, 'imp')
        for _, row in imp_df.iterrows():
            print(f"  {row['f']:30s}: {row['imp']:10.2f}")

    # 按维度验证
    print("\n验证集按维度分析:")
    val_result = train_full[val_mask][['date', 'meal', 'canteen_area', 'volume']].copy()
    if best_type == 'xgboost':
        val_result['pred'] = final_model.predict(xgb.DMatrix(X_val))
    else:
        val_result['pred'] = final_model.predict(X_val)

    for meal in ['breakfast', 'lunch', 'dinner']:
        sub = val_result[val_result['meal'] == meal]
        mae = mean_absolute_error(sub['volume'], sub['pred'])
        print(f"  {meal:10s}: MAE={mae:.2f}")

    for area in ['A01', 'A02', 'A03', 'A04', 'A05', 'A06']:
        sub = val_result[val_result['canteen_area'] == area]
        mae = mean_absolute_error(sub['volume'], sub['pred'])
        print(f"  {area}: MAE={mae:.2f}")

    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"验证集 MAE: {best_val_mae:.2f}")
    print(f"最终模型: {best_name}")
    print(f"策略: 无 Lag 特征, 避免迭代误差累积")
    print(f"下一步: python 136b479a-05b3-426d-b95d-1530d094f5be/tools/check_format.py results.csv --test-csv 136b479a-05b3-426d-b95d-1530d094f5be/test.csv")


if __name__ == "__main__":
    main()
