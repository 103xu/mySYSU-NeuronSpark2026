"""完整训练+预测脚本——修复GPU兼容性 + 直接生成 submission"""
import numpy as np
import pandas as pd
import json
import warnings
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")

DATA = r"C:\Users\32010\ai-competition\04\2a9fa6fc-563a-43b9-b8bb-1301289bb22d"
OUTPUT = r"C:\Users\32010\ai-competition\04"


def safe_spearman(y_true, y_pred):
    try:
        corr, _ = spearmanr(y_true, y_pred)
        return max(0.0, corr) if not np.isnan(corr) else 0.0
    except:
        return 0.0


def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    spr = safe_spearman(y_true, y_pred)
    score = 500 * max(0, 1 - rmse / 0.80) + 250 * spr + 50
    print(f"  {label}: RMSE={rmse:.5f}, Spearman={spr:.5f}, score={score:.2f}")
    return rmse, spr, score


def main():
    print("=" * 60)
    print("加载数据")
    print("=" * 60)

    # Load features
    feats = np.load(f"{OUTPUT}/X_train_features.npz")
    X = feats["X"].astype(np.float64)
    feats_test = np.load(f"{OUTPUT}/X_test_features.npz")
    X_test = feats_test["X"].astype(np.float64)

    df = pd.read_csv(f"{DATA}/train.csv")
    y = df["target"].values.astype(np.float64)
    test_df = pd.read_csv(f"{DATA}/test.csv")

    with open(f"{DATA}/scaffold_split.json") as f:
        split = json.load(f)

    print(f"X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")

    # Clean
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    X_test = np.clip(X_test, -1e6, 1e6)

    # Variance filter
    sel = VarianceThreshold(threshold=0.0)
    X = sel.fit_transform(X)
    X_test = sel.transform(X_test)
    print(f"After variance filter: {X.shape[1]} features")

    # Build scaffold folds
    valid_ids = set(split["valid_ids"])
    df_ids = df["id"].tolist()
    valid_mask = np.array([(iid in valid_ids) for iid in df_ids])
    all_idx = np.arange(len(df))
    non_valid_idx = all_idx[~valid_mask]
    valid_idx = all_idx[valid_mask]

    folds = [(non_valid_idx, valid_idx)]
    kf = KFold(n_splits=5, shuffle=True, random_state=202604)
    for train_sub, val_sub in kf.split(non_valid_idx):
        folds.append((non_valid_idx[train_sub], non_valid_idx[val_sub]))

    print(f"Scaffold CV: {len(folds)} folds")

    oof_dict = {}
    test_dict = {}

    # ============================================================
    # 1. CatBoost (GPU)
    # ============================================================
    print("\n" + "=" * 50)
    print("训练 CatBoost (GPU)")
    print("=" * 50)

    cb_params = {
        "iterations": 3000,
        "depth": 7,
        "learning_rate": 0.02,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.5,
        "bagging_temperature": 0.5,
        "od_type": "Iter",
        "od_wait": 200,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "GPU",
        "devices": "0",
    }

    cat_oof = np.zeros(len(y))
    cat_test = np.zeros(len(X_test))

    for i, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = cb.CatBoostRegressor(**cb_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        cat_oof[val_idx] = model.predict(X_val)
        cat_test += model.predict(X_test) / len(folds)
        evaluate(y_val, cat_oof[val_idx], f"CAT fold {i+1}")

    evaluate(y, cat_oof, "CAT OOF")
    oof_dict["cat"] = cat_oof
    test_dict["cat"] = cat_test

    # ============================================================
    # 2. XGBoost (CPU - 兼容)
    # ============================================================
    print("\n" + "=" * 50)
    print("训练 XGBoost (CPU)")
    print("=" * 50)

    xgb_params = {
        "n_estimators": 3000,
        "max_depth": 6,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.01,
        "min_child_weight": 5,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu",
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": 200,
    }

    xgb_oof = np.zeros(len(y))
    xgb_test = np.zeros(len(X_test))

    for i, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_oof[val_idx] = model.predict(X_val)
        xgb_test += model.predict(X_test) / len(folds)
        evaluate(y_val, xgb_oof[val_idx], f"XGB fold {i+1}")

    evaluate(y, xgb_oof, "XGB OOF")
    oof_dict["xgb"] = xgb_oof
    test_dict["xgb"] = xgb_test

    # ============================================================
    # 3. LightGBM (CPU - 兼容)
    # ============================================================
    print("\n" + "=" * 50)
    print("训练 LightGBM (CPU)")
    print("=" * 50)

    lgb_params = {
        "n_estimators": 3000,
        "max_depth": 7,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "num_leaves": 63,
        "objective": "regression",
        "metric": "rmse",
        "random_state": 42,
        "verbosity": -1,
    }

    lgb_oof = np.zeros(len(y))
    lgb_test = np.zeros(len(X_test))

    for i, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        lgb_oof[val_idx] = model.predict(X_val)
        lgb_test += model.predict(X_test) / len(folds)
        evaluate(y_val, lgb_oof[val_idx], f"LGB fold {i+1}")

    evaluate(y, lgb_oof, "LGB OOF")
    oof_dict["lgb"] = lgb_oof
    test_dict["lgb"] = lgb_test

    # ============================================================
    # 4. Ridge
    # ============================================================
    print("\n" + "=" * 50)
    print("训练 Ridge 回归")
    print("=" * 50)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    ridge_oof = np.zeros(len(y))
    ridge_test = np.zeros(len(X_test))

    for i, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = Ridge(alpha=100.0)
        model.fit(X_tr, y_tr)
        ridge_oof[val_idx] = model.predict(X_val)
        ridge_test += model.predict(X_test_scaled) / len(folds)

    evaluate(y, ridge_oof, "Ridge OOF")
    oof_dict["ridge"] = ridge_oof
    test_dict["ridge"] = ridge_test

    # ============================================================
    # 5. 加权集成
    # ============================================================
    print("\n" + "=" * 50)
    print("集成预测")
    print("=" * 50)

    # 基于 OOF 表现计算权重 (RMSE 越小权重越大)
    weights = {}
    for name, oof in oof_dict.items():
        rmse = np.sqrt(mean_squared_error(y, oof))
        w = 1.0 / (rmse ** 2)  # inverse variance weighting
        weights[name] = w
        print(f"  {name}: RMSE={rmse:.5f}, weight={w:.4f}")

    total_w = sum(weights.values())
    for name in weights:
        weights[name] /= total_w
        print(f"  {name}: normalized weight={weights[name]:.4f}")

    # Weighted ensemble
    ensemble_test = np.zeros(len(X_test))
    for name, w in weights.items():
        ensemble_test += w * test_dict[name]

    # Simple average for comparison
    avg_test = np.mean([test_dict[name] for name in test_dict], axis=0)

    # Ridge stacking
    X_meta = np.column_stack([oof_dict[name] for name in ["cat", "xgb", "lgb", "ridge"]])
    X_meta_test = np.column_stack([test_dict[name] for name in ["cat", "xgb", "lgb", "ridge"]])

    stack_oof = np.zeros(len(y))
    for tr_idx, val_idx in folds:
        stack_model = Ridge(alpha=1.0)
        stack_model.fit(X_meta[tr_idx], y[tr_idx])
        stack_oof[val_idx] = stack_model.predict(X_meta[val_idx])

    stack_model = Ridge(alpha=1.0)
    stack_model.fit(X_meta, y)
    stack_test = stack_model.predict(X_meta_test)
    print(f"\nStacking coefs: {dict(zip(['cat','xgb','lgb','ridge'], stack_model.coef_))}")

    evaluate(y, stack_oof, "Stack OOF")

    # ============================================================
    # 保存结果
    # ============================================================
    print("\n" + "=" * 50)
    print("保存结果")
    print("=" * 50)

    submissions = {
        "cat": test_dict["cat"],
        "xgb": test_dict["xgb"],
        "lgb": test_dict["lgb"],
        "ridge": test_dict["ridge"],
        "weighted": ensemble_test,
        "average": avg_test,
        "stack": stack_test,
    }

    best_so_far = None
    for name, preds in submissions.items():
        sub = pd.DataFrame({"id": test_df["id"], "prediction": preds})
        sub.to_csv(f"{OUTPUT}/pred_{name}.csv", index=False)
        print(f"  pred_{name}.csv")

    # 使用 stacking 结果作为最终提交
    final_pred = stack_test
    sub = pd.DataFrame({"id": test_df["id"], "prediction": final_pred})
    sub.to_csv(f"{OUTPUT}/results.csv", index=False)
    print(f"\n  results.csv (stacking ensemble) saved")

    # 格式检查
    print("\n格式检查...")
    import subprocess as sp
    check_script = f"{DATA}/tools/check_format.py"
    test_csv = f"{DATA}/test.csv"
    result = sp.run(
        [r"C:\Users\32010\AppData\Local\Programs\Python\Python312\python.exe",
         check_script, f"{OUTPUT}/results.csv", "--test-csv", test_csv],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # 打包
    print("\n打包提交...")
    import zipfile
    import os
    zip_path = f"{OUTPUT}/NS-2026-04-answer.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(f"{OUTPUT}/results.csv", "results.csv")
    print(f"  {zip_path} ({os.path.getsize(zip_path)} bytes)")

    print("\n完成!")


if __name__ == "__main__":
    main()
