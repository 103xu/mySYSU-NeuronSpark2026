"""模型训练：多模型 + scaffold CV + 超参数搜索"""
import numpy as np
import pandas as pd
import json
import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
warnings.filterwarnings("ignore")

DATA = r"C:\Users\32010\ai-competition\04\2a9fa6fc-563a-43b9-b8bb-1301289bb22d"
OUTPUT = r"C:\Users\32010\ai-competition\04"


def load_all():
    print("加载数据...")
    feats = np.load(f"{OUTPUT}/X_train_features.npz")
    X = feats["X"].astype(np.float32)
    feats_test = np.load(f"{OUTPUT}/X_test_features.npz")
    X_test = feats_test["X"].astype(np.float32)
    df = pd.read_csv(f"{DATA}/train.csv")
    y = df["target"].values.astype(np.float32)
    test_df = pd.read_csv(f"{DATA}/test.csv")

    with open(f"{DATA}/scaffold_split.json") as f:
        split = json.load(f)

    print(f"X: {X.shape}, y: {y.shape}")
    print(f"X_test: {X_test.shape}")
    return X, y, X_test, df, test_df, split


def safe_spearman(y_true, y_pred):
    try:
        corr, _ = spearmanr(y_true, y_pred)
        return max(0.0, corr) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    spr = safe_spearman(y_true, y_pred)
    raw = 500 * max(0, 1 - rmse / 0.80) + 250 * spr + 50
    print(f"  {label}: RMSE={rmse:.5f}, Spearman={spr:.5f}, score={raw:.2f}")
    return rmse, spr, raw


def remove_low_variance(X, X_test, threshold=0.0):
    sel = VarianceThreshold(threshold=threshold)
    X_new = sel.fit_transform(X)
    X_test_new = sel.transform(X_test)
    return X_new, X_test_new, sel.get_support()


def select_top_k(X, y, X_test, k=3000):
    """Mutual information feature selection."""
    if X.shape[1] <= k:
        return X, X_test, np.ones(X.shape[1], dtype=bool)
    mi = mutual_info_regression(X, y, random_state=42)
    idx = np.argsort(mi)[::-1][:k]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[idx] = True
    return X[:, mask], X_test[:, mask], mask


def get_scaffold_folds(df, split, n_folds=5):
    """Build scaffold-aware cross-validation folds."""
    valid_ids = set(split["valid_ids"])
    df_ids = df["id"].tolist()
    valid_mask = np.array([(iid in valid_ids) for iid in df_ids])

    # Get indices for non-valid and valid
    all_idx = np.arange(len(df))

    # For scaffold CV: use valid split + random splits on the rest
    folds = []

    # Fold 1: train on non-valid, validate on valid
    non_valid_idx = all_idx[~valid_mask]
    valid_idx = all_idx[valid_mask]
    folds.append((non_valid_idx, valid_idx))

    # Additional folds: split non-valid into k-fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=202604)
    for train_sub, val_sub in kf.split(non_valid_idx):
        fold_train = non_valid_idx[train_sub]
        fold_val = non_valid_idx[val_sub]
        folds.append((fold_train, fold_val))

    return folds


def train_xgb(X, y, X_test, folds, feature_mask=None):
    print("\n" + "=" * 50)
    print("训练 XGBoost")
    print("=" * 50)

    params = {
        "n_estimators": 2000,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.01,
        "min_child_weight": 5,
        "objective": "reg:squarederror",
        "tree_method": "gpu_hist",
        "device": "cuda",
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": 100,
    }

    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))

    for i, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / len(folds)

        if i < 3:
            evaluate(y_val, oof[val_idx], f"XGB fold {i+1}")

    evaluate(y, oof, "XGB OOF")
    return oof, test_preds, "xgb"


def train_lgb(X, y, X_test, folds):
    print("\n" + "=" * 50)
    print("训练 LightGBM")
    print("=" * 50)

    params = {
        "n_estimators": 2000,
        "max_depth": 7,
        "learning_rate": 0.03,
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
        "device": "cuda",
    }

    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))

    for i, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / len(folds)

        if i < 3:
            evaluate(y_val, oof[val_idx], f"LGB fold {i+1}")

    evaluate(y, oof, "LGB OOF")
    return oof, test_preds, "lgb"


def train_cat(X, y, X_test, folds):
    print("\n" + "=" * 50)
    print("训练 CatBoost")
    print("=" * 50)

    params = {
        "iterations": 2000,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.5,
        "bagging_temperature": 0.5,
        "od_type": "Iter",
        "od_wait": 100,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "GPU",
        "devices": "0",
    }

    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))

    for i, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            verbose=False,
        )
        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / len(folds)

        if i < 3:
            evaluate(y_val, oof[val_idx], f"CAT fold {i+1}")

    evaluate(y, oof, "CAT OOF")
    return oof, test_preds, "cat"


def train_ridge(X, y, X_test, folds):
    print("\n" + "=" * 50)
    print("训练 Ridge 回归")
    print("=" * 50)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))

    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    best_alpha = 1.0
    best_rmse = float("inf")

    # simple alpha search on first fold
    tr_idx, val_idx = folds[0]
    for a in alphas:
        m = Ridge(alpha=a)
        m.fit(X_scaled[tr_idx], y[tr_idx])
        p = m.predict(X_scaled[val_idx])
        rmse = np.sqrt(mean_squared_error(y[val_idx], p))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = a
    print(f"  best alpha: {best_alpha}")

    for i, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = Ridge(alpha=best_alpha)
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test_scaled) / len(folds)

    evaluate(y, oof, "Ridge OOF")
    return oof, test_preds, "ridge"


def train_mlp(X, y, X_test, folds):
    print("\n" + "=" * 50)
    print("训练 MLP")
    print("=" * 50)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))

    for i, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=64,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test_scaled) / len(folds)

        if i < 3:
            evaluate(y_val, oof[val_idx], f"MLP fold {i+1}")

    evaluate(y, oof, "MLP OOF")
    return oof, test_preds, "mlp"


def stack_ensemble(oof_dict, test_dict, y, folds):
    print("\n" + "=" * 50)
    print("Stacking Ensemble (Ridge)")
    print("=" * 50)

    model_names = list(oof_dict.keys())
    n = len(y)
    n_test = len(list(test_dict.values())[0])

    # Build meta-features
    X_meta = np.column_stack([oof_dict[name] for name in model_names])
    X_meta_test = np.column_stack([test_dict[name] for name in model_names])

    # Cross-validated meta-model to avoid overfitting
    meta_oof = np.zeros(n)
    meta_test = np.zeros(n_test)

    scaler = StandardScaler()

    for train_idx, val_idx in folds:
        X_tr, X_val = X_meta[train_idx], X_meta[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        meta = Ridge(alpha=1.0, positive=False)
        meta.fit(X_tr_s, y_tr)
        meta_oof[val_idx] = meta.predict(X_val_s)

    # final meta-model on all data
    X_meta_s = scaler.fit_transform(X_meta)
    final_meta = Ridge(alpha=1.0, positive=False)
    final_meta.fit(X_meta_s, y)
    meta_test = final_meta.predict(scaler.transform(X_meta_test))

    print("Meta-model coefficients:")
    for name, coef in zip(model_names, final_meta.coef_):
        print(f"  {name}: {coef:.4f}")

    evaluate(y, meta_oof, "Stack OOF")

    # Also try a weighted average baseline
    wavg_test = np.mean(X_meta_test, axis=1)
    print("\nSimple average baseline:")

    return meta_oof, meta_test


def main():
    X, y, X_test, df, test_df, split = load_all()

    # Feature preprocessing
    print("\n特征预处理...")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    X_test = np.clip(X_test, -1e6, 1e6)

    # Remove low variance
    X, X_test, var_mask = remove_low_variance(X, X_test, threshold=0.0)
    print(f"After variance filter: {X.shape[1]} features")

    # Get scaffold-aware folds
    folds = get_scaffold_folds(df, split, n_folds=5)
    print(f"Scaffold CV: {len(folds)} folds")
    for i, (tr, val) in enumerate(folds):
        print(f"  Fold {i+1}: train={len(tr)}, val={len(val)}")

    # Train models
    oof_dict = {}
    test_dict = {}

    for train_fn in [train_xgb, train_lgb, train_cat, train_ridge, train_mlp]:
        try:
            oof, test_pred, name = train_fn(X, y, X_test, folds)
            oof_dict[name] = oof
            test_dict[name] = test_pred
        except Exception as e:
            print(f"  {train_fn.__name__} failed: {e}")

    # Stacking ensemble
    if len(oof_dict) >= 2:
        ensemble_oof, ensemble_test = stack_ensemble(oof_dict, test_dict, y, folds)
        test_dict["ensemble"] = ensemble_test
        oof_dict["ensemble"] = ensemble_oof

    # Save predictions
    print("\n" + "=" * 50)
    print("保存预测结果")
    print("=" * 50)

    # best single model + ensemble
    for name, preds in test_dict.items():
        sub = pd.DataFrame({"id": test_df["id"], "prediction": preds})
        sub.to_csv(f"{OUTPUT}/pred_{name}.csv", index=False)
        print(f"  pred_{name}.csv saved")

    # Use ensemble prediction if available, otherwise best single
    if "ensemble" in test_dict:
        final_pred = test_dict["ensemble"]
    else:
        # fallback to xgb
        final_pred = test_dict.get("xgb", list(test_dict.values())[0])

    sub = pd.DataFrame({"id": test_df["id"], "prediction": final_pred})
    sub.to_csv(f"{OUTPUT}/results.csv", index=False)
    print(f"\n  results.csv saved")

    # also check format
    check_script = f"{DATA}/tools/check_format.py"
    test_csv = f"{DATA}/test.csv"
    print(f"\n格式检查:")
    import subprocess
    result = subprocess.run(
        [r"C:\Users\32010\AppData\Local\Programs\Python\Python312\python.exe",
         check_script, f"{OUTPUT}/results.csv", "--test-csv", test_csv],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


if __name__ == "__main__":
    main()
