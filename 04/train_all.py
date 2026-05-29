"""全模型训练：CatBoost(GPU) + XGBoost(CPU) + LightGBM(CPU) + Ridge + MLP → Stacking"""
import numpy as np
import pandas as pd
import json, os, zipfile
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

DATA = r"C:\Users\32010\ai-competition\04\2a9fa6fc-563a-43b9-b8bb-1301289bb22d"
OUTPUT = r"C:\Users\32010\ai-competition\04"


def spr(y_true, y_pred):
    try:
        c, _ = spearmanr(y_true, y_pred)
        return max(0.0, c) if not np.isnan(c) else 0.0
    except:
        return 0.0


def ev(y_true, y_pred, label=""):
    r = np.sqrt(mean_squared_error(y_true, y_pred))
    s = spr(y_true, y_pred)
    sc = 500 * max(0, 1 - r / 0.80) + 250 * s + 50
    print(f"  {label}: RMSE={r:.5f}, Spearman={s:.5f}, score={sc:.2f}")
    return r, s, sc


def main():
    print("=" * 60)
    print("全模型训练 + Stacking 集成")
    print("=" * 60)

    # Load
    feats = np.load(f"{OUTPUT}/X_train_features.npz")
    X = feats["X"].astype(np.float64)
    feats_test = np.load(f"{OUTPUT}/X_test_features.npz")
    X_test = feats_test["X"].astype(np.float64)
    df = pd.read_csv(f"{DATA}/train.csv")
    y = df["target"].values.astype(np.float64)
    test_df = pd.read_csv(f"{DATA}/test.csv")

    with open(f"{DATA}/scaffold_split.json") as f:
        split = json.load(f)

    # Clean
    X = np.nan_to_num(np.clip(X, -1e6, 1e6), nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(np.clip(X_test, -1e6, 1e6), nan=0.0, posinf=0.0, neginf=0.0)

    sel = VarianceThreshold(threshold=0.0)
    X = sel.fit_transform(X)
    X_test = sel.transform(X_test)
    print(f"Features: {X.shape[1]}")

    # Scaffold folds
    valid_ids = set(split["valid_ids"])
    df_ids = df["id"].tolist()
    valid_mask = np.array([(iid in valid_ids) for iid in df_ids])
    all_idx = np.arange(len(df))
    non_valid_idx = all_idx[~valid_mask]
    valid_idx = all_idx[valid_mask]

    folds = [(non_valid_idx, valid_idx)]
    kf = KFold(n_splits=5, shuffle=True, random_state=202604)
    for tr, va in kf.split(non_valid_idx):
        folds.append((non_valid_idx[tr], non_valid_idx[va]))
    print(f"CV folds: {len(folds)}")

    oof_dict, test_dict = {}, {}
    model_names = []

    # ---- CatBoost GPU ----
    print("\n[CatBoost GPU]")
    name = "cat"
    cb_params = dict(iterations=5000, depth=7, learning_rate=0.015, l2_leaf_reg=3.0,
                     random_strength=0.5, bagging_temperature=0.3, od_type="Iter",
                     od_wait=200, random_seed=42, verbose=False, allow_writing_files=False,
                     task_type="GPU", devices="0")
    oof = np.zeros(len(y))
    tp = np.zeros(len(X_test))
    for i, (tr, va) in enumerate(folds):
        m = cb.CatBoostRegressor(**cb_params)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]), verbose=False)
        oof[va] = m.predict(X[va])
        tp += m.predict(X_test) / len(folds)
        if i < 3: ev(y[va], oof[va], f"Cat fold{i+1}")
    ev(y, oof, "Cat OOF")
    oof_dict[name] = oof; test_dict[name] = tp; model_names.append(name)

    # ---- XGBoost CPU ----
    print("\n[XGBoost CPU]")
    name = "xgb"
    xgb_params = dict(n_estimators=5000, max_depth=6, learning_rate=0.015,
                      subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.7,
                      reg_alpha=0.1, reg_lambda=1.0, gamma=0.01, min_child_weight=5,
                      objective="reg:squarederror", tree_method="hist", device="cpu",
                      random_state=42, verbosity=0, early_stopping_rounds=200)
    oof = np.zeros(len(y))
    tp = np.zeros(len(X_test))
    for i, (tr, va) in enumerate(folds):
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict(X[va])
        tp += m.predict(X_test) / len(folds)
        if i < 3: ev(y[va], oof[va], f"XGB fold{i+1}")
    ev(y, oof, "XGB OOF")
    oof_dict[name] = oof; test_dict[name] = tp; model_names.append(name)

    # ---- LightGBM CPU ----
    print("\n[LightGBM CPU]")
    name = "lgb"
    lgb_params = dict(n_estimators=5000, max_depth=7, learning_rate=0.015,
                      subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                      reg_lambda=1.0, min_child_samples=20, num_leaves=63,
                      objective="regression", metric="rmse", random_state=42,
                      verbosity=-1)
    oof = np.zeros(len(y))
    tp = np.zeros(len(X_test))
    for i, (tr, va) in enumerate(folds):
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va])
        tp += m.predict(X_test) / len(folds)
        if i < 3: ev(y[va], oof[va], f"LGB fold{i+1}")
    ev(y, oof, "LGB OOF")
    oof_dict[name] = oof; test_dict[name] = tp; model_names.append(name)

    # ---- Ridge ----
    print("\n[Ridge]")
    name = "ridge"
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xts = sc.transform(X_test)
    oof = np.zeros(len(y))
    tp = np.zeros(len(X_test))
    for i, (tr, va) in enumerate(folds):
        m = Ridge(alpha=100.0)
        m.fit(Xs[tr], y[tr])
        oof[va] = m.predict(Xs[va])
        tp += m.predict(Xts) / len(folds)
    ev(y, oof, "Ridge OOF")
    oof_dict[name] = oof; test_dict[name] = tp; model_names.append(name)

    # ---- MLP sklearn ----
    print("\n[MLP sklearn]")
    name = "mlp"
    oof = np.zeros(len(y))
    tp = np.zeros(len(X_test))
    for i, (tr, va) in enumerate(folds):
        m = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu",
                         solver="adam", alpha=0.001, batch_size=64,
                         learning_rate="adaptive", learning_rate_init=0.001,
                         max_iter=500, early_stopping=True, validation_fraction=0.1,
                         random_state=42)
        m.fit(Xs[tr], y[tr])
        oof[va] = m.predict(Xs[va])
        tp += m.predict(Xts) / len(folds)
        if i < 3: ev(y[va], oof[va], f"MLP fold{i+1}")
    ev(y, oof, "MLP OOF")
    oof_dict[name] = oof; test_dict[name] = tp; model_names.append(name)

    # ---- Stacking Ensemble ----
    print("\n" + "=" * 50)
    print("Stacking Ensemble")
    print("=" * 50)

    X_meta = np.column_stack([oof_dict[n] for n in model_names])
    X_meta_test = np.column_stack([test_dict[n] for n in model_names])

    # Cross-validated meta-model
    stack_oof = np.zeros(len(y))
    for tr, va in folds:
        sm = Ridge(alpha=1.0)
        sm.fit(X_meta[tr], y[tr])
        stack_oof[va] = sm.predict(X_meta[va])
    ev(y, stack_oof, "Stack OOF")

    # Final meta-model
    final_meta = Ridge(alpha=1.0)
    final_meta.fit(X_meta, y)
    stack_test = final_meta.predict(X_meta_test)

    print("Meta coefficients:")
    for n, c in zip(model_names, final_meta.coef_):
        print(f"  {n}: {c:.4f}")
    print(f"  intercept: {final_meta.intercept_:.4f}")

    # Weighted average
    weights = {}
    for n in model_names:
        r = np.sqrt(mean_squared_error(y, oof_dict[n]))
        weights[n] = 1.0 / r**2
    tw = sum(weights.values())
    wavg_test = sum(test_dict[n] * weights[n] / tw for n in model_names)
    wavg_oof = sum(oof_dict[n] * weights[n] / tw for n in model_names)
    ev(y, wavg_oof, "Weighted OOF")

    # ---- Save ----
    print("\n保存...")

    # Save all individual models
    for n in model_names + ["stack", "wavg"]:
        preds = locals().get(f"{n}_test")
        if preds is None:
            meta_preds = {"stack": stack_test, "wavg": wavg_test}
            preds = meta_preds.get(n)
        sub = pd.DataFrame({"id": test_df["id"], "prediction": preds})
        sub.to_csv(f"{OUTPUT}/pred_{n}.csv", index=False)

    # Best = stacking
    final = pd.DataFrame({"id": test_df["id"], "prediction": stack_test})
    final.to_csv(f"{OUTPUT}/results.csv", index=False)
    print("results.csv saved")

    # Format check
    import subprocess as sp
    cp = sp.run([r"C:\Users\32010\AppData\Local\Programs\Python\Python312\python.exe",
                  f"{DATA}/tools/check_format.py", f"{OUTPUT}/results.csv",
                  "--test-csv", f"{DATA}/test.csv"], capture_output=True, text=True)
    print(cp.stdout.strip())
    if cp.stderr: print(cp.stderr.strip())

    # ZIP
    zf = zipfile.ZipFile(f"{OUTPUT}/NS-2026-04-answer.zip", "w", zipfile.ZIP_DEFLATED)
    zf.write(f"{OUTPUT}/results.csv", "results.csv")
    zf.close()
    print(f"NS-2026-04-answer.zip: {os.path.getsize(f'{OUTPUT}/NS-2026-04-answer.zip')} bytes")

    print("\n全部完成!")


if __name__ == "__main__":
    main()
