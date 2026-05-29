"""深度学习方案：PyTorch MLP + GPU"""
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

DATA = r"C:\Users\32010\ai-competition\04\2a9fa6fc-563a-43b9-b8bb-1301289bb22d"
OUTPUT = r"C:\Users\32010\ai-competition\04"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class MolMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.append(model(X_batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        corr, _ = spearmanr(y_true, y_pred)
        spr = max(0.0, corr) if not np.isnan(corr) else 0.0
    except:
        spr = 0.0
    score = 500 * max(0, 1 - rmse / 0.80) + 250 * spr + 50
    print(f"  {label}: RMSE={rmse:.5f}, Spearman={spr:.5f}, score={score:.2f}")
    return rmse, spr, score


def train_single_model(X_train, y_train, X_val, y_val, input_dim, seed):
    """Train one MLP model with a specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = MolMLP(input_dim, hidden_dims=(512, 256, 128, 64), dropout=0.3).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    max_patience = 50

    for epoch in range(500):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        model.eval()
        val_preds = predict(model, val_loader)
        val_loss = mean_squared_error(y_val, val_preds)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            break

    model.load_state_dict(best_weights)
    return model


def main():
    print("=" * 60)
    print("深度学习方案：PyTorch MLP + GPU")
    print("=" * 60)

    # Load data
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

    # Clean
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    X_test = np.clip(X_test, -1e6, 1e6)

    # Remove low variance
    sel = VarianceThreshold(threshold=0.0)
    X = sel.fit_transform(X)
    X_test = sel.transform(X_test)
    print(f"After variance filter: {X.shape[1]} features")

    # Feature selection: mutual information top 2048
    if X.shape[1] > 2048:
        print("Running mutual information feature selection...")
        mi = mutual_info_regression(X, y, random_state=42)
        keep_idx = np.argsort(mi)[::-1][:2048]
        X = X[:, keep_idx]
        X_test = X_test[:, keep_idx]
        print(f"After MI selection: {X.shape[1]} features")

    # Scale features
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    # Scaffold split
    valid_ids = set(split["valid_ids"])
    df_ids = df["id"].tolist()
    valid_mask = np.array([(iid in valid_ids) for iid in df_ids])

    train_idx = np.where(~valid_mask)[0]
    val_idx = np.where(valid_mask)[0]
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    print(f"Scaffold split: train={len(X_tr)}, val={len(X_val)}")

    # Train ensemble of 5 models with different seeds
    print("\n" + "=" * 50)
    print("训练 5 个 MLP 集成")
    print("=" * 50)

    models = []
    test_preds_list = []

    for i, seed in enumerate([42, 123, 456, 789, 2026]):
        print(f"\n--- Model {i+1} (seed={seed}) ---")
        model = train_single_model(X_tr, y_tr, X_val, y_val, X.shape[1], seed)

        val_pred = predict(model, DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=512
        ))
        evaluate(y_val, val_pred, f"Val seed={seed}")

        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.zeros(len(X_test))),
            batch_size=512
        )
        test_pred = predict(model, test_loader)
        test_preds_list.append(test_pred)
        models.append(model)

    # Ensemble prediction
    print("\n" + "=" * 50)
    print("集成预测")
    print("=" * 50)

    test_preds = np.stack(test_preds_list, axis=0)

    # Also do OOF predictions
    oof_preds = np.zeros((len(models), len(y)))
    for i, model in enumerate(models):
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X), torch.zeros(len(y))),
            batch_size=512
        )
        oof_preds[i] = predict(model, loader)

    # Mean ensemble
    ensemble_test = test_preds.mean(axis=0)
    ensemble_oof = oof_preds.mean(axis=0)

    evaluate(y_val, ensemble_oof[val_idx], "Ensemble Val")
    evaluate(y, ensemble_oof, "Ensemble Full OOF")

    # Save
    print("\n" + "=" * 50)
    print("保存结果")
    print("=" * 50)

    sub = pd.DataFrame({"id": test_df["id"], "prediction": ensemble_test})
    sub.to_csv(f"{OUTPUT}/results.csv", index=False)
    print("  results.csv saved")

    # Format check
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

    # Zip
    import zipfile, os
    zip_path = f"{OUTPUT}/NS-2026-04-answer.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(f"{OUTPUT}/results.csv", "results.csv")
    print(f"\n  {zip_path} ({os.path.getsize(zip_path)} bytes)")
    print("\n完成!")


if __name__ == "__main__":
    main()
