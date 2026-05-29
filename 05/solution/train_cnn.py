"""
NS-2026-05 - Mel-Spectrogram CNN 多标签分类
使用 log-mel spectrogram + 轻量 CNN + 数据增强
"""

import csv
import json
import os
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import librosa
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ===== 配置 =====
DATA_ROOT = Path("92dd0c9e-0f5f-43f5-b1e7-2321f2dc181a")
TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
LABEL_MAP = DATA_ROOT / "label_map.json"
AUDIO_TRAIN = DATA_ROOT / "audio" / "train"
AUDIO_TEST = DATA_ROOT / "audio" / "test"
OUTPUT_DIR = Path("solution")
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
DUR = 6.0
N_MELS = 128
HOP_LENGTH = 512  # 32ms
N_FFT = 2048
N_TIME = int(SR * DUR / HOP_LENGTH) + 1  # ~188 time frames
RARE_LABELS = {"alarm", "door", "glass_break", "vehicle"}
DEVICE = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS = 80
LR = 0.001

# 加载标签
with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_info = json.load(f)
ALL_LABELS = [item["name"] for item in label_info["labels"]]
NON_AMBIENT = [l for l in ALL_LABELS if l != "ambient"]
N_CLASSES = len(ALL_LABELS)
L2I = {n: i for i, n in enumerate(ALL_LABELS)}
I2L = {i: n for n, i in L2I.items()}
print(f"标签 ({N_CLASSES}): {ALL_LABELS}")


# ===== 数据加载 =====
def load_data():
    for csv_path, audio_dir in [(TRAIN_CSV, AUDIO_TRAIN), (TEST_CSV, AUDIO_TEST)]:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            rows = []
            for row in csv.DictReader(f):
                rows.append({
                    "id": row["id"],
                    "path": str(audio_dir / f"{row['id']}.wav"),
                    "site": row.get("site", ""),
                    "device": row.get("device", ""),
                    "labels": row.get("labels", ""),
                })
        if "train" in str(csv_path):
            train_rows = rows
        else:
            test_rows = rows
    print(f"训练: {len(train_rows)}, 测试: {len(test_rows)}")
    return train_rows, test_rows


def audio_to_melspec(y, sr=SR, augment=False):
    """转 log-mel spectrogram，支持数据增强"""
    if len(y) < int(SR * DUR):
        y = np.pad(y, (0, int(SR * DUR) - len(y)))
    y = y[:int(SR * DUR)]

    # 数据增强
    if augment:
        # 加噪声
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.005 * np.std(y), len(y))
            y = y + noise
        # 时间偏移
        if random.random() < 0.5:
            shift = random.randint(-SR//2, SR//2)
            if shift > 0:
                y = np.pad(y, (shift, 0))[:len(y)]
            else:
                y = np.pad(y, (0, -shift))[-shift:]
        # 音量变化
        if random.random() < 0.5:
            y = y * random.uniform(0.8, 1.2)
        # 时间拉伸
        if random.random() < 0.3:
            rate = random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y=y, rate=rate)
            if len(y) < int(SR * DUR):
                y = np.pad(y, (0, int(SR * DUR) - len(y)))
            y = y[:int(SR * DUR)]
        # 音高偏移
        if random.random() < 0.3:
            y = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=random.uniform(-2, 2))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80)

    # 归一化
    log_mel = (log_mel + 80) / 80  # → [0, 1] approximately
    log_mel = np.clip(log_mel, 0, 1)
    return log_mel.astype(np.float32)


# ===== Dataset =====
class AudioDataset(Dataset):
    def __init__(self, rows, labels_dict=None, augment=False):
        self.rows = rows
        self.labels_dict = labels_dict or {}
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        y, sr = librosa.load(row["path"], sr=SR, mono=True)
        spec = audio_to_melspec(y, sr, augment=self.augment)
        spec = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, n_time)

        if row["id"] in self.labels_dict:
            label = torch.tensor(self.labels_dict[row["id"]], dtype=torch.float32)
        else:
            label = torch.zeros(N_CLASSES, dtype=torch.float32)
        return spec, label, row["id"]


# ===== 模型 =====
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, pool=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, padding=kernel//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel, padding=kernel//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.conv(x)


class MelCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        # Input: (1, 128, 188)
        self.block1 = ConvBlock(1, 32, pool=(2, 2))   # (32, 64, 94)
        self.block2 = ConvBlock(32, 64, pool=(2, 2))   # (64, 32, 47)
        self.block3 = ConvBlock(64, 128, pool=(2, 2))  # (128, 16, 23)
        self.block4 = ConvBlock(128, 256, pool=(2, 2)) # (256, 8, 11)
        self.block5 = ConvBlock(256, 512, pool=(2, 2)) # (512, 4, 5)

        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.Flatten(),
            nn.Softmax(dim=1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)  # (B, 512, 4, 5)

        avg = self.global_pool(x).flatten(1)  # (B, 512)
        max_ = self.max_pool(x).flatten(1)    # (B, 512)
        x = torch.cat([avg, max_], dim=1)     # (B, 1024)

        x = self.classifier(x)
        return x


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for specs, labels, _ in loader:
        specs, labels = specs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_probs, all_ids = [], []
    for specs, _, ids in loader:
        specs = specs.to(DEVICE)
        outputs = model(specs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.append(probs)
        all_ids.extend(ids)
    return np.concatenate(all_probs), all_ids


def find_thresholds(y_true, y_prob):
    """搜索每类最优阈值"""
    best = {}
    for i, label in enumerate(ALL_LABELS):
        if label == "ambient":
            best[label] = 0.3
            continue
        bt, bf = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, 60):
            pred = (y_prob[:, i] >= t).astype(int)
            if pred.sum() == 0:
                continue
            f = f1_score(y_true[:, i], pred, zero_division=0)
            if f > bf:
                bf, bt = f, t
        best[label] = bt
        print(f"  {label}: thr={bt:.3f}, f1={bf:.4f}")
    return best


def evaluate(y_true, y_prob, thr):
    y_pred = np.zeros_like(y_true)
    for i in range(N_CLASSES):
        y_pred[:, i] = (y_prob[:, i] >= thr[ALL_LABELS[i]]).astype(int)

    sf1 = f1_score(y_true, y_pred, average="samples", zero_division=0)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rare_idx = [L2I[l] for l in RARE_LABELS]
    rr = np.mean([y_true[:, j].dot(y_pred[:, j]) / max(y_true[:, j].sum(), 1) for j in rare_idx])
    pc = {}
    for i, l in enumerate(ALL_LABELS):
        pc[l] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    return sf1, mf1, rr, pc


def main():
    train_rows, test_rows = load_data()

    # 构建标签字典
    labels_dict = {}
    for r in train_rows:
        y = np.zeros(N_CLASSES, dtype=np.float32)
        for l in r["labels"].split("|"):
            l = l.strip()
            if l in L2I:
                y[L2I[l]] = 1.0
        labels_dict[r["id"]] = y
    y_train = np.stack([labels_dict[r["id"]] for r in train_rows])

    # 划分 K 折
    label_combo = [tuple(int(v) for v in row) for row in y_train]
    combo_map = {}
    combo_ids = []
    for c in label_combo:
        if c not in combo_map:
            combo_map[c] = len(combo_map)
        combo_ids.append(combo_map[c])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 存储每折的模型和预测
    fold_models = []
    oof_probs = np.zeros((len(train_rows), N_CLASSES))
    test_probs_folds = np.zeros((len(test_rows), N_CLASSES))

    test_dataset = AudioDataset(test_rows, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(train_rows)), combo_ids)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/5")
        print(f"{'='*50}")

        train_subset = [train_rows[i] for i in train_idx]
        val_subset = [train_rows[i] for i in val_idx]

        train_ds = AudioDataset(train_subset, labels_dict, augment=True)
        val_ds = AudioDataset(val_subset, labels_dict, augment=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = MelCNN(N_CLASSES).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_f1 = 0
        best_state = None
        patience = 15
        no_improve = 0

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()

            val_prob, val_ids = predict(model, val_loader)
            y_val = np.stack([labels_dict[i] for i in val_ids])
            temp_thr = {l: (0.3 if l == "ambient" else 0.5) for l in ALL_LABELS}
            sf1, mf1, _, _ = evaluate(y_val, val_prob, temp_thr)
            combined = 0.5 * sf1 + 0.5 * mf1

            if combined > best_val_f1:
                best_val_f1 = combined
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{EPOCHS}  loss={train_loss:.4f}  "
                      f"val_sf1={sf1:.4f}  val_mf1={mf1:.4f}  best={best_val_f1:.4f}")

            if no_improve >= patience:
                print(f"  早停 @ epoch {epoch+1}")
                break

        # 恢复最佳状态
        model.load_state_dict(best_state)
        fold_models.append(best_state)

        # OOF 预测
        val_prob, val_ids = predict(model, val_loader)
        for j, vid in enumerate(val_ids):
            idx = train_rows.index(next(r for r in train_rows if r["id"] == vid))
            oof_probs[idx] = val_prob[j]

        # 测试集预测
        test_prob, _ = predict(model, test_loader)
        test_probs_folds += test_prob / 5.0

        # 验证集评估
        print(f"  Fold {fold+1} 最佳模型: val_combined={best_val_f1:.4f}")

    # ===== 阈值优化 =====
    print(f"\n===== OOF 阈值优化 =====")
    thresholds = find_thresholds(y_train, oof_probs)

    # ===== OOF 评估 =====
    print(f"\n===== OOF 评估 (5-fold CV) =====")
    sf1, mf1, rr, pc = evaluate(y_train, oof_probs, thresholds)
    print(f"  sample_f1:    {sf1:.4f}  (target: 0.74) {'✓' if sf1 >= 0.74 else '✗'}")
    print(f"  macro_f1:     {mf1:.4f}  (target: 0.72) {'✓' if mf1 >= 0.72 else '✗'}")
    print(f"  rare_recall:  {rr:.4f}  (target: 0.84) {'✓' if rr >= 0.84 else '✗'}")
    print("  每类F1:")
    for l, f in sorted(pc.items()):
        print(f"    {l:>12}: {f:.4f}")

    # ===== 测试集预测 =====
    print(f"\n===== 生成测试集预测 =====")
    results = []
    for i, row in enumerate(test_rows):
        probs = test_probs_folds[i]
        pred = [l for l in NON_AMBIENT if probs[L2I[l]] >= thresholds[l]]
        if not pred:
            pred = ["ambient"]
        results.append({"id": row["id"], "labels": "|".join(pred)})

    out_path = OUTPUT_DIR / "results.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "labels"])
        w.writeheader()
        w.writerows(results)
    print(f"保存: {out_path}")

    # 预测分布
    dist = {l: 0 for l in ALL_LABELS}
    for r in results:
        for l in r["labels"].split("|"):
            dist[l] += 1
    for l, c in sorted(dist.items()):
        print(f"  {l}: {c}")

    # 保存模型
    print(f"\n===== 保存模型 =====")
    for i, state in enumerate(fold_models):
        torch.save(state, MODEL_DIR / f"fold_{i}.pt")
    with open(MODEL_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("完成!")


if __name__ == "__main__":
    main()
