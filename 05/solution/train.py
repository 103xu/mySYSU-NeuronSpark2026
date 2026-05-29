"""
NS-2026-05 校园声景事件检测 - 简化版 Mel-CNN 多标签分类
"""

import csv, json, os, random, warnings
from pathlib import Path

import numpy as np
import librosa
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ===== 配置 =====
DATA = Path("92dd0c9e-0f5f-43f5-b1e7-2321f2dc181a")
SR, DUR = 16000, 6.0
N_MELS, HOP, N_FFT = 128, 512, 2048
BATCH, EPOCHS, LR = 16, 40, 0.001
DEVICE = torch.device("cpu")

# 加载标签
with open(DATA / "label_map.json", "r", encoding="utf-8") as f:
    lbs = json.load(f)["labels"]
ALL_LABELS = [l["name"] for l in lbs]
NON_AMBIENT = [l for l in ALL_LABELS if l != "ambient"]
N_CLS = len(ALL_LABELS)
L2I = {n: i for i, n in enumerate(ALL_LABELS)}
RARE = {"alarm", "door", "glass_break", "vehicle"}

print(f"标签: {ALL_LABELS}")


# ===== 数据 =====
def load_csv(path, audio_dir):
    with open(path, "r", encoding="utf-8-sig") as f:
        return [{**r, "path": str(audio_dir / f"{r['id']}.wav")}
                for r in csv.DictReader(f)]

train_rows = load_csv(DATA / "train.csv", DATA / "audio" / "train")
test_rows = load_csv(DATA / "test.csv", DATA / "audio" / "test")
print(f"训练: {len(train_rows)}, 测试: {len(test_rows)}")


def to_melspec(path, augment=False):
    y, sr = librosa.load(path, sr=SR, mono=True)
    target = int(SR * DUR)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    y = y[:target]

    if augment:
        if random.random() < 0.5:
            y += np.random.normal(0, 0.005 * np.std(y), len(y))
        if random.random() < 0.5:
            y *= random.uniform(0.8, 1.2)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80)
    log_mel = np.clip((log_mel + 80) / 80, 0, 1).astype(np.float32)
    return torch.from_numpy(log_mel).unsqueeze(0)


class AudioDS(Dataset):
    def __init__(self, rows, labels_dict, augment=False):
        self.rows = rows
        self.labels = labels_dict
        self.aug = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rid = self.rows[i]["id"]
        spec = to_melspec(self.rows[i]["path"], augment=self.aug)
        y = torch.tensor(self.labels.get(rid, np.zeros(N_CLS)), dtype=torch.float32)
        return spec, y


# 构建标签
labels_dict = {}
for r in train_rows:
    y = np.zeros(N_CLS, dtype=np.float32)
    for l in r["labels"].split("|"):
        if l.strip() in L2I:
            y[L2I[l.strip()]] = 1.0
    labels_dict[r["id"]] = y

# 划分
tr_idx, vl_idx = train_test_split(range(len(train_rows)), test_size=0.2,
                                   random_state=42, stratify=None)
train_ds = AudioDS([train_rows[i] for i in tr_idx], labels_dict, augment=True)
val_ds = AudioDS([train_rows[i] for i in vl_idx], labels_dict, augment=False)
test_ds = AudioDS(test_rows, {}, augment=False)

train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False)


# ===== 模型 =====
class MelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, N_CLS),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ===== 训练 =====
model = MelCNN().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.BCEWithLogitsLoss()

best_score, best_state, patience = 0, None, 10
no_imp = 0

for epoch in range(EPOCHS):
    model.train()
    for specs, labels in train_ld:
        specs, labels = specs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(specs), labels)
        loss.backward()
        opt.step()
    sch.step()

    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for specs, labels in val_ld:
            out = torch.sigmoid(model(specs.to(DEVICE))).cpu().numpy()
            all_p.append(out)
            all_y.append(labels.cpu().numpy())
    p = np.concatenate(all_p)
    y = np.concatenate(all_y)

    # 简单阈值 0.5 + ambient=0.3
    yp = np.zeros_like(y)
    for i, l in enumerate(ALL_LABELS):
        thr = 0.3 if l == "ambient" else 0.5
        yp[:, i] = (p[:, i] >= thr).astype(int)

    sf1 = f1_score(y, yp, average="samples", zero_division=0)
    mf1 = f1_score(y, yp, average="macro", zero_division=0)
    score = 0.5 * sf1 + 0.5 * mf1

    if score > best_score:
        best_score = score
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}  sf1={sf1:.4f}  mf1={mf1:.4f}  best={best_score:.4f}")

    if no_imp >= patience:
        print(f"早停 @ epoch {epoch+1}")
        break

model.load_state_dict(best_state)

# ===== 阈值优化 =====
model.eval()
val_p, val_y = [], []
with torch.no_grad():
    for specs, labels in val_ld:
        out = torch.sigmoid(model(specs.to(DEVICE))).cpu().numpy()
        val_p.append(out)
        val_y.append(labels.cpu().numpy())
val_p = np.concatenate(val_p)
val_y = np.concatenate(val_y)

thresholds = {}
for i, l in enumerate(ALL_LABELS):
    if l == "ambient":
        thresholds[l] = 0.3
        continue
    bt, bf = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 60):
        pred = (val_p[:, i] >= t).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(val_y[:, i], pred, zero_division=0)
        if f > bf:
            bf, bt = f, t
    thresholds[l] = bt

# 验证集评估
yp = np.zeros_like(val_y)
for i, l in enumerate(ALL_LABELS):
    yp[:, i] = (val_p[:, i] >= thresholds[l]).astype(int)

sf1 = f1_score(val_y, yp, average="samples", zero_division=0)
mf1 = f1_score(val_y, yp, average="macro", zero_division=0)
rare_idx = [L2I[l] for l in RARE]
rr = np.mean([val_y[:, j].dot(yp[:, j]) / max(val_y[:, j].sum(), 1) for j in rare_idx])

print(f"\n===== 结果 =====")
print(f"sample_f1:   {sf1:.4f}  (target: 0.74) {'OK' if sf1 >= 0.74 else '--'}")
print(f"macro_f1:    {mf1:.4f}  (target: 0.72) {'OK' if mf1 >= 0.72 else '--'}")
print(f"rare_recall: {rr:.4f}  (target: 0.84) {'OK' if rr >= 0.84 else '--'}")
print("每类F1/阈值:")
for i, l in enumerate(ALL_LABELS):
    f = f1_score(val_y[:, i], yp[:, i], zero_division=0)
    print(f"  {l:>12}: f1={f:.4f}  thr={thresholds[l]:.3f}")

# ===== 测试预测 =====
model.eval()
test_p = []
with torch.no_grad():
    for specs, _ in test_ld:
        out = torch.sigmoid(model(specs.to(DEVICE))).cpu().numpy()
        test_p.append(out)
test_p = np.concatenate(test_p)

results = []
for i, row in enumerate(test_rows):
    pred = [l for l in NON_AMBIENT if test_p[i][L2I[l]] >= thresholds[l]]
    if not pred:
        pred = ["ambient"]
    results.append({"id": row["id"], "labels": "|".join(pred)})

out = Path("solution/results.csv")
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "labels"])
    w.writeheader()
    w.writerows(results)
print(f"\n保存: {out}")

dist = {l: 0 for l in ALL_LABELS}
for r in results:
    for l in r["labels"].split("|"):
        dist[l] += 1
for l, c in sorted(dist.items()):
    print(f"  {l}: {c}")

# 保存模型
torch.save(best_state, Path("solution/models/model.pt"))
with open(Path("solution/models/thresholds.json"), "w") as f:
    json.dump(thresholds, f, indent=2)
print("完成!")
