"""
NS-2026-05 - Wav2Vec2 特征 + 集成分类器 (单次前向传播版本)
"""

import csv
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               ExtraTreesClassifier)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

warnings.filterwarnings("ignore")

DATA_ROOT = Path("92dd0c9e-0f5f-43f5-b1e7-2321f2dc181a")
TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
LABEL_MAP = DATA_ROOT / "label_map.json"
AUDIO_TRAIN = DATA_ROOT / "audio" / "train"
AUDIO_TEST = DATA_ROOT / "audio" / "test"
OUTPUT_DIR = Path("solution")
MODEL_DIR = OUTPUT_DIR / "models"
FEAT_DIR = MODEL_DIR / "features"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
DUR = 6.0
TARGET_LEN = int(SR * DUR)
RARE_LABELS = {"alarm", "door", "glass_break", "vehicle"}

with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_info = json.load(f)
ALL_LABELS = [item["name"] for item in label_info["labels"]]
NON_AMBIENT = [l for l in ALL_LABELS if l != "ambient"]
L2I = {n: i for i, n in enumerate(ALL_LABELS)}

print(f"标签 ({len(ALL_LABELS)}): {ALL_LABELS}")
print(f"稀有标签: {RARE_LABELS}")

# 加载模型
print("加载 Wav2Vec2...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()
H = model.config.hidden_size


def load_csv(path, audio_dir):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({
                "id": row["id"],
                "path": str(audio_dir / f"{row['id']}.wav"),
                "site": row.get("site", ""),
                "device": row.get("device", ""),
                "labels": row.get("labels", ""),
            })
    return rows


@torch.no_grad()
def extract_embedding(audio_path):
    """提取 Wav2Vec2 嵌入 - 完整6秒一次前向传播"""
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    y = y[:TARGET_LEN].astype(np.float32)

    # 使用更大的 stride 减少帧数, 加速且减少内存
    # 每 40ms 取一帧 (原始是20ms)
    inputs = processor(y, sampling_rate=SR, return_tensors="pt")
    input_values = inputs.input_values

    # 如果太长, 截断
    max_len = 100000
    if input_values.shape[1] > max_len:
        input_values = input_values[:, :max_len]

    outputs = model(input_values, output_hidden_states=True)
    hidden = outputs.last_hidden_state  # (1, T, 768)

    # 多层池化
    mean_p = hidden.mean(dim=1).numpy().flatten()
    max_p = hidden.max(dim=1).values.numpy().flatten()
    std_p = hidden.std(dim=1).numpy().flatten()

    # 分段池化 (前/中/后)
    t = hidden.shape[1]
    segments = []
    for start, end in [(0, t//3), (t//3, 2*t//3), (2*t//3, t)]:
        seg = hidden[:, start:end, :]
        segments.append(seg.mean(dim=1).numpy().flatten())

    feat = np.concatenate([mean_p, max_p, std_p] + segments)
    return feat.astype(np.float32)


def extract_all(rows, name, cache_file):
    """提取所有样本的特征（支持缓存）"""
    if os.path.exists(cache_file):
        print(f"  从缓存加载 {name} 特征...")
        data = np.load(cache_file)
        return data["X"], list(data["ids"])

    X_list, ids = [], []
    for i, row in enumerate(rows):
        if (i + 1) % 30 == 0:
            print(f"  {name}: {i+1}/{len(rows)}")
        feat = extract_embedding(row["path"])
        X_list.append(feat)
        ids.append(row["id"])

    X = np.stack(X_list)
    np.savez_compressed(cache_file, X=X, ids=np.array(ids))
    print(f"  {name} 完成: {X.shape}")
    return X, ids


def get_labels(rows):
    y = np.zeros((len(rows), len(ALL_LABELS)), dtype=np.float32)
    for i, r in enumerate(rows):
        for l in r["labels"].split("|"):
            l = l.strip()
            if l in L2I:
                y[i, L2I[l]] = 1.0
    return y


def find_thresholds(y_true, y_prob):
    """搜索每类最优阈值"""
    best = {}
    for i, label in enumerate(ALL_LABELS):
        if label == "ambient":
            best[label] = 0.3
            continue
        bt, bf = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, 90):
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
    for i in range(len(ALL_LABELS)):
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
    # 加载数据
    train_rows = load_csv(TRAIN_CSV, AUDIO_TRAIN)
    test_rows = load_csv(TEST_CSV, AUDIO_TEST)
    print(f"训练: {len(train_rows)}, 测试: {len(test_rows)}")

    # 特征提取
    print("\n===== 特征提取 =====")
    X_train, train_ids = extract_all(train_rows, "训练集", str(FEAT_DIR / "train_w2v.npz"))
    X_test, test_ids = extract_all(test_rows, "测试集", str(FEAT_DIR / "test_w2v.npz"))

    # 合并手工特征增强
    print("\n===== 添加手工特征 =====")
    import librosa
    from scipy import stats as sp_stats

    def handcrafted(path):
        y, sr = librosa.load(path, sr=SR, duration=DUR)
        if len(y) < TARGET_LEN:
            y = np.pad(y, (0, TARGET_LEN - len(y)))
        feats = []
        feats.append(np.sqrt(np.mean(y**2)))
        feats.append(np.std(y))
        feats.append(float(sp_stats.skew(y)))
        feats.append(float(sp_stats.kurtosis(y)))
        feats.append(np.max(np.abs(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        feats.extend(np.mean(mfcc, axis=1))
        feats.extend(np.std(mfcc, axis=1))
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        lm = librosa.power_to_db(mel, ref=np.max)
        feats.append(np.mean(lm))
        feats.append(np.std(lm))
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats.append(np.mean(cent))
        feats.append(np.std(cent))
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats.append(np.mean(zcr))
        return np.array(feats, dtype=np.float32)

    hc_cache = str(FEAT_DIR / "hc_feats.npz")
    if os.path.exists(hc_cache):
        d = np.load(hc_cache)
        HC_train, HC_test = d["train"], d["test"]
    else:
        HC_train = np.stack([handcrafted(r["path"]) for r in train_rows])
        HC_test = np.stack([handcrafted(r["path"]) for r in test_rows])
        np.savez_compressed(hc_cache, train=HC_train, test=HC_test)

    X_train = np.concatenate([X_train, HC_train], axis=1)
    X_test = np.concatenate([X_test, HC_test], axis=1)
    print(f"最终特征维度: {X_train.shape[1]}")

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.nan_to_num(X_train, 0, 0, 0)
    X_test = np.nan_to_num(X_test, 0, 0, 0)

    y_train = get_labels(train_rows)
    for l, c in zip(ALL_LABELS, y_train.sum(axis=0).astype(int)):
        print(f"  {l}: {c}")

    # 训练集成
    print("\n===== 训练模型 =====")

    rf = MultiOutputClassifier(RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    rf.fit(X_train, y_train)
    print("  RandomForest 完成")

    gb = MultiOutputClassifier(GradientBoostingClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        subsample=0.8, random_state=42))
    gb.fit(X_train, y_train)
    print("  GradientBoosting 完成")

    et = MultiOutputClassifier(ExtraTreesClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    et.fit(X_train, y_train)
    print("  ExtraTrees 完成")

    mlp = MultiOutputClassifier(MLPClassifier(
        hidden_layer_sizes=(512, 256, 128), activation="relu",
        alpha=0.001, max_iter=500, early_stopping=True, random_state=42))
    mlp.fit(X_train, y_train)
    print("  MLP 完成")

    # CV 预测用于阈值校准
    print("\n===== CV预测 =====")
    p_rf = cross_val_predict(rf, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1)
    if isinstance(p_rf, list):
        p_rf = np.column_stack(p_rf)
    p_gb = cross_val_predict(gb, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1)
    if isinstance(p_gb, list):
        p_gb = np.column_stack(p_gb)
    p_et = cross_val_predict(et, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1)
    if isinstance(p_et, list):
        p_et = np.column_stack(p_et)
    p_mlp = cross_val_predict(mlp, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1)
    if isinstance(p_mlp, list):
        p_mlp = np.column_stack(p_mlp)

    p_ens = (p_rf * 0.3 + p_gb * 0.2 + p_et * 0.3 + p_mlp * 0.2)

    # 阈值
    print("\n===== 阈值优化 =====")
    thr = find_thresholds(y_train, p_ens)

    # 评估
    print("\n===== CV 评估 =====")
    sf1, mf1, rr, pc = evaluate(y_train, p_ens, thr)
    print(f"  sample_f1:    {sf1:.4f}  (target: 0.74)")
    print(f"  macro_f1:     {mf1:.4f}  (target: 0.72)")
    print(f"  rare_recall:  {rr:.4f}  (target: 0.84)")
    for t, v in [("macro_f1", mf1), ("sample_f1", sf1), ("rare_recall", rr)]:
        target = {"macro_f1": 0.72, "sample_f1": 0.74, "rare_recall": 0.84}[t]
        print(f"    {t}: {'PASS' if v >= target else 'FAIL'}")
    print("  每类F1:")
    for l, f in sorted(pc.items()):
        print(f"    {l}: {f:.4f}")

    # 测试预测
    print("\n===== 测试预测 =====")
    tp_rf = np.column_stack(rf.predict_proba(X_test))
    tp_gb = np.column_stack(gb.predict_proba(X_test))
    tp_et = np.column_stack(et.predict_proba(X_test))
    tp_mlp = np.column_stack(mlp.predict_proba(X_test))
    tp = (tp_rf * 0.3 + tp_gb * 0.2 + tp_et * 0.3 + tp_mlp * 0.2)

    results = []
    for i, row in enumerate(test_rows):
        probs = tp[i]
        pred = [l for l in NON_AMBIENT if probs[L2I[l]] >= thr[l]]
        if not pred:
            pred = ["ambient"]
        results.append({"id": row["id"], "labels": "|".join(pred)})

    out = OUTPUT_DIR / "results.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "labels"])
        w.writeheader()
        w.writerows(results)
    print(f"保存: {out}")

    # 保存模型
    for name, obj in [("scaler", scaler), ("rf", rf), ("gb", gb), ("et", et), ("mlp", mlp)]:
        with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    with open(MODEL_DIR / "thresholds.json", "w") as fh:
        json.dump(thr, fh, indent=2)

    dist = {l: 0 for l in ALL_LABELS}
    for r in results:
        for l in r["labels"].split("|"):
            dist[l] += 1
    for l, c in sorted(dist.items()):
        print(f"  {l}: {c}")


if __name__ == "__main__":
    main()
