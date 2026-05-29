"""
NS-2026-05 校园声景事件检测 - 使用 Wav2Vec2 深度特征 + 多标签分类器
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

warnings.filterwarnings("ignore")

# 路径配置
DATA_ROOT = Path("92dd0c9e-0f5f-43f5-b1e7-2321f2dc181a")
TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
LABEL_MAP = DATA_ROOT / "label_map.json"
AUDIO_TRAIN = DATA_ROOT / "audio" / "train"
AUDIO_TEST = DATA_ROOT / "audio" / "test"
OUTPUT_DIR = Path("solution")
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 6.0
RARE_LABELS = {"alarm", "door", "glass_break", "vehicle"}
DEVICE = torch.device("cpu")

# 加载标签
with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_info = json.load(f)
ALL_LABELS = [item["name"] for item in label_info["labels"]]
NON_AMBIENT_LABELS = [l for l in ALL_LABELS if l != "ambient"]
LABEL_TO_IDX = {name: i for i, name in enumerate(ALL_LABELS)}
print(f"标签: {ALL_LABELS}")

# 加载 Wav2Vec2
print("加载 Wav2Vec2 模型...")
model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
wav2vec2 = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
wav2vec2.eval()
HIDDEN_SIZE = wav2vec2.config.hidden_size  # 768


def load_data():
    """加载所有数据"""
    for csv_path, audio_dir in [(TRAIN_CSV, AUDIO_TRAIN), (TEST_CSV, AUDIO_TEST)]:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                audio_path = str(audio_dir / f"{row['id']}.wav")
                rows.append({
                    "id": row["id"],
                    "audio_path": audio_path,
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


@torch.no_grad()
def extract_wav2vec2_embedding(audio_path):
    """提取 Wav2Vec2 嵌入（聚合统计量）"""
    y, sr = sf.read(audio_path)
    if len(y.shape) > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    y = y[:target_len].astype(np.float32)

    # 分段处理以减少内存 (3段 x 2秒)
    segment_duration = 2.0
    segment_samples = int(SAMPLE_RATE * segment_duration)
    n_segments = 3
    all_embeddings = []

    for seg_idx in range(n_segments):
        start = seg_idx * segment_samples
        end = start + segment_samples
        seg = y[start:end]

        inputs = processor(seg, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                          padding=True, truncation=True, max_length=int(SAMPLE_RATE * segment_duration))
        input_values = inputs.input_values.to(DEVICE)
        # attention_mask = inputs.attention_mask.to(DEVICE)

        outputs = wav2vec2(input_values, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # (1, T, 768)

        # 池化
        mean_pool = last_hidden.mean(dim=1).cpu().numpy().flatten()  # (768,)
        max_pool = last_hidden.max(dim=1).values.cpu().numpy().flatten()  # (768,)
        std_pool = last_hidden.std(dim=1).cpu().numpy().flatten()  # (768,)
        all_embeddings.append(np.concatenate([mean_pool, max_pool, std_pool]))

    # 聚合多段
    stacked = np.stack(all_embeddings, axis=0)  # (3, 2304)
    final_embedding = np.concatenate([
        stacked.mean(axis=0),
        stacked.max(axis=0),
        stacked.std(axis=0),
    ])
    return final_embedding


def extract_handcrafted_features(audio_path):
    """提取手工音频特征"""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        if len(y) < int(SAMPLE_RATE * DURATION):
            y = np.pad(y, (0, int(SAMPLE_RATE * DURATION) - len(y)))

        features = []
        # 时域统计
        features.append(np.sqrt(np.mean(y ** 2)))  # rms
        features.append(np.mean(np.abs(y)))  # mean_abs
        features.append(np.std(y))
        features.append(float(stats.skew(y)))
        features.append(float(stats.kurtosis(y)))
        features.append(np.max(np.abs(y)))

        # MFCC 统计
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfcc, axis=1).tolist())
        features.extend(np.std(mfcc, axis=1).tolist())

        # 频谱特征
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        features.append(np.mean(log_mel))
        features.append(np.std(log_mel))

        # 频谱质心
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(centroid))
        features.append(np.std(centroid))

        # 过零率
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"  手工特征提取错误 {audio_path}: {e}")
        return np.zeros(86, dtype=np.float32)


def build_features(rows, desc="", cache_path=None):
    """构建特征矩阵"""
    if cache_path and os.path.exists(cache_path):
        print(f"  从缓存加载 {desc} 特征...")
        data = np.load(cache_path)
        return data["X_w2v"], data["X_hc"], data["ids"]

    X_w2v_list, X_hc_list, ids = [], [], []
    for i, row in enumerate(rows):
        if (i + 1) % 20 == 0:
            print(f"  {desc} 特征提取: {i + 1}/{len(rows)}")

        w2v_emb = extract_wav2vec2_embedding(row["audio_path"])
        hc_feat = extract_handcrafted_features(row["audio_path"])
        X_w2v_list.append(w2v_emb)
        X_hc_list.append(hc_feat)
        ids.append(row["id"])

    X_w2v = np.stack(X_w2v_list, axis=0)
    X_hc = np.stack(X_hc_list, axis=0)

    if cache_path:
        np.savez_compressed(cache_path, X_w2v=X_w2v, X_hc=X_hc, ids=np.array(ids))
    return X_w2v, X_hc, ids


def build_labels(train_rows):
    """构建标签矩阵"""
    y = np.zeros((len(train_rows), len(ALL_LABELS)), dtype=np.float32)
    for i, row in enumerate(train_rows):
        labels = [l.strip() for l in row["labels"].split("|") if l.strip()]
        for label in labels:
            y[i, LABEL_TO_IDX[label]] = 1.0
    return y


def optimize_thresholds(y_true, y_prob, labels):
    """每类最优阈值搜索"""
    thresholds = {}
    for i, label in enumerate(labels):
        if label == "ambient":
            thresholds[label] = 0.35
            continue
        best_t, best_f = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, 90):
            pred = (y_prob[:, i] >= t).astype(int)
            if pred.sum() == 0:
                continue
            f = f1_score(y_true[:, i], pred, zero_division=0)
            if f > best_f:
                best_f = f
                best_t = t
        thresholds[label] = best_t
        print(f"  {label}: thr={best_t:.3f}, f1={best_f:.4f}")
    return thresholds


def evaluate(y_true, y_prob, thresholds):
    """全面评估"""
    y_pred = np.zeros_like(y_true)
    for i in range(len(ALL_LABELS)):
        y_pred[:, i] = (y_prob[:, i] >= thresholds[ALL_LABELS[i]]).astype(int)

    sample_f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = {}
    for i, l in enumerate(ALL_LABELS):
        per_class[l] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)

    rare_idx = [LABEL_TO_IDX[l] for l in RARE_LABELS]
    rare_recall = np.mean([
        np.sum(y_true[:, j] * y_pred[:, j]) / max(np.sum(y_true[:, j]), 1)
        for j in rare_idx
    ])

    return {"sample_f1": sample_f1, "macro_f1": macro_f1,
            "rare_recall": rare_recall, "per_class_f1": per_class}


def main():
    train_rows, test_rows = load_data()

    # 提取特征（Wav2Vec2 深度特征 + 手工特征）
    FEAT_CACHE_DIR = MODEL_DIR / "features"
    FEAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n===== 提取训练特征 =====")
    X_train_w2v, X_train_hc, train_ids = build_features(
        train_rows, "训练", str(FEAT_CACHE_DIR / "train_feat.npz"))
    print(f"Wav2Vec2 特征: {X_train_w2v.shape}")
    print(f"手工特征: {X_train_hc.shape}")

    print("\n===== 提取测试特征 =====")
    X_test_w2v, X_test_hc, test_ids = build_features(
        test_rows, "测试", str(FEAT_CACHE_DIR / "test_feat.npz"))
    print(f"Wav2Vec2 特征: {X_test_w2v.shape}")
    print(f"手工特征: {X_test_hc.shape}")

    # 合并特征
    X_train = np.concatenate([X_train_w2v, X_train_hc], axis=1)
    X_test = np.concatenate([X_test_w2v, X_test_hc], axis=1)
    print(f"\n合并特征维度: {X_train.shape[1]}")

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = build_labels(train_rows)
    print(f"标签分布: {dict(zip(ALL_LABELS, y_train.sum(axis=0)))}")

    # ===== 训练多模型集成 =====
    print("\n===== 训练分类器 =====")

    # 模型1: RandomForest
    print("  训练 RandomForest...")
    rf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=3,
                               class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    rf.fit(X_train, y_train)

    # 模型2: GradientBoosting
    print("  训练 GradientBoosting...")
    gb = MultiOutputClassifier(
        GradientBoostingClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                   subsample=0.8, random_state=42))
    gb.fit(X_train, y_train)

    # 模型3: ExtraTrees
    print("  训练 ExtraTrees...")
    et = MultiOutputClassifier(
        ExtraTreesClassifier(n_estimators=300, max_depth=20, min_samples_leaf=3,
                            class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    et.fit(X_train, y_train)

    # 模型4: MLP (神经网络分类器)
    print("  训练 MLP...")
    mlp = MultiOutputClassifier(
        MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation="relu",
                      alpha=0.001, batch_size=32, max_iter=500, early_stopping=True,
                      random_state=42, verbose=False))
    mlp.fit(X_train, y_train)

    # ===== 交叉验证 + 阈值优化 =====
    print("\n===== CV 预测 (3-fold) =====")
    n_splits = min(3, min(y_train.sum(axis=0).astype(int)))

    def cv_proba(model, X, y):
        probas = cross_val_predict(model, X, y, cv=5, method="predict_proba", n_jobs=-1)
        if isinstance(probas, list):
            return np.column_stack(probas)
        return probas

    y_prob_rf = cv_proba(rf, X_train, y_train)
    y_prob_gb = cv_proba(gb, X_train, y_train)
    y_prob_et = cv_proba(et, X_train, y_train)
    y_prob_mlp = cv_proba(mlp, X_train, y_train)

    # 加权集成
    y_prob_ensemble = (y_prob_rf * 0.3 + y_prob_gb * 0.2 +
                       y_prob_et * 0.3 + y_prob_mlp * 0.2)

    print("\n===== 阈值优化 =====")
    thresholds = optimize_thresholds(y_train, y_prob_ensemble, ALL_LABELS)

    # 评估
    print("\n===== CV 评估 =====")
    metrics = evaluate(y_train, y_prob_ensemble, thresholds)
    print(f"  sample_f1: {metrics['sample_f1']:.4f}")
    print(f"  macro_f1: {metrics['macro_f1']:.4f}")
    print(f"  rare_recall: {metrics['rare_recall']:.4f}")
    print(f"  Gold-band:")
    for name, target in [("macro_f1", 0.72), ("sample_f1", 0.74), ("rare_recall", 0.84)]:
        v = metrics[name]
        print(f"    {name} >= {target}: {v:.4f} [{'PASS' if v >= target else 'FAIL'}]")
    print("  每类F1:")
    for l, f1 in sorted(metrics["per_class_f1"].items()):
        print(f"    {l}: {f1:.4f}")

    # ===== 测试集预测 =====
    print("\n===== 测试集预测 =====")
    test_p_rf = np.column_stack(rf.predict_proba(X_test))
    test_p_gb = np.column_stack(gb.predict_proba(X_test))
    test_p_et = np.column_stack(et.predict_proba(X_test))
    test_p_mlp = np.column_stack(mlp.predict_proba(X_test))
    test_prob = (test_p_rf * 0.3 + test_p_gb * 0.2 +
                 test_p_et * 0.3 + test_p_mlp * 0.2)

    results = []
    for i, row in enumerate(test_rows):
        probs = test_prob[i]
        predicted = []
        for label in NON_AMBIENT_LABELS:
            idx = LABEL_TO_IDX[label]
            if probs[idx] >= thresholds[label]:
                predicted.append(label)
        if not predicted:
            predicted = ["ambient"]
        results.append({"id": row["id"], "labels": "|".join(predicted)})

    output_path = OUTPUT_DIR / "results.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "labels"])
        writer.writeheader()
        writer.writerows(results)
    print(f"结果已保存: {output_path}")

    # 测试集预测分布
    dist = {l: 0 for l in ALL_LABELS}
    for r in results:
        for l in r["labels"].split("|"):
            dist[l] += 1
    print("测试集预测分布:")
    for l, c in sorted(dist.items()):
        print(f"  {l}: {c}")

    # 保存所有模型
    print("\n===== 保存模型 =====")
    for name, obj in [("scaler", scaler), ("rf", rf), ("gb", gb), ("et", et), ("mlp", mlp)]:
        with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    with open(MODEL_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)


if __name__ == "__main__":
    main()
