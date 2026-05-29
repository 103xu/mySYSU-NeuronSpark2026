import os
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import pickle
import jieba
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = "1fa3ed1b-c5a5-48e0-b0e5-09a27ac38e4d"
CORPUS_PATH = f"{DATA_DIR}/corpus.jsonl"

print("Loading corpus...")

docs = []
texts = []
tokenized = []

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        docs.append(item)
        text = f"{item['title']} {item['text']}"
        texts.append(text)
        tokenized.append(list(jieba.cut(text)))

print(f"Loaded {len(docs)} documents")

print("Building BM25...")
bm25 = BM25Okapi(tokenized)

with open("bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

print("Encoding corpus...")
embeddings = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

print(f"Saving embeddings ({embeddings.shape[0]} x {embeddings.shape[1]})...")
np.save("embeddings.npy", embeddings)

print("Done.")
