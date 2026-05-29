import json
import numpy as np
import pandas as pd
from scipy import sparse

DATA = r"C:\Users\32010\ai-competition\04\2a9fa6fc-563a-43b9-b8bb-1301289bb22d"

# scaffold split
with open(f"{DATA}/scaffold_split.json") as f:
    split = json.load(f)
print("scaffold_split type:", type(split))
print("keys:", list(split.keys()))
train_ids = split["train_ids"]
val_ids = split["valid_ids"]
print(f"train: {len(train_ids)}, val: {len(val_ids)}")
print("train samples:", train_ids[:3])
print("val samples:", val_ids[:3])

# train.csv
df = pd.read_csv(f"{DATA}/train.csv")
print(f"\ntrain shape: {df.shape}")
print(df.head())
print(f"\ntarget stats: mean={df['target'].mean():.4f}, std={df['target'].std():.4f}")
print(f"min={df['target'].min():.4f}, max={df['target'].max():.4f}")

# test.csv
test = pd.read_csv(f"{DATA}/test.csv")
print(f"\ntest shape: {test.shape}")
print(test.head())

# features
X_train = sparse.load_npz(f"{DATA}/features/hashed_smiles_train.npz")
X_test = sparse.load_npz(f"{DATA}/features/hashed_smiles_test.npz")
print(f"\nhashed features train: {X_train.shape}, test: {X_test.shape}")
print(f"nnz train: {X_train.nnz}, test: {X_test.nnz}")

# check scaffold split overlap
df_ids = set(df["id"])
train_set = set(train_ids)
val_set = set(val_ids)
print(f"\nscaffold split analysis:")
print(f"train_ids in df: {len(train_set & df_ids)}")
print(f"valid_ids in df: {len(val_set & df_ids)}")
print(f"overlap train/valid: {len(train_set & val_set)}")
print(f"valid not in train df: {len(val_set - df_ids)}")
print(f"note: {split.get('note', 'N/A')}")
print(f"method: {split.get('method', 'N/A')}")
# check target distribution in train vs valid
train_df_idx = df[df["id"].isin(train_set)].index
val_df_idx = df[df["id"].isin(val_set)].index
print(f"\ntrain scaffold target: mean={df.loc[train_df_idx, 'target'].mean():.4f}, std={df.loc[train_df_idx, 'target'].std():.4f}")
print(f"valid scaffold target: mean={df.loc[val_df_idx, 'target'].mean():.4f}, std={df.loc[val_df_idx, 'target'].std():.4f}")
