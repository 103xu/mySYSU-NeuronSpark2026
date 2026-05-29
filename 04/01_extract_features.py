"""特征工程：多维分子特征提取"""
import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings("ignore")

DATA = r"C:\Users\32010\ai-competition\04\2a9fa6fc-563a-43b9-b8bb-1301289bb22d"
OUTPUT = r"C:\Users\32010\ai-competition\04"


def load_data():
    df = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    return df, test


def mol_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
    except Exception:
        pass
    # fallback: sanitize
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is not None:
            return mol
    except Exception:
        pass
    return None


def compute_morgan_fp(mol, radius=3, nbits=2048):
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(nbits, dtype=np.float32)


def compute_maccs(mol):
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(167, dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(167, dtype=np.float32)


def compute_atompair_fp(mol, nbits=2048):
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    try:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(nbits, dtype=np.float32)


def get_descriptor_dim():
    return len(Descriptors._descList)


def compute_rdkit_descriptors(mol):
    dim = get_descriptor_dim()
    if mol is None:
        return np.zeros(dim, dtype=np.float32)
    try:
        desc_names = [name for name, _ in Descriptors._descList]
        values = []
        for name in desc_names:
            try:
                fn = getattr(Descriptors, name)
                values.append(float(fn(mol)))
            except Exception:
                values.append(0.0)
        return np.array(values, dtype=np.float32)
    except Exception:
        return np.zeros(dim, dtype=np.float32)


def compute_extra_features(mol, smiles):
    """Additional handcrafted features from SMILES string and mol object."""
    feats = []
    s = str(smiles)
    feats.append(float(len(s)))  # SMILES length
    feats.append(float(s.count("(") + s.count(")")))  # branches
    feats.append(float(s.count("=")))  # double bonds
    feats.append(float(s.count("#")))  # triple bonds
    feats.append(float(s.count("+") + s.count("-")))  # charged
    feats.append(float(s.count("@")))  # chiral
    feats.append(float(sum(1 for c in s if c.isdigit())))  # rings

    if mol is not None:
        try:
            feats.append(float(rdMolDescriptors.CalcNumRotatableBonds(mol)))
            feats.append(float(rdMolDescriptors.CalcNumHBD(mol)))
            feats.append(float(rdMolDescriptors.CalcNumHBA(mol)))
            feats.append(float(rdMolDescriptors.CalcNumRings(mol)))
            feats.append(float(rdMolDescriptors.CalcNumAromaticRings(mol)))
            feats.append(float(rdMolDescriptors.CalcNumSaturatedRings(mol)))
            feats.append(float(rdMolDescriptors.CalcNumAliphaticRings(mol)))
            feats.append(float(rdMolDescriptors.CalcNumHeterocycles(mol)))
            feats.append(float(Descriptors.MolWt(mol)))
            feats.append(float(Descriptors.MolLogP(mol)))
            feats.append(float(Descriptors.TPSA(mol)))
            feats.append(float(Descriptors.FractionCSP3(mol)))
            feats.append(float(Descriptors.NumValenceElectrons(mol)))
            feats.append(float(Descriptors.HeavyAtomCount(mol)))
        except Exception:
            feats.extend([0.0] * 14)
    else:
        feats.extend([0.0] * 14)
    return np.array(feats, dtype=np.float32)


def process_smiles_list(smiles_list, desc="processing"):
    n = len(smiles_list)
    desc_dim = get_descriptor_dim()
    morgan_r2 = np.zeros((n, 2048), dtype=np.float32)
    morgan_r3 = np.zeros((n, 2048), dtype=np.float32)
    maccs = np.zeros((n, 167), dtype=np.float32)
    atompair = np.zeros((n, 2048), dtype=np.float32)
    rdkit_desc = np.zeros((n, desc_dim), dtype=np.float32)
    extra = np.zeros((n, 21), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        if (i + 1) % 500 == 0:
            print(f"  {desc}: {i+1}/{n}")
        mol = mol_from_smiles(smi)
        morgan_r2[i] = compute_morgan_fp(mol, radius=2, nbits=2048)
        morgan_r3[i] = compute_morgan_fp(mol, radius=3, nbits=2048)
        maccs[i] = compute_maccs(mol)
        atompair[i] = compute_atompair_fp(mol, nbits=2048)
        rdkit_desc[i] = compute_rdkit_descriptors(mol)
        extra[i] = compute_extra_features(mol, smi)

    return {
        "morgan_r2": morgan_r2,
        "morgan_r3": morgan_r3,
        "maccs": maccs,
        "atompair": atompair,
        "rdkit_desc": rdkit_desc,
        "extra": extra,
    }


def main():
    print("=" * 60)
    print("特征工程：多维分子特征")
    print("=" * 60)

    df, test = load_data()
    train_smiles = df["smiles"].tolist()
    test_smiles = test["smiles"].tolist()
    print(f"训练集: {len(train_smiles)} 分子")
    print(f"测试集: {len(test_smiles)} 分子")

    # load hashed features
    print("\n加载预提取 hashed features...")
    hashed_train = sparse.load_npz(f"{DATA}/features/hashed_smiles_train.npz").toarray()
    hashed_test = sparse.load_npz(f"{DATA}/features/hashed_smiles_test.npz").toarray()
    print(f"hashed train: {hashed_train.shape}, test: {hashed_test.shape}")

    # compute RDKit features
    print("\n计算 RDKit 特征 (训练集)...")
    train_feats = process_smiles_list(train_smiles, desc="train")
    print("计算 RDKit 特征 (测试集)...")
    test_feats = process_smiles_list(test_smiles, desc="test")

    # combine all features
    print("\n合并所有特征...")
    X_train_parts = [hashed_train]
    X_test_parts = [hashed_test]
    feat_names = []

    for name, feat in train_feats.items():
        X_train_parts.append(feat)
        X_test_parts.append(test_feats[name])
        dim = feat.shape[1]
        feat_names.append(f"{name}:{dim}")
        print(f"  {name}: {dim} dims")

    X_train_all = np.concatenate(X_train_parts, axis=1).astype(np.float32)
    X_test_all = np.concatenate(X_test_parts, axis=1).astype(np.float32)
    print(f"\n总特征维度: {X_train_all.shape}")

    # handle NaN/Inf
    X_train_all = np.nan_to_num(X_train_all, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=1e6, neginf=-1e6)

    # clip extreme values
    X_train_all = np.clip(X_train_all, -1e6, 1e6)
    X_test_all = np.clip(X_test_all, -1e6, 1e6)

    # save
    print("\n保存特征...")
    np.savez_compressed(f"{OUTPUT}/X_train_features.npz", X=X_train_all)
    np.savez_compressed(f"{OUTPUT}/X_test_features.npz", X=X_test_all)

    # save target
    df[["id", "target"]].to_csv(f"{OUTPUT}/y_train.csv", index=False)
    test[["id"]].to_csv(f"{OUTPUT}/test_ids.csv", index=False)

    print(f"\n特征保存完成!")
    print(f"  X_train: {X_train_all.shape} ({X_train_all.dtype})")
    print(f"  X_test:  {X_test_all.shape} ({X_test_all.dtype})")


if __name__ == "__main__":
    main()
