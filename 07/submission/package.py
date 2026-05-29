#!/usr/bin/env python3
"""Package the submission into a competition-ready zip file.

Usage:
    python package.py [--output NS-2026-07-answer.zip]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path


def sha256_hex(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Package NS-2026-07 submission")
    parser.add_argument("--output", default="NS-2026-07-answer.zip")
    parser.add_argument("--root", default=None, help="Submission root (default: parent of this script)")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    output = Path(args.output)

    # required files
    required = [
        root / "agent.py",
        root / "model" / "config.py",
        root / "model" / "encoders.py",
        root / "model" / "policy.py",
        root / "model" / "best_model.pt",
    ]
    for r in required:
        if not r.exists():
            print(f"ERROR: Missing required file: {r}")
            return 1

    # generate model manifest
    model_dir = root / "model"
    checkpoint = model_dir / "best_model.pt"
    size_bytes = checkpoint.stat().st_size
    sha = sha256_hex(checkpoint)

    manifest = {
        "model_architecture": "DexPolicy",
        "framework": "pytorch",
        "pytorch_version": "2.12.0",
        "parameter_count": "1.84M",
        "training_dataset": "NS-2026-07 demonstrations/weak_train_rollouts.jsonl (min_score01=0.1 filtered)",
        "training_epochs": 60,
        "best_val_loss": 0.1164,
        "checkpoint_format": "state_dict",
        "vram_requirement_mb": "~300 (FP16 batch=64)",
        "inference_time_ms": "~3 (single) / ~50 (batch=64)",
        "files": [
            {
                "path": "model/best_model.pt",
                "size_bytes": size_bytes,
                "sha256": sha,
            }
        ],
    }

    manifest_path = model_dir / "model_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # files to include
    include = [
        "agent.py",
        "model/__init__.py",
        "model/config.py",
        "model/encoders.py",
        "model/policy.py",
        "model/best_model.pt",
        "model/model_manifest.json",
    ]

    # training/ directory NOT included in submission (only used for offline training)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path in include:
            file_path = root / rel_path
            if file_path.exists():
                zf.write(file_path, rel_path)
                print(f"  added: {rel_path}")
            else:
                print(f"  WARNING: {rel_path} not found, skipping")

    zip_size = output.stat().st_size
    print(f"\nSubmission zip: {output} ({zip_size / 1024:.1f} KB)")

    # verify
    if zip_size > 8192 * 1024 * 1024:
        print("WARNING: zip exceeds 8192 MB limit!")
    if zip_size > 512 * 1024 * 1024:
        print("WARNING: zip exceeds 512 MB single-file limit! Consider sharding.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
