from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class DexDemoDataset(Dataset):
    """Parses weak_train_rollouts.jsonl into (observation, action) step pairs.

    Builds tactile_history as a sliding window. Splits at the episode level.
    Supports quality filtering via min_score01.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        split: str = "train",
        split_ratio: float = 0.9,
        augment: bool = False,
        min_score01: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.split = split
        self.augment = augment and split == "train"
        self.rng = random.Random(seed)

        episodes = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))

        rng = random.Random(seed)
        rng.shuffle(episodes)

        total = len(episodes)
        # filter by episode quality
        if min_score01 > 0.0:
            episodes = self._filter_episodes(episodes, min_score01)

        split_idx = int(len(episodes) * split_ratio)

        if split == "train":
            episodes = episodes[:split_idx]
        else:
            episodes = episodes[split_idx:]

        self.samples: list[dict[str, Any]] = []
        for ep in episodes:
            self._process_episode(ep)

    def _filter_episodes(self, episodes, min_score01):
        kept = []
        for ep in episodes:
            metrics = ep.get("metrics", {})
            score = _safe_float(metrics.get("score01", 0.0))
            if score >= min_score01:
                kept.append(ep)
        return kept

    def _process_episode(self, episode: dict[str, Any]) -> None:
        task_type = episode.get("task_type", "nonprehensile_relocation")
        task_type_id = {"nonprehensile_relocation": 0, "tool_use": 1, "resource_sequence": 2}.get(task_type, 0)

        variant_id = episode.get("variant_id", "proxy_0")
        variant_idx = 0
        if "proxy_1" in str(variant_id):
            variant_idx = 1
        elif "proxy_2" in str(variant_id):
            variant_idx = 2

        pose_dropout_base = [0.0165, 0.0118, 0.0296][variant_idx]
        tactile_dropout_base = [0.0275, 0.0469, 0.0037][variant_idx]
        action_delay_hint = 0.0

        # episode score as sample weight
        metrics = episode.get("metrics", {})
        ep_weight = max(0.1, _safe_float(metrics.get("score01", 0.3)))

        history: list[list[float]] = []
        steps = episode.get("steps", [])
        for step in steps:
            low_dim_state = step.get("low_dim_state", [0.0] * 14)
            contact_summary = step.get("contact_summary", {})
            stage_ctx = step.get("stage_context", {})
            tactile_heatmap = step.get("tactile_heatmap_7x4", [[0.0] * 4] * 7)
            action = step.get("action", {})

            if not isinstance(low_dim_state, list) or len(low_dim_state) < 14:
                continue

            if isinstance(tactile_heatmap, list) and len(tactile_heatmap) >= 7:
                hm = []
                for i in range(7):
                    row = tactile_heatmap[i] if i < len(tactile_heatmap) else [0.0] * 4
                    if isinstance(row, list) and len(row) >= 4:
                        hm.append([_safe_float(row[j]) for j in range(4)])
                    else:
                        hm.append([0.0, 0.0, 0.0, 0.0])
            else:
                hm = [[0.0] * 4 for _ in range(7)]

            normal = sum(hm[i][0] for i in range(7)) / 7.0
            shear = sum(hm[i][1] for i in range(7)) / 7.0
            slip_risk = _safe_float(contact_summary.get("slip_risk", 0.0))
            contact = _safe_float(contact_summary.get("coverage", 0.0))
            damage_risk = _safe_float(contact_summary.get("damage_risk", 0.0))
            current_tactile = [normal, shear, slip_risk, contact, damage_risk]

            history.append(current_tactile)
            if len(history) > 6:
                history = history[-6:]

            padded_history = [[0.0] * 5 for _ in range(6)]
            offset = 6 - len(history)
            for i, h in enumerate(history):
                idx = offset + i
                if 0 <= idx < 6:
                    padded_history[idx] = list(h)

            stage_enabled = float(bool(stage_ctx.get("enabled", False)))
            stage_index = _safe_float(stage_ctx.get("current_stage_index", 0)) / max(1.0, _safe_float(stage_ctx.get("stage_count", 1)))
            stage_count = min(_safe_float(stage_ctx.get("stage_count", 0)) / 8.0, 1.0)
            completion = _safe_float(stage_ctx.get("completion_fraction", 1.0))
            stage_features = [stage_enabled, stage_index, stage_count, completion]

            primitive = action.get("primitive", "wait")
            finger = action.get("finger", "palm")
            force = _safe_float(action.get("force", 0.0))
            direction = action.get("direction", [0.0, 0.0])
            if not isinstance(direction, list) or len(direction) < 2:
                direction = [0.0, 0.0]

            primitive_idx = {
                "brace": 0, "push": 1, "drag": 2, "pivot": 3, "roll": 4,
                "lift_edge": 5, "tap": 6, "stabilize": 7, "wait": 8, "finish": 9,
            }.get(primitive, 8)

            finger_idx = {
                "thumb": 0, "index": 1, "middle": 2, "ring": 3,
                "pinky": 4, "palm": 5, "wrist": 6,
            }.get(finger, 5)

            sample = {
                "low_dim_state": [float(v) for v in low_dim_state[:14]],
                "contact_summary": [
                    _safe_float(contact_summary.get("coverage", 0.0)),
                    _safe_float(contact_summary.get("min_contact_so_far", 0.0)),
                    _safe_float(contact_summary.get("slip_risk", 0.0)),
                    _safe_float(contact_summary.get("damage_risk", 0.0)),
                ],
                "tactile_heatmap": hm,
                "stage_context": stage_features,
                "tactile_history": padded_history,
                "task_type": task_type_id,
                "sensor_status": [pose_dropout_base, tactile_dropout_base, action_delay_hint],
                "primitive": primitive_idx,
                "finger": finger_idx,
                "force": force,
                "direction": [float(direction[0]), float(direction[1])],
                "sample_weight": ep_weight,
            }
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if self.augment:
            sample = self._augment(sample)

        return {
            "low_dim_state": torch.tensor(sample["low_dim_state"], dtype=torch.float32),
            "contact_summary": torch.tensor(sample["contact_summary"], dtype=torch.float32),
            "tactile_heatmap": torch.tensor(sample["tactile_heatmap"], dtype=torch.float32),
            "stage_context": torch.tensor(sample["stage_context"], dtype=torch.float32),
            "tactile_history": torch.tensor(sample["tactile_history"], dtype=torch.float32),
            "task_type": torch.tensor(sample["task_type"], dtype=torch.long),
            "sensor_status": torch.tensor(sample["sensor_status"], dtype=torch.float32),
            "vision_grid": torch.zeros(6, 16, 16, dtype=torch.float32),
            "tactile_image": torch.zeros(7, 8, 8, dtype=torch.float32),
            "sample_weight": torch.tensor(sample["sample_weight"], dtype=torch.float32),
            "primitive_target": torch.tensor(sample["primitive"], dtype=torch.long),
            "finger_target": torch.tensor(sample["finger"], dtype=torch.long),
            "force_target": torch.tensor(sample["force"], dtype=torch.float32),
            "direction_target": torch.tensor(sample["direction"], dtype=torch.float32),
        }

    def _augment(self, sample: dict[str, Any]) -> dict[str, Any]:
        rng = self.rng
        sample = dict(sample)

        low_dim = list(sample["low_dim_state"])
        for i in range(len(low_dim)):
            low_dim[i] += rng.gauss(0.0, 0.018)
        sample["low_dim_state"] = low_dim

        if rng.random() < 0.15:
            hm = sample["tactile_heatmap"]
            drop_count = rng.randint(1, 3)
            drop_rows = rng.sample(range(7), drop_count)
            hm = [row[:] for row in hm]
            for r in drop_rows:
                hm[r] = [0.0, 0.0, 0.0, 0.0]
            sample["tactile_heatmap"] = hm

        cs = list(sample["contact_summary"])
        for i in range(len(cs)):
            cs[i] += rng.gauss(0.0, 0.015)
        sample["contact_summary"] = [max(0.0, min(1.0, v)) for v in cs]

        hist = [row[:] for row in sample["tactile_history"]]
        for row in hist:
            for i in range(len(row)):
                row[i] += rng.gauss(0.0, 0.010)
        sample["tactile_history"] = hist

        sample["force"] = max(0.0, min(1.0, sample["force"] + rng.gauss(0.0, 0.04)))

        d = list(sample["direction"])
        d[0] += rng.gauss(0.0, 0.030)
        d[1] += rng.gauss(0.0, 0.030)
        n = (d[0] ** 2 + d[1] ** 2) ** 0.5
        if n > 1e-8:
            d[0] /= n
            d[1] /= n
        sample["direction"] = d

        sensor = list(sample["sensor_status"])
        sensor[0] = max(0.0, min(1.0, sensor[0] + rng.gauss(0.0, 0.04)))
        sensor[1] = max(0.0, min(1.0, sensor[1] + rng.gauss(0.0, 0.04)))
        sample["sensor_status"] = sensor

        return sample


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch]) for key in keys}
