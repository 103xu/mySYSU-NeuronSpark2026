from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DexConfig


class VisionGridEncoder(nn.Module):
    """6×16×16 uint8 -> 128-dim."""

    def __init__(self, cfg: DexConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.vision_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(64, cfg.vision_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class TactileImageEncoder(nn.Module):
    """7×8×8 uint8 -> 128-dim."""

    def __init__(self, cfg: DexConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.tactile_fingers, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
        )
        self.proj = nn.Linear(64 * 2 * 2, cfg.tactile_img_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class TactileHeatmapEncoder(nn.Module):
    """7×4 float32 -> 64-dim."""

    def __init__(self, cfg: DexConfig):
        super().__init__()
        in_dim = cfg.tactile_heatmap_rows * cfg.tactile_heatmap_cols
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, cfg.tactile_heatmap_out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class LowDimStateEncoder(nn.Module):
    """14 float32 -> 128-dim."""

    def __init__(self, cfg: DexConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.low_dim_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.low_dim_out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContactSummaryEncoder(nn.Module):
    """4 float32 -> 32-dim."""

    def __init__(self, cfg: DexConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.contact_summary_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, cfg.contact_summary_out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TactileHistoryEncoder(nn.Module):
    """(6, 5) float32 -> 64-dim via GRU."""

    def __init__(self, cfg: DexConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=cfg.tactile_history_values,
            hidden_size=cfg.tactile_history_out_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return h.squeeze(0)


class StageContextEncoder(nn.Module):
    """4 float32 -> 32-dim."""

    def __init__(self, cfg: DexConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.stage_context_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, cfg.stage_context_out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskTypeEmbedding(nn.Module):
    """int [0,1,2] -> 16-dim."""

    def __init__(self, cfg: DexConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.num_task_types, cfg.task_type_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x.long())


class SensorStatusEncoder(nn.Module):
    """3 float32 -> 16-dim."""

    def __init__(self, cfg: DexConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.sensor_status_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, cfg.sensor_status_out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
