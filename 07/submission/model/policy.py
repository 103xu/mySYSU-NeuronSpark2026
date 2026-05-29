from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DexConfig
from .encoders import (
    ContactSummaryEncoder,
    LowDimStateEncoder,
    SensorStatusEncoder,
    StageContextEncoder,
    TactileHeatmapEncoder,
    TactileHistoryEncoder,
    TactileImageEncoder,
    TaskTypeEmbedding,
    VisionGridEncoder,
)


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation conditioned on a context vector."""

    def __init__(self, feature_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.bias_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale = self.scale_net(context)
        bias = self.bias_net(context)
        return features * (1.0 + scale) + bias


class DexPolicy(nn.Module):
    """Multi-modal policy with task-conditioned FiLM modulation."""

    def __init__(self, cfg: DexConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = DexConfig()
        self.cfg = cfg

        self.vision_encoder = VisionGridEncoder(cfg)
        self.tactile_img_encoder = TactileImageEncoder(cfg)
        self.tactile_heatmap_encoder = TactileHeatmapEncoder(cfg)
        self.low_dim_encoder = LowDimStateEncoder(cfg)
        self.contact_summary_encoder = ContactSummaryEncoder(cfg)
        self.tactile_history_encoder = TactileHistoryEncoder(cfg)
        self.stage_context_encoder = StageContextEncoder(cfg)
        self.task_type_embed = TaskTypeEmbedding(cfg)
        self.sensor_status_encoder = SensorStatusEncoder(cfg)

        prev_dim = cfg.fusion_input_dim
        layers = []
        for h_dim in cfg.fusion_hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(cfg.fusion_dropout))
            prev_dim = h_dim
        self.fusion = nn.Sequential(*layers)
        self.fusion_out_dim = cfg.fusion_hidden_dims[-1]

        self.film = FiLMBlock(self.fusion_out_dim, cfg.task_type_embed_dim, cfg.film_hidden_dim)

        self.primitive_head = nn.Linear(self.fusion_out_dim, cfg.num_primitives)
        self.finger_head = nn.Linear(self.fusion_out_dim, cfg.num_fingers)
        self.force_head = nn.Sequential(
            nn.Linear(self.fusion_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.direction_head = nn.Linear(self.fusion_out_dim, 2)
        self.task_aux_head = nn.Linear(self.fusion_out_dim, cfg.num_task_types)

    def _encode_modality(
        self, encoder: nn.Module, tensor: torch.Tensor | None,
        batch_size: int, default_dim: int, device: torch.device,
    ) -> torch.Tensor:
        if tensor is not None:
            return encoder(tensor)
        return torch.zeros(batch_size, default_dim, device=device)

    def forward(self, obs: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor]:
        B = obs["low_dim_state"].shape[0]
        device = obs["low_dim_state"].device

        features = [
            self._encode_modality(self.vision_encoder, obs.get("vision_grid"), B, self.cfg.vision_out_dim, device),
            self._encode_modality(self.tactile_img_encoder, obs.get("tactile_image"), B, self.cfg.tactile_img_out_dim, device),
            self._encode_modality(self.tactile_heatmap_encoder, obs.get("tactile_heatmap"), B, self.cfg.tactile_heatmap_out_dim, device),
            self.low_dim_encoder(obs["low_dim_state"]),
            self._encode_modality(self.contact_summary_encoder, obs.get("contact_summary"), B, self.cfg.contact_summary_out_dim, device),
            self._encode_modality(self.tactile_history_encoder, obs.get("tactile_history"), B, self.cfg.tactile_history_out_dim, device),
            self._encode_modality(self.stage_context_encoder, obs.get("stage_context"), B, self.cfg.stage_context_out_dim, device),
            self._encode_modality(self.task_type_embed, obs.get("task_type"), B, self.cfg.task_type_embed_dim, device),
            self._encode_modality(self.sensor_status_encoder, obs.get("sensor_status"), B, self.cfg.sensor_status_out_dim, device),
        ]

        fused = self.fusion(torch.cat(features, dim=-1))

        # FiLM modulation by task type
        task_embed = features[7]  # task_type embedding
        fused = self.film(fused, task_embed)

        direction_raw = self.direction_head(fused)
        direction = F.normalize(direction_raw, p=2, dim=-1)

        return {
            "primitive": self.primitive_head(fused),
            "finger": self.finger_head(fused),
            "force": self.force_head(fused).squeeze(-1),
            "direction": direction,
            "task_aux": self.task_aux_head(fused),
        }
