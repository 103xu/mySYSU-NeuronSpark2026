from __future__ import annotations

from dataclasses import dataclass, field

PRIMITIVE_NAMES = [
    "brace", "push", "drag", "pivot", "roll",
    "lift_edge", "tap", "stabilize", "wait", "finish",
]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]
TASK_TYPES = ["nonprehensile_relocation", "tool_use", "resource_sequence"]

PRIMITIVE_TO_IDX = {name: i for i, name in enumerate(PRIMITIVE_NAMES)}
FINGER_TO_IDX = {name: i for i, name in enumerate(FINGER_NAMES)}
TASK_TYPE_TO_IDX = {name: i for i, name in enumerate(TASK_TYPES)}


@dataclass
class DexConfig:
    # spatial modalities
    vision_channels: int = 6
    vision_size: int = 16
    tactile_fingers: int = 7
    tactile_size: int = 8
    tactile_heatmap_rows: int = 7
    tactile_heatmap_cols: int = 4

    # vector modalities
    low_dim_size: int = 14
    contact_summary_size: int = 4
    tactile_history_steps: int = 6
    tactile_history_values: int = 5
    stage_context_size: int = 4
    sensor_status_size: int = 3

    # encoder output dims (wider)
    vision_out_dim: int = 128
    tactile_img_out_dim: int = 128
    tactile_heatmap_out_dim: int = 64
    low_dim_out_dim: int = 128
    contact_summary_out_dim: int = 32
    tactile_history_out_dim: int = 64
    stage_context_out_dim: int = 32
    task_type_embed_dim: int = 32
    sensor_status_out_dim: int = 16

    # fusion (deeper & wider)
    fusion_input_dim: int = 624
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [768, 768, 512])
    fusion_dropout: float = 0.15
    film_hidden_dim: int = 128

    # output heads
    num_primitives: int = 10
    num_fingers: int = 7
    num_task_types: int = 3

    # training
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    grad_clip_norm: float = 1.0
    validation_interval: int = 2
    aux_task_weight: float = 0.1
