from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLoss(nn.Module):
    """Weighted multi-head loss with per-sample weights and auxiliary task head."""

    def __init__(
        self,
        primitive_weight: float = 1.0,
        finger_weight: float = 0.7,
        force_weight: float = 1.0,
        direction_weight: float = 1.5,
        aux_weight: float = 0.1,
        class_weights_primitive: torch.Tensor | None = None,
        class_weights_finger: torch.Tensor | None = None,
    ):
        super().__init__()
        self.primitive_w = primitive_weight
        self.finger_w = finger_weight
        self.force_w = force_weight
        self.direction_w = direction_weight
        self.aux_w = aux_weight

        self.ce_primitive = nn.CrossEntropyLoss(weight=class_weights_primitive, reduction="none")
        self.ce_finger = nn.CrossEntropyLoss(weight=class_weights_finger, reduction="none")
        self.mse_force = nn.MSELoss(reduction="none")
        self.ce_aux = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        sample_w = targets.get("sample_weight", None)

        loss_primitive = self.ce_primitive(predictions["primitive"], targets["primitive"])
        loss_finger = self.ce_finger(predictions["finger"], targets["finger"])
        loss_force = self.mse_force(predictions["force"], targets["force"])

        cos_sim = F.cosine_similarity(
            predictions["direction"], targets["direction"], dim=-1
        )
        loss_direction = 1.0 - cos_sim

        loss_aux = self.ce_aux(predictions["task_aux"], targets["task_type"])

        per_sample = (
            self.primitive_w * loss_primitive
            + self.finger_w * loss_finger
            + self.force_w * loss_force
            + self.direction_w * loss_direction
            + self.aux_w * loss_aux
        )

        if sample_w is not None:
            per_sample = per_sample * sample_w

        total = per_sample.mean()

        with torch.no_grad():
            prim_acc = (predictions["primitive"].argmax(-1) == targets["primitive"]).float().mean()
            finger_acc = (predictions["finger"].argmax(-1) == targets["finger"]).float().mean()
            force_mae = (predictions["force"] - targets["force"]).abs().mean()
            dir_cos = cos_sim.mean()
            aux_acc = (predictions["task_aux"].argmax(-1) == targets["task_type"]).float().mean()

        metrics = {
            "loss_primitive": loss_primitive.mean().item(),
            "loss_finger": loss_finger.mean().item(),
            "loss_force": loss_force.mean().item(),
            "loss_direction": loss_direction.mean().item(),
            "loss_aux": loss_aux.mean().item(),
            "total_loss": total.item(),
            "primitive_acc": prim_acc.item(),
            "finger_acc": finger_acc.item(),
            "force_mae": force_mae.item(),
            "direction_cos_sim": dir_cos.item(),
            "aux_acc": aux_acc.item(),
        }
        return total, metrics
