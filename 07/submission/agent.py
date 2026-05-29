from __future__ import annotations

import math
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parent / "model"


def _norm(vec):
    x, y = float(vec[0]), float(vec[1])
    n = math.sqrt(x * x + y * y)
    if n < 1e-9:
        return [0.0, 0.0]
    return [x / n, y / n]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_model():
    """Attempt to load the trained policy. Returns None on failure."""
    try:
        checkpoint = MODEL_DIR / "best_model.pt"
        if not checkpoint.exists():
            return None

        from model.config import DexConfig
        from model.policy import DexPolicy

        cfg = DexConfig()
        model = DexPolicy(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, device, cfg
    except Exception:
        return None


_MODEL_CACHE = None


def _get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _load_model()
    return _MODEL_CACHE


def _extract_tactile_vector(observation):
    """Extract a 5-dim tactile summary from observation."""
    heatmap = observation.get("tactile_heatmap_7x4", [[0.0] * 4] * 7)
    contact = observation.get("contact_summary", {})

    if isinstance(heatmap, list) and len(heatmap) >= 7:
        normal = sum(
            _safe_float(heatmap[i][0]) if i < len(heatmap) and isinstance(heatmap[i], list) and len(heatmap[i]) >= 1 else 0.0
            for i in range(7)
        ) / 7.0
        shear = sum(
            _safe_float(heatmap[i][1]) if i < len(heatmap) and isinstance(heatmap[i], list) and len(heatmap[i]) >= 2 else 0.0
            for i in range(7)
        ) / 7.0
    else:
        normal = 0.0
        shear = 0.0

    slip_risk = _safe_float(contact.get("slip_risk", 0.0))
    coverage = _safe_float(contact.get("coverage", 0.0))
    damage_risk = _safe_float(contact.get("damage_risk", 0.0))
    return [normal, shear, slip_risk, coverage, damage_risk]


def _build_model_input(observation, tactile_history, task_type_id, variant_idx, step_count):
    """Convert observation dict to model input tensor dict (single sample, batched)."""
    pose_dropout_base = [0.0165, 0.0118, 0.0296][variant_idx]
    tactile_dropout_base = [0.0275, 0.0469, 0.0037][variant_idx]

    # low_dim_state
    state = observation.get("low_dim_state", {})
    if isinstance(state, dict):
        raw = state.get("values", [0.0] * 14)
    elif isinstance(state, list):
        raw = state
    else:
        raw = [0.0] * 14
    low_dim = [_safe_float(v) for v in raw[:14]]

    # tactile_heatmap
    heatmap = observation.get("tactile_heatmap_7x4", [[0.0] * 4] * 7)
    if isinstance(heatmap, list) and len(heatmap) >= 7:
        hm = []
        for i in range(7):
            row = heatmap[i] if i < len(heatmap) else [0.0] * 4
            if isinstance(row, list) and len(row) >= 4:
                hm.append([_safe_float(row[j]) for j in range(4)])
            else:
                hm.append([0.0, 0.0, 0.0, 0.0])
    else:
        hm = [[0.0] * 4 for _ in range(7)]

    # contact_summary
    contact = observation.get("contact_summary", {})
    cs = [
        _safe_float(contact.get("coverage", 0.0)),
        _safe_float(contact.get("min_contact_so_far", 0.0)),
        _safe_float(contact.get("slip_risk", 0.0)),
        _safe_float(contact.get("damage_risk", 0.0)),
    ]

    # stage_context
    stage_ctx = observation.get("stage_context", {})
    stage_enabled = float(bool(stage_ctx.get("enabled", False)))
    stage_index = _safe_float(stage_ctx.get("current_stage_index", 0)) / max(1.0, _safe_float(stage_ctx.get("stage_count", 1)))
    stage_count = min(_safe_float(stage_ctx.get("stage_count", 0)) / 8.0, 1.0)
    completion = _safe_float(stage_ctx.get("completion_fraction", 1.0))
    stage_features = [stage_enabled, stage_index, stage_count, completion]

    # tactile_history
    padded_history = [[0.0] * 5 for _ in range(6)]
    offset = 6 - len(tactile_history)
    for i, h in enumerate(tactile_history):
        idx = offset + i
        if 0 <= idx < 6:
            padded_history[idx] = list(h)

    return {
        "low_dim_state": torch.tensor([low_dim], dtype=torch.float32),
        "contact_summary": torch.tensor([cs], dtype=torch.float32),
        "tactile_heatmap": torch.tensor([hm], dtype=torch.float32),
        "stage_context": torch.tensor([stage_features], dtype=torch.float32),
        "tactile_history": torch.tensor([padded_history], dtype=torch.float32),
        "task_type": torch.tensor([task_type_id], dtype=torch.long),
        "sensor_status": torch.tensor([[pose_dropout_base, tactile_dropout_base, 0.0]], dtype=torch.float32),
        "vision_grid": torch.zeros(1, 6, 16, 16, dtype=torch.float32),
        "tactile_image": torch.zeros(1, 7, 8, 8, dtype=torch.float32),
    }


class Agent:
    """Hybrid closed-loop agent: model predicts direction/force, rules govern safety and task structure."""

    def __init__(self):
        self._step = 0
        self._tactile_history: list[list[float]] = []
        self._batch_agents = None
        self.task_type = "nonprehensile_relocation"
        self.task_type_id = 0
        self.tool_axis = [1.0, 0.0]
        self.reserved = []
        self.obstacles = []
        self.fragile = False
        self.taps_done = 0
        self.variant_idx = 0

    def reset(self, task_info):
        self._step = 0
        self._tactile_history = []
        self.taps_done = 0

        self.task_type = task_info.get("task_type", "nonprehensile_relocation")
        self.task_type_id = {"nonprehensile_relocation": 0, "tool_use": 1, "resource_sequence": 2}.get(
            self.task_type, 0
        )
        self.tool_axis = task_info.get("tool_goal", {}).get("axis", [1.0, 0.0])
        self.reserved = task_info.get("resource_constraint", {}).get("reserve_fingers", [])

        variant_id = task_info.get("variant_id", "proxy_0")
        if "proxy_1" in str(variant_id):
            self.variant_idx = 1
        elif "proxy_2" in str(variant_id):
            self.variant_idx = 2
        else:
            self.variant_idx = 0

        scene = task_info.get("scene_context", {})
        self.obstacles = scene.get("obstacles", []) if isinstance(scene, dict) else []

        obj = task_info.get("object", {})
        self.fragile = float(obj.get("fragility", 0.0)) > 0.58

    def reset_batch(self, task_infos):
        self._batch_agents = [Agent() for _ in task_infos]
        for agent, ti in zip(self._batch_agents, task_infos):
            agent.reset(ti)

    def act_batch(self, observations):
        if self._batch_agents is None:
            raise RuntimeError("reset_batch must be called before act_batch")

        model_result = _get_model()
        if model_result is not None:
            model, device, _cfg = model_result
            return self._model_act_batch(observations, model, device)

        return [self._batch_agents[i].act(obs) for i, obs in enumerate(observations)]

    def _model_act_batch(self, observations, model, device):
        batch_inputs = {
            "low_dim_state": [],
            "contact_summary": [],
            "tactile_heatmap": [],
            "stage_context": [],
            "tactile_history": [],
            "task_type": [],
            "sensor_status": [],
            "vision_grid": [],
            "tactile_image": [],
        }

        for i, obs in enumerate(observations):
            agent = self._batch_agents[i]
            agent._step += 1

            tactile_vec = _extract_tactile_vector(obs)
            agent._tactile_history.append(tactile_vec)
            if len(agent._tactile_history) > 6:
                agent._tactile_history = agent._tactile_history[-6:]

            inp = _build_model_input(
                obs, agent._tactile_history, agent.task_type_id, agent.variant_idx, agent._step
            )
            for k in batch_inputs:
                batch_inputs[k].append(inp[k])

        model_input = {k: torch.cat(v, dim=0).to(device) for k, v in batch_inputs.items()}

        with torch.no_grad():
            with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
                predictions = model(model_input)

        actions = []
        for i, obs in enumerate(observations):
            agent = self._batch_agents[i]

            pred_primitive = predictions["primitive"][i].argmax(-1).item()
            pred_finger = predictions["finger"][i].argmax(-1).item()
            pred_force = predictions["force"][i].item()
            pred_dir = predictions["direction"][i].tolist()

            action = agent._hybrid_act(obs, pred_primitive, pred_finger, pred_force, pred_dir)
            actions.append(action)

        return actions

    def act(self, observation):
        self._step += 1

        tactile_vec = _extract_tactile_vector(observation)
        self._tactile_history.append(tactile_vec)
        if len(self._tactile_history) > 6:
            self._tactile_history = self._tactile_history[-6:]

        model_result = _get_model()
        if model_result is not None:
            model, device, _cfg = model_result
            inp = _build_model_input(
                observation, self._tactile_history, self.task_type_id, self.variant_idx, self._step
            )
            inp = {k: v.to(device) for k, v in inp.items()}

            with torch.no_grad():
                with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
                    predictions = model(inp)

            pred_primitive = predictions["primitive"][0].argmax(-1).item()
            pred_finger = predictions["finger"][0].argmax(-1).item()
            pred_force = predictions["force"][0].item()
            pred_dir = predictions["direction"][0].tolist()
        else:
            pred_primitive = 8  # wait
            pred_finger = 5   # palm
            pred_force = 0.75
            pred_dir = [0.0, 0.0]

        return self._hybrid_act(observation, pred_primitive, pred_finger, pred_force, pred_dir)

    def _hybrid_act(self, observation, pred_primitive, pred_finger, pred_force, pred_dir):
        PRIMITIVES = ["brace", "push", "drag", "pivot", "roll", "lift_edge", "tap", "stabilize", "wait", "finish"]
        FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]

        state = observation.get("low_dim_state", {})
        if isinstance(state, dict):
            raw = state.get("values", [0.0] * 14)
        else:
            raw = state
        dx, dy, dtheta = _safe_float(raw[0]), _safe_float(raw[1]), _safe_float(raw[2])
        dist = math.sqrt(dx * dx + dy * dy)
        contact = _safe_float(observation.get("contact_summary", {}).get("coverage", 0.0))
        slip_risk = _safe_float(observation.get("contact_summary", {}).get("slip_risk", 0.0))
        damage_risk = _safe_float(observation.get("contact_summary", {}).get("damage_risk", 0.0))

        # Rule overrides for safety and task structure

        # 1. Finish when close enough
        if dist < 0.085 and abs(dtheta) < 0.34:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}

        # 2. Brace when contact lost or risk is critically high
        if self._step <= 1 or contact < 0.04 or slip_risk > 0.92 or damage_risk > 0.98:
            return {"primitive": "brace", "finger": "palm", "force": 0.30, "direction": [0.0, 0.0]}

        # 3. Task-specific taps for tool_use / resource_sequence
        if self.task_type in {"tool_use", "resource_sequence"} and self.taps_done < 8 and self._step % 3 == 0:
            self.taps_done += 1
            finger = self.reserved[0] if self.task_type == "resource_sequence" and self.reserved else "index"
            return {
                "primitive": "tap",
                "finger": finger,
                "force": self._cap(0.85, damage_risk),
                "direction": _norm(self.tool_axis),
            }

        # 4. Correct orientation with pivot
        if abs(dtheta) > 0.22:
            return {
                "primitive": "pivot",
                "finger": "thumb",
                "force": self._cap(0.95, damage_risk),
                "direction": [1.0 if dtheta >= 0.0 else -1.0, 0.0],
            }

        # 5. Model-guided manipulation
        model_primitive = PRIMITIVES[pred_primitive] if 0 <= pred_primitive < 10 else "push"
        model_finger = FINGERS[pred_finger] if 0 <= pred_finger < 7 else "palm"

        if model_primitive in {"push", "drag", "roll", "lift_edge"}:
            primitive = model_primitive
        elif model_primitive == "finish":
            primitive = "push"
        elif model_primitive in {"brace", "stabilize"}:
            primitive = "brace"
        else:
            primitive = "push"

        mx, my = pred_dir[0], pred_dir[1]
        if abs(mx) + abs(my) < 1e-6:
            direction = _norm([dx, dy])
        else:
            direction = _norm([mx, my])

        direction = self._avoid_obstacles(direction, observation)

        force = self._cap(pred_force if pred_force > 0.1 else 0.75, damage_risk)

        return {
            "primitive": primitive,
            "finger": model_finger,
            "force": force,
            "direction": direction,
        }

    def _cap(self, force, damage_risk):
        if self.fragile and damage_risk > 0.10:
            return min(force, 0.68)
        return force

    def _avoid_obstacles(self, direction, obs):
        pose = obs.get("object_pose_estimate", {})
        ox = float(pose.get("x", 0.0))
        oy = float(pose.get("y", 0.0))
        ax, ay = float(direction[0]), float(direction[1])
        for ob in self.obstacles:
            try:
                cx = float(ob.get("x", 0.0))
                cy = float(ob.get("y", 0.0))
                r = float(ob.get("radius", 0.08))
            except Exception:
                continue
            dx, dy = ox - cx, oy - cy
            d2 = dx * dx + dy * dy
            if d2 < (r + 0.18) ** 2 and d2 > 1e-8:
                scale = 0.20 / max(0.08, math.sqrt(d2))
                ax += dx * scale
                ay += dy * scale
        return _norm([ax, ay])
