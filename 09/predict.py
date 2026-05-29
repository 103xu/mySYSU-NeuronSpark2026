"""
NS-2026-09 完整预测 pipeline
1. 对每个 context，使用 probe 数据校准模拟器参数
2. 对每个 query，运行校准后的模拟器进行反事实 rollout
3. 输出 results.json
"""

from __future__ import annotations

import json
import sys
import os
from collections import Counter
from simulator import Simulator, GRID_SIZE, EVENT_KEYS, grid_diff

# 将 simulator 模块路径加入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_entities_from_grid(grid: list[str]) -> dict:
    """从网格中提取实体位置"""
    entities = {}
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            ch = grid[y][x]
            if ch in 'ABOK':
                entities[ch] = f"{x},{y}"
    return entities


class ProfileSearch:
    """搜索最优 profile 参数"""

    # 搜索空间
    ICE_MODES = ["slide_until_blocked", "slide_one", "slide_none"]
    FIELD_MODES = ["push_one", "push_continuous", "push_none"]
    FIELD_PRIORITY = ["before_move", "after_move"]
    ICE_DIRECTIONS = ["momentum", "reverse"]

    @classmethod
    def default_profiles(cls) -> list[dict]:
        """生成候选 profile 列表"""
        profiles = []
        # 基础 profile
        base = Simulator._default_profile()
        profiles.append(base)

        # 不同 ice 模式
        for ice in cls.ICE_MODES:
            if ice != base["ice_mode"]:
                p = dict(base)
                p["ice_mode"] = ice
                profiles.append(p)

        # 不同 field 模式
        for fm in cls.FIELD_MODES:
            if fm != base["field_mode"]:
                p = dict(base)
                p["field_mode"] = fm
                profiles.append(p)

        # ice + field combinations
        for ice in cls.ICE_MODES:
            for fm in cls.FIELD_MODES:
                p = dict(base)
                p["ice_mode"] = ice
                p["field_mode"] = fm
                profiles.append(p)

        # field priority combos
        for fp in cls.FIELD_PRIORITY:
            p = dict(base)
            p["field_priority"] = fp
            profiles.append(p)

        return profiles

    @classmethod
    def evaluate_profile(cls, profile: dict, initial_grid: list[str],
                         episodes: list[dict]) -> float:
        """评估一个 profile 在 probe episodes 上的表现"""
        score = 0.0
        for ep in episodes:
            sim = Simulator(initial_grid, profile)
            result = sim.run(ep["actions"])
            pred_grid = result["final_grid"]
            true_grid = ep["observed_final_full_grid"]

            # 比较动态实体位置 (A, B, O, K)
            pred_entities = parse_entities_from_grid(pred_grid)
            true_entities = parse_entities_from_grid(true_grid)

            # Agent 位置匹配
            if pred_entities.get('A') == true_entities.get('A'):
                score += 3.0

            # Box 位置匹配
            if pred_entities.get('B') == true_entities.get('B'):
                score += 2.0

            # Orb 位置匹配
            if pred_entities.get('O') == true_entities.get('O'):
                score += 1.5

            # Key 位置匹配
            if pred_entities.get('K') == true_entities.get('K'):
                score += 1.0

            # 事件匹配
            pred_events = result["events"]
            true_events = ep["observed_final_events"]
            event_match = all(pred_events.get(k) == true_events.get(k) for k in EVENT_KEYS)
            if event_match:
                score += 2.0

            # 终局类型匹配
            if result["terminal"] == ep["observed_final_terminal"]:
                score += 2.0

            # 网格整体匹配 (动态位置)
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    pc = pred_grid[y][x]
                    tc = true_grid[y][x]
                    if pc != tc and (pc in 'ABOKDH^v<>' or tc in 'ABOKDH^v<>'):
                        if pc in '^v<>' or tc in '^v<>':
                            score -= 0.2
                        elif pc in 'DH' or tc in 'DH':
                            score -= 0.3
                        else:
                            score -= 0.5
                    elif pc == tc and pc in 'ABOK':
                        score += 0.2

        return score

    @classmethod
    def find_best_profile(cls, initial_grid: list[str],
                          episodes: list[dict]) -> dict:
        """找到最佳 profile"""
        candidates = cls.default_profiles()
        best_profile = candidates[0]
        best_score = float('-inf')

        for profile in candidates:
            score = cls.evaluate_profile(profile, initial_grid, episodes)
            if score > best_score:
                best_score = score
                best_profile = profile

        return best_profile


class RLCalibrator:
    """基于强化学习的校准器 —— 直接从 probe 数据学习转移规律"""

    def __init__(self):
        self.known_ice = set()      # 已知的冰面位置
        self.known_fields = {}      # 已知的场效应
        self.portal_map = {}        # 已知的传送门配对
        self.entity_dirs = {}       # 实体的可能方向

    def calibrate_from_probes(self, initial_grid: list[str],
                              episodes: list[dict]) -> dict:
        """从 probe 数据中学习转移规律"""
        # 解析静态地图
        static_map = self._parse_static(initial_grid)

        # 从每个 probe 学习
        dynamics_info = {
            "ice_behavior": "slide_until_blocked",
            "field_behavior": "push_one",
            "portal_map": self._infer_portals_from_grid(initial_grid),
            "ice_tiles": static_map["ice"],
            "field_tiles": static_map["fields"],
        }

        # 分析每个 probe 以推断动力学
        for ep in episodes:
            self._analyze_episode(initial_grid, ep, dynamics_info)

        return dynamics_info

    def _parse_static(self, grid: list[str]) -> dict:
        result = {"ice": set(), "fields": {}, "portals": set(),
                   "walls": set(), "doors": {}, "goals": set(),
                   "hazards": set()}
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                ch = grid[y][x]
                if ch == '#':
                    result["walls"].add((x, y))
                elif ch == 'I':
                    result["ice"].add((x, y))
                elif ch in '^v<>':
                    result["fields"][(x, y)] = ch
                elif ch == 'P':
                    result["portals"].add((x, y))
                elif ch == 'D':
                    result["doors"][(x, y)] = True
                elif ch == 'G':
                    result["goals"].add((x, y))
                elif ch == 'H':
                    result["hazards"].add((x, y))
        return result

    def _infer_portals_from_grid(self, grid: list[str]) -> dict:
        """根据位置推断传送门配对 (最近邻配对)"""
        portal_positions = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if grid[y][x] == 'P':
                    portal_positions.append((x, y))

        portal_map = {}
        # 按位置排序，成对配对
        portal_positions.sort()
        for i in range(0, len(portal_positions) - 1, 2):
            a, b = portal_positions[i], portal_positions[i + 1]
            portal_map[a] = b
            portal_map[b] = a
        return portal_map

    def _analyze_episode(self, initial_grid: list[str],
                         ep: dict, dynamics: dict):
        """分析单个 episode 来推断动力学"""
        init_grid = [list(row) for row in initial_grid]
        final_grid = [list(row) for row in ep["observed_final_full_grid"]]

        # 找出差异
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if init_grid[y][x] != final_grid[y][x]:
                    ic, fc = init_grid[y][x], final_grid[y][x]
                    # 记录状态转移
                    if ic == 'A' and fc == '.':
                        dynamics.setdefault("agent_left", []).append((x, y))
                    elif ic == '.' and fc == 'A':
                        dynamics.setdefault("agent_arrived", []).append((x, y))
                    elif ic == 'B' and fc == '.':
                        dynamics.setdefault("box_left", []).append((x, y))
                    elif ic == '.' and fc == 'B':
                        dynamics.setdefault("box_arrived", []).append((x, y))
                    elif ic == 'O' and fc == '.':
                        dynamics.setdefault("orb_left", []).append((x, y))
                    elif ic == '.' and fc == 'O':
                        dynamics.setdefault("orb_arrived", []).append((x, y))
                    elif ic == 'K' and fc == '.':
                        dynamics.setdefault("key_consumed", []).append((x, y))
                    elif ic == '^':
                        dynamics.setdefault("field_removed", []).append((x, y, '^'))


def predict_context(context: dict) -> list[dict]:
    """为单个 context 生成所有 query 的预测"""
    initial_grid = context["initial_full_grid"]
    episodes = context["context_episodes"]

    # 校准
    calibrator = RLCalibrator()
    dynamics = calibrator.calibrate_from_probes(initial_grid, episodes)

    # 搜索最优 profile
    best_profile = ProfileSearch.find_best_profile(initial_grid, episodes)

    results = []
    for query in context["queries"]:
        # 运行模拟器
        sim = Simulator(initial_grid, best_profile)
        sim_result = sim.run(query["future_actions"], query["query_horizon"])

        # 计算 timeline
        horizon = query["query_horizon"]
        timeline = sim.get_event_timeline(horizon)

        # 计算 event_order
        event_order = sim.get_event_order_padded(3)

        # 最终 terminal：如果所有动作执行后还是 active，改为 timeout
        terminal = sim_result["terminal"]
        if terminal == "active":
            terminal = "timeout"

        results.append({
            "id": query["query_id"],
            "final_grid": sim_result["final_grid"],
            "events": {k: sim_result["events"].get(k, False) for k in EVENT_KEYS},
            "event_timeline": {k: timeline.get(k, "never") for k in EVENT_KEYS},
            "event_order": event_order,
            "terminal": terminal,
        })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="test.jsonl")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--mode", default="predict",
                        choices=["predict", "validate", "analyze"])
    parser.add_argument("--context-id", default=None)
    args = parser.parse_args()

    results = []
    total_contexts = 0
    total_queries = 0

    for context in read_jsonl(args.tasks):
        total_contexts += 1

        if args.context_id and context["id"] != args.context_id:
            continue

        if args.mode == "analyze":
            print(f"\n=== Context: {context['id']} ===")
            print(f"Episodes: {len(context['context_episodes'])}")
            queries = predict_context(context)
            for q in queries:
                print(f"  Query: {q['id']}")
                print(f"    Terminal: {q['terminal']}")
                print(f"    Events: {json.dumps(q['events'])}")
            continue

        predictions = predict_context(context)
        results.extend(predictions)
        total_queries += len(predictions)

    if args.mode == "predict":
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Generated {len(results)} predictions from {total_contexts} contexts")
        print(f"Saved to {args.out}")

    if args.mode == "validate":
        # 在 train/valid 上验证
        correct = {k: 0 for k in ["events", "terminal", "entity_A", "entity_B",
                                    "entity_O", "entity_K", "timeline", "order"]}
        total = 0
        for context in read_jsonl(args.tasks):
            predictions = predict_context(context)
            for pred, query in zip(predictions, context["queries"]):
                total += 1
                label = query["label"]

                # 事件比较
                if all(pred["events"].get(k) == label["events"].get(k) for k in EVENT_KEYS):
                    correct["events"] += 1

                # 终局比较
                if pred["terminal"] == label["terminal"]:
                    correct["terminal"] += 1

                # 实体位置比较
                pred_entities = parse_entities_from_grid(pred["final_grid"])
                label_entities = label.get("entities", {})
                for entity in ["A", "B", "O", "K"]:
                    if pred_entities.get(entity) == label_entities.get(entity):
                        correct[f"entity_{entity}"] += 1

                # Timeline
                if pred["event_timeline"] == label.get("event_timeline", {}):
                    correct["timeline"] += 1

                # Order
                if pred["event_order"] == label.get("event_order", []):
                    correct["order"] += 1

        print(f"\n=== Validation Results ({total} queries) ===")
        for key, val in correct.items():
            print(f"  {key}: {val}/{total} = {100*val/total:.1f}%")


if __name__ == "__main__":
    main()
