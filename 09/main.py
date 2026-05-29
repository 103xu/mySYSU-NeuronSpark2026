"""
NS-2026-09 主预测脚本
用法：
  python main.py --tasks test.jsonl --out results.json
  python main.py --tasks valid.jsonl --out pred_valid.json --mode validate
"""

from __future__ import annotations

import json
import sys
import os
from collections import Counter

GRID_SIZE = 12
EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_entity(grid: list[str], char: str):
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] == char:
                return (x, y)
    return None


def extract_entities(grid: list[str]) -> dict[str, str]:
    result = {}
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            ch = grid[y][x]
            if ch in 'ABOK':
                result[ch] = f"{x},{y}"
    return result


DIR_VEC = {'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0)}
FIELD_VEC = {'^': (0, -1), 'v': (0, 1), '<': (-1, 0), '>': (1, 0)}


class FastSimulator:
    """轻量级快速模拟器 — 用于大规模预测"""

    def __init__(self, static_grid: list[str], initial_dir: str = 'D'):
        self.sgrid = static_grid  # 静态地图
        self.grid = [list(row) for row in static_grid]
        self.agent_dir = initial_dir
        self.step_count = 0
        self.events = {k: False for k in EVENT_KEYS}
        self.event_step = {k: -1 for k in EVENT_KEYS}
        self.event_order: list[str] = []
        self.keys_held = 0
        self.portal_cd = 0
        self.keys_collected_pos: set[tuple[int,int]] = set()
        self.fields_consumed: set[tuple[int,int]] = set()

    def _in_bounds(self, x, y):
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def _get_static(self, x, y):
        if not self._in_bounds(x, y):
            return '#'
        if (x, y) in self.keys_collected_pos:
            return '.'
        if (x, y) in self.fields_consumed:
            return '.'
        return self.sgrid[y][x]

    def _get(self, x, y):
        if not self._in_bounds(x, y):
            return '#'
        d = self.grid[y][x]
        if d != '.':
            return d
        return self._get_static(x, y)

    def _trigger(self, event: str):
        if not self.events[event]:
            self.events[event] = True
            self.event_step[event] = self.step_count
            self.event_order.append(event)

    def _process_orb_fields(self):
        """处理站在方向场上的 orb 移动"""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == 'O':
                    sc = self._get_static(x, y)
                    if sc in '^v<>':
                        fdx, fdy = FIELD_VEC[sc]
                        nx, ny = x + fdx, y + fdy
                        if self._in_bounds(nx, ny) and self._get(nx, ny) not in '#DBO':
                            self.grid[y][x] = '.'
                            self.grid[ny][nx] = 'O'

    def step_agent(self, action: str):
        self.step_count += 1

        # 每个步骤都处理全局 orb 在场上的移动
        self._process_orb_fields()

        if action == 'WAIT':
            self._check_pos()
            return

        dx, dy = DIR_VEC.get(action, (0, 0))
        if dx == 0 and dy == 0:
            return

        self.agent_dir = action

        # 找到 agent
        ax = ay = -1
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == 'A':
                    ax, ay = x, y
                    break
            if ax >= 0:
                break

        if ax < 0:
            return

        tx, ty = ax + dx, ay + dy

        if not self._in_bounds(tx, ty):
            self._trigger('collision')
            return

        target = self._get(tx, ty)

        if target == '#':
            self._trigger('collision')
            return

        if target == 'D':
            if self.keys_held > 0:
                self.keys_held -= 1
            else:
                self._trigger('collision')
                return

        if target == 'B':
            bnx, bny = tx + dx, ty + dy
            bc = self._get(bnx, bny) if self._in_bounds(bnx, bny) else '#'
            if bc in '#DB' or self.grid[bny][bnx] == 'B':
                self._trigger('collision')
                return
            self.grid[ty][tx] = '.'
            self.grid[bny][bnx] = 'B'
            if self._get_static(bnx, bny) == 'O':
                self._trigger('box_on_goal')

        # 移动 agent
        self.grid[ay][ax] = '.'
        self.grid[ty][tx] = 'A'

        # 传送门
        if self._get_static(tx, ty) == 'P' and self.portal_cd <= 0:
            dest = self._find_portal_dest(tx, ty)
            if dest and self._get(*dest) not in '#BD':
                self.grid[ty][tx] = '.'
                self.grid[dest[1]][dest[0]] = 'A'
                self._trigger('portal_used')
                self.portal_cd = 3
                tx, ty = dest

        if self.portal_cd > 0:
            self.portal_cd -= 1

        # 冰面滑动：仅滑动 1 步
        if self._get_static(tx, ty) == 'I':
            nx, ny = tx + dx, ty + dy
            if self._in_bounds(nx, ny):
                target2 = self._get(nx, ny)
                if target2 not in '#DB' and not (target2 == 'B' and
                    not self._in_bounds(nx + dx, ny + dy) or self._get(nx + dx, ny + dy) in '#DB'):
                    if target2 == 'B':
                        bnx, bny = nx + dx, ny + dy
                        if self._in_bounds(bnx, bny) and self._get(bnx, bny) not in '#DB':
                            self.grid[ny][nx] = '.'
                            self.grid[bny][bnx] = 'B'
                            if self._get_static(bnx, bny) == 'O':
                                self._trigger('box_on_goal')
                        else:
                            self._check_pos()
                            return
                    self.grid[ty][tx] = '.'
                    self.grid[ny][nx] = 'A'
                    tx, ty = nx, ny

                    # 滑冰后传送门
                    if self._get_static(tx, ty) == 'P' and self.portal_cd <= 0:
                        dest = self._find_portal_dest(tx, ty)
                        if dest and self._get(*dest) not in '#BD':
                            self.grid[ty][tx] = '.'
                            self.grid[dest[1]][dest[0]] = 'A'
                            self._trigger('portal_used')
                            self.portal_cd = 3
                            tx, ty = dest

        self._check_pos()

    def _find_portal_dest(self, x: int, y: int):
        """找到最近的另一个传送门"""
        best = None
        best_dist = 999
        for py in range(GRID_SIZE):
            for px in range(GRID_SIZE):
                if self._get_static(px, py) == 'P' and (px, py) != (x, y):
                    d = abs(px - x) + abs(py - y)
                    if d < best_dist:
                        best_dist = d
                        best = (px, py)
        return best

    def _check_pos(self):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == 'A':
                    sc = self._get_static(x, y)
                    if sc == 'G':
                        self._trigger('goal_reached')
                    if sc == 'H':
                        self._trigger('hazard')
                    if sc == 'K' and (x, y) not in self.keys_collected_pos:
                        self._trigger('key_collected')
                        self.keys_held += 1
                        self.keys_collected_pos.add((x, y))
                    return

    def run(self, actions: list[str], horizon: int = 0) -> dict:
        limit = horizon if horizon > 0 else len(actions)
        steps = min(limit, len(actions))

        for i in range(steps):
            self.step_agent(actions[i])
            if self.events.get('goal_reached') or self.events.get('hazard'):
                break

        final_grid = ["".join(row) for row in self.grid]

        # Terminal
        if self.events.get('goal_reached'):
            terminal = 'goal'
        elif self.events.get('hazard'):
            terminal = 'hazard'
        elif self.events.get('collision'):
            terminal = 'blocked'
        elif self.step_count >= steps:
            terminal = 'timeout'
        else:
            terminal = 'timeout'

        # Timeline
        eff_horizon = max(horizon, steps, 1)
        timeline = {}
        for key in EVENT_KEYS:
            step = self.event_step.get(key, -1)
            if step < 0:
                timeline[key] = "never"
            else:
                ratio = step / eff_horizon
                if ratio < 0.33:
                    timeline[key] = "early"
                elif ratio < 0.67:
                    timeline[key] = "mid"
                else:
                    timeline[key] = "late"

        order = list(self.event_order)
        while len(order) < 3:
            order.append("none")

        return {
            "final_grid": final_grid,
            "events": dict(self.events),
            "event_timeline": timeline,
            "event_order": order[:3],
            "terminal": terminal,
        }


class ProbeLearner:
    """从 probe episodes 学习上下文特征"""

    @staticmethod
    def learn(context: dict) -> dict:
        episodes = context["context_episodes"]
        info = {}

        # 探针终局统计
        terminals = [ep["observed_final_terminal"] for ep in episodes]
        info["probe_terminals"] = terminals
        info["common_terminal"] = Counter(terminals).most_common(1)[0][0]

        # 探针事件统计
        event_counts = {k: 0 for k in EVENT_KEYS}
        for ep in episodes:
            for k in EVENT_KEYS:
                if ep["observed_final_events"].get(k):
                    event_counts[k] += 1
        info["event_freq"] = {k: v / len(episodes) for k, v in event_counts.items()}

        # 探针事件顺序
        probe_orders = [ep.get("observed_final_event_order", []) for ep in episodes]
        info["probe_orders"] = probe_orders

        # 探针时间线
        probe_timelines = [ep.get("observed_final_event_timeline", {}) for ep in episodes]
        info["probe_timelines"] = probe_timelines

        # 初始实体位置
        init_entities = context.get("initial_entities", {})
        info["init_A"] = init_entities.get("A", "")
        info["init_B"] = init_entities.get("B", "")
        info["init_O"] = init_entities.get("O", "")
        info["init_K"] = init_entities.get("K", "")

        # 地图特征
        full = "".join(context["initial_full_grid"])
        info["n_ice"] = full.count('I')
        info["n_portal"] = full.count('P')
        info["n_field"] = sum(1 for c in full if c in '^v<>')
        info["n_hazard"] = full.count('H')
        info["n_goal"] = full.count('G')
        info["n_door"] = full.count('D')

        # 探针中 agent 的位移模式
        init_a = ProbeLearner._parse_pos(init_entities.get("A"))
        if init_a:
            displacements = []
            for ep in episodes:
                final_a = find_entity(ep["observed_final_full_grid"], 'A')
                if final_a:
                    displacements.append((final_a[0] - init_a[0], final_a[1] - init_a[1]))
            info["probe_displacements"] = displacements

        return info

    @staticmethod
    def _parse_pos(s: str | None) -> tuple[int, int] | None:
        if not s:
            return None
        parts = s.split(",")
        return (int(parts[0]), int(parts[1]))


def predict_context(context: dict) -> list[dict]:
    """为一个 context 的所有 query 生成预测"""
    initial_grid = context["initial_full_grid"]
    info = ProbeLearner.learn(context)

    results = []
    for query in context["queries"]:
        # 获取初始方向
        init_dir = query.get("initial_observation", {}).get("sensor", {}).get("agent_dir", "D")

        # 运行模拟器
        sim = FastSimulator(initial_grid, init_dir)
        result = sim.run(query["future_actions"], query["query_horizon"])

        # === 后处理校正 ===

        # 终端校正：根据探针数据调整
        terminal = result["terminal"]

        # 事件校正
        events = dict(result["events"])

        # 如果绝大多数探针都有 collision，而模拟器没有检测到，添加它
        if info["event_freq"].get("collision", 0) >= 0.8 and not events.get("collision"):
            events["collision"] = True

        # 如果探针从未触发某个事件，保持模拟器的判断
        for k in EVENT_KEYS:
            if info["event_freq"].get(k, 0) == 0 and events.get(k):
                events[k] = False  # 探针从未触发，可能模拟器错了

        # Timeline: 优先使用模拟器，回退到探针统计
        timeline = {}
        for k in EVENT_KEYS:
            if events.get(k):
                t = result["event_timeline"].get(k, "never")
                if t == "never":
                    # 从探针推断
                    probe_tls = [
                        pt.get(k, "never") for pt in info["probe_timelines"]
                        if pt.get(k, "never") != "never"
                    ]
                    if probe_tls:
                        t = Counter(probe_tls).most_common(1)[0][0]
                    else:
                        t = "early"
                timeline[k] = t
            else:
                timeline[k] = "never"

        # Event order
        order = list(result["event_order"])
        # Filter to only events that are actually True
        order = [e for e in order if events.get(e, False)]
        # Add additional true events from probe order if space
        if len(order) < 3:
            for po in info["probe_orders"]:
                for e in po:
                    if e != "none" and events.get(e) and e not in order and len(order) < 3:
                        order.append(e)
        while len(order) < 3:
            order.append("none")

        # Terminal: 如果模拟器给出 timeout 但探针倾向不同
        if terminal == "timeout" and info["event_freq"].get("collision", 0) >= 0.8:
            if info["event_freq"].get("goal_reached", 0) > 0:
                terminal = "goal"
            elif info["event_freq"].get("hazard", 0) > 0:
                terminal = "hazard"
            else:
                terminal = "blocked"

        results.append({
            "id": query["query_id"],
            "final_grid": result["final_grid"],
            "events": events,
            "event_timeline": timeline,
            "event_order": order[:3],
            "terminal": terminal,
        })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NS-2026-09 预测")
    parser.add_argument("--tasks", default="test.jsonl", help="输入 JSONL 文件")
    parser.add_argument("--out", default="results.json", help="输出结果文件")
    parser.add_argument("--mode", default="predict",
                        choices=["predict", "validate"])
    args = parser.parse_args()

    if not os.path.exists(args.tasks):
        print(f"Error: {args.tasks} not found")
        sys.exit(1)

    all_results = []
    ctx_count = 0

    for context in read_jsonl(args.tasks):
        ctx_count += 1
        predictions = predict_context(context)
        all_results.extend(predictions)

        if ctx_count % 50 == 0:
            print(f"  Processed {ctx_count} contexts...")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_results)} predictions to {args.out}")

    if args.mode == "validate":
        _validate(args.tasks, all_results)


def _validate(tasks_path: str, predictions: list[dict]):
    """在训练/验证集上评估"""
    pred_by_id = {p["id"]: p for p in predictions}
    total = 0
    correct = {
        "events": 0, "terminal": 0, "timeline": 0,
        "order": 0, "entity_A": 0, "entity_B": 0,
        "entity_O": 0, "entity_K": 0,
    }

    for context in read_jsonl(tasks_path):
        for query in context["queries"]:
            qid = query["query_id"]
            pred = pred_by_id.get(qid)
            if pred is None:
                continue
            total += 1
            label = query["label"]

            if all(pred["events"].get(k) == label["events"].get(k) for k in EVENT_KEYS):
                correct["events"] += 1

            if pred["terminal"] == label["terminal"]:
                correct["terminal"] += 1

            if pred["event_timeline"] == label.get("event_timeline", {}):
                correct["timeline"] += 1

            if pred["event_order"] == label.get("event_order", []):
                correct["order"] += 1

            pred_e = extract_entities(pred["final_grid"])
            label_e = label.get("entities", {})
            for ent in ["A", "B", "O", "K"]:
                if pred_e.get(ent) == label_e.get(ent):
                    correct[f"entity_{ent}"] += 1

    print(f"\n=== Validation ({total} queries) ===")
    for k, v in correct.items():
        print(f"  {k}: {v}/{total} = {100*v/max(total,1):.1f}%")


if __name__ == "__main__":
    main()
