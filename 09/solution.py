"""
NS-2026-09 完整解决方案
结合物理模拟器 + 统计推断 + 探针校准
"""

from __future__ import annotations

import json
import sys
import os
from typing import Optional
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GRID_SIZE = 12
EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_entity(grid: list[str], char: str) -> tuple[int, int] | None:
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] == char:
                return (x, y)
    return None


def find_all_entities(grid: list[str], chars: str) -> dict:
    result = {}
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            ch = grid[y][x]
            if ch in chars:
                if ch not in result:
                    result[ch] = []
                result[ch].append((x, y))
    return result


def clone_grid(grid: list[str]) -> list[list[str]]:
    return [list(row) for row in grid]


def grid_to_strings(grid: list[list[str]]) -> list[str]:
    return ["".join(row) for row in grid]


DIR_VEC = {'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0)}
FIELD_DIR = {'^': (0, -1), 'v': (0, 1), '<': (-1, 0), '>': (1, 0)}


class WorldSimulator:
    """改进的物理模拟器 —— 使用更完整的物理规则"""

    def __init__(self, grid: list[str], params: dict | None = None):
        self.static_grid = grid
        self.p = params or {}
        self.reset()

    def reset(self):
        self.grid = clone_grid(self.static_grid)
        self.agent_dir = 'D'
        self.keys_held = 0
        self.portal_cool_down = 0
        self.step_idx = 0
        self.events = {k: False for k in EVENT_KEYS}
        self.event_step = {k: -1 for k in EVENT_KEYS}
        self.event_order = []

    def get_agent_pos(self) -> tuple[int, int] | None:
        return find_entity(grid_to_strings(self.grid), 'A')

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def is_wall(self, x: int, y: int) -> bool:
        return self.grid[y][x] == '#'

    def is_door(self, x: int, y: int) -> bool:
        return self.grid[y][x] == 'D'

    def is_blocked(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return True
        ch = self.grid[y][x]
        return ch in '#D'  # D 默认锁住

    def is_pushable(self, x: int, y: int) -> bool:
        return self.grid[y][x] == 'B'

    def push_box(self, bx: int, by: int, dx: int, dy: int) -> bool:
        """尝试推箱子，返回是否成功"""
        nx, ny = bx + dx, by + dy
        if not self.in_bounds(nx, ny):
            return False
        target = self.grid[ny][nx]
        if target in '#DB':  # 阻挡物
            return False
        # 移动箱子
        self.grid[by][bx] = '.'
        self.grid[ny][nx] = 'B'
        # 检查 box_on_goal
        if self.static_grid[ny][nx] == 'O' or self.grid[ny][nx] == 'B' and self._has_orb_at(nx, ny):
            self._trigger('box_on_goal')
        return True

    def _has_orb_at(self, x: int, y: int) -> bool:
        # O 可能在过程中移动，检查当前位置
        for yy in range(GRID_SIZE):
            for xx in range(GRID_SIZE):
                if self.grid[yy][xx] == 'O' and xx == x and yy == y:
                    return True
        return False

    def _trigger(self, event: str):
        if not self.events.get(event, False):
            self.events[event] = True
            self.event_step[event] = self.step_idx
            self.event_order.append(event)

    def step(self, action: str):
        """执行一步动作，处理完整物理"""
        self.step_idx += 1

        if action == 'WAIT':
            self._apply_field_to_agent()
            self._check_position()
            return

        dx, dy = DIR_VEC.get(action, (0, 0))
        if dx == 0 and dy == 0:
            self._check_position()
            return

        self.agent_dir = action

        agent_pos = self.get_agent_pos()
        if agent_pos is None:
            return
        ax, ay = agent_pos

        # 目标位置
        tx, ty = ax + dx, ay + dy

        # 边界/墙壁检查
        if not self.in_bounds(tx, ty) or self.grid[ty][tx] == '#':
            self._trigger('collision')
            self._check_position()
            return

        target = self.grid[ty][tx]

        # 门检查
        if target == 'D':
            if self.keys_held > 0:
                self.keys_held -= 1
                self.grid[ty][tx] = '.'
            else:
                self._trigger('collision')
                self._check_position()
                return

        # 箱子推动
        if target == 'B':
            if not self.push_box(tx, ty, dx, dy):
                self._trigger('collision')
                self._check_position()
                return

        # 移动 agent
        self.grid[ay][ax] = '.'
        self.grid[ty][tx] = 'A'

        # 传送门
        if self.static_grid[ty][tx] == 'P' and self.portal_cool_down <= 0:
            dest = self._find_portal_dest(tx, ty)
            if dest:
                dt_x, dt_y = dest
                if self.grid[dt_y][dt_x] not in '#DB':
                    self.grid[ty][tx] = '.'
                    self.grid[dt_y][dt_x] = 'A'
                    tx, ty = dt_x, dt_y
                    self._trigger('portal_used')
                    self.portal_cool_down = 3
        if self.portal_cool_down > 0:
            self.portal_cool_down -= 1

        # 冰面滑动
        if self.static_grid[ty][tx] == 'I':
            self._apply_ice_slide(tx, ty, dx, dy)

        # 方向场效应
        self._apply_field_to_agent()

        # 位置交互
        self._check_position()

    def _find_portal_dest(self, x: int, y: int) -> tuple[int, int] | None:
        """找到传送门目的地（最近邻配对）"""
        all_p = []
        for py in range(GRID_SIZE):
            for px in range(GRID_SIZE):
                if self.static_grid[py][px] == 'P' and (px, py) != (x, y):
                    all_p.append((px, py))
        if all_p:
            # 返回最近的一个
            return min(all_p, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
        return None

    def _apply_ice_slide(self, x: int, y: int, dx: int, dy: int):
        """冰面滑动：沿移动方向继续滑动直到非冰面"""
        steps = 0
        while steps < 10 and self.static_grid[y][x] == 'I':
            nx, ny = x + dx, y + dy
            if not self.in_bounds(nx, ny):
                break
            if self.grid[ny][nx] == '#':
                break
            if self.grid[ny][nx] == 'D':
                break
            if self.grid[ny][nx] == 'B':
                if not self.push_box(nx, ny, dx, dy):
                    break
            # 滑动
            self.grid[y][x] = '.'
            self.grid[ny][nx] = 'A'
            x, y = nx, ny
            steps += 1

            # 检查滑动过程中的传送门
            if self.static_grid[y][x] == 'P' and self.portal_cool_down <= 0:
                dest = self._find_portal_dest(x, y)
                if dest:
                    dt_x, dt_y = dest
                    if self.grid[dt_y][dt_x] not in '#DB':
                        self.grid[y][x] = '.'
                        self.grid[dt_y][dt_x] = 'A'
                        x, y = dt_x, dt_y
                        self._trigger('portal_used')
                        self.portal_cool_down = 3
                        break

            self._check_position()
            if self.events.get('hazard') or self.events.get('goal_reached'):
                break

    def _apply_field_to_agent(self):
        """对 agent 当前位置应用方向场"""
        agent_pos = self.get_agent_pos()
        if agent_pos is None:
            return
        x, y = agent_pos
        static_ch = self.static_grid[y][x]
        if static_ch not in '^v<>':
            return
        dx, dy = FIELD_DIR[static_ch]
        nx, ny = x + dx, y + dy
        if not self.in_bounds(nx, ny):
            return
        if self.grid[ny][nx] in '#D':
            return
        if self.grid[ny][nx] == 'B':
            if not self.push_box(nx, ny, dx, dy):
                return
        self.grid[y][x] = '.'
        self.grid[ny][nx] = 'A'
        self._check_position()

    def _check_position(self):
        """检查 agent 当前位置的各种交互"""
        agent_pos = self.get_agent_pos()
        if agent_pos is None:
            return
        x, y = agent_pos

        # 目标
        if self.static_grid[y][x] == 'G':
            self._trigger('goal_reached')

        # 危险
        if self.static_grid[y][x] == 'H':
            self._trigger('hazard')

        # 钥匙
        if self.static_grid[y][x] == 'K':
            self._trigger('key_collected')
            self.keys_held += 1
            # 从静态地图中移除钥匙（防止重复收集）
            # 但 static_grid 不可变，用 keys_held 追踪

        # Box on goal check
        agents = find_entity(grid_to_strings(self.grid), 'A')
        boxes = find_all_entities(grid_to_strings(self.grid), 'B')
        orbs = find_all_entities(grid_to_strings(self.grid), 'O')
        for bpos in boxes.get('B', []):
            for opos in orbs.get('O', []):
                if bpos == opos:
                    self._trigger('box_on_goal')

    def run(self, actions: list[str], horizon: int | None = None) -> dict:
        self.reset()
        limit = horizon if horizon is not None else len(actions)
        max_steps = min(limit, len(actions))

        for i in range(max_steps):
            self.step_idx = i
            self.step(actions[i])
            if self.events.get('goal_reached') or self.events.get('hazard'):
                break

        return self._result(limit)

    def _result(self, horizon: int) -> dict:
        final_grid = grid_to_strings(self.grid)

        # Terminal
        if self.events.get('goal_reached'):
            terminal = 'goal'
        elif self.events.get('hazard'):
            terminal = 'hazard'
        elif self.step_idx >= len(self.events) and not self.events.get('goal_reached'):
            # 检查是否 blocked
            if self.events.get('collision'):
                terminal = 'blocked'
            elif self.step_idx >= horizon:
                terminal = 'timeout'
            else:
                terminal = 'timeout'
        elif self.events.get('collision'):
            terminal = 'blocked'
        else:
            terminal = 'timeout'

        # Event timeline
        timeline = {}
        for key in EVENT_KEYS:
            step = self.event_step.get(key, -1)
            if step < 0:
                timeline[key] = "never"
            else:
                ratio = step / max(horizon, 1)
                if ratio < 0.33:
                    timeline[key] = "early"
                elif ratio < 0.67:
                    timeline[key] = "mid"
                else:
                    timeline[key] = "late"

        # Event order
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


class ContextAnalyzer:
    """从 context 的 probe 数据中提取特征并推断隐藏参数"""

    def __init__(self):
        pass

    def analyze(self, context: dict) -> dict:
        """分析 context 并返回推断的信息"""
        initial_grid = context["initial_full_grid"]
        episodes = context["context_episodes"]

        info = {
            "initial_agent_pos": self._parse_pos(context["initial_entities"].get("A", "0,0")),
            "initial_box_pos": self._parse_pos(context["initial_entities"].get("B", None)),
            "initial_orb_pos": self._parse_pos(context["initial_entities"].get("O", None)),
            "initial_key_pos": self._parse_pos(context["initial_entities"].get("K", None)),
        }

        # 静态分析
        grid_str = "".join(initial_grid)
        info["num_ice"] = grid_str.count('I')
        info["num_portal"] = grid_str.count('P')
        info["num_door"] = grid_str.count('D')
        info["num_key"] = grid_str.count('K')
        info["num_goal"] = grid_str.count('G')
        info["num_hazard"] = grid_str.count('H')
        info["num_field"] = sum(1 for c in grid_str if c in '^v<>')
        info["num_box"] = grid_str.count('B')
        info["num_orb"] = grid_str.count('O')

        # 计算 agent 到各元素的距离
        if info["initial_agent_pos"]:
            ax, ay = info["initial_agent_pos"]
            info["dist_to_goal"] = self._min_dist_to(initial_grid, ax, ay, 'G')
            info["dist_to_hazard"] = self._min_dist_to(initial_grid, ax, ay, 'H')
            info["dist_to_key"] = self._min_dist_to(initial_grid, ax, ay, 'K')
            info["dist_to_door"] = self._min_dist_to(initial_grid, ax, ay, 'D')
            info["dist_to_ice"] = self._min_dist_to(initial_grid, ax, ay, 'I')
            info["dist_to_portal"] = self._min_dist_to(initial_grid, ax, ay, 'P')

        # 从 probe episode 中学习
        probe_terminals = []
        probe_events_list = []
        for ep in episodes:
            probe_terminals.append(ep["observed_final_terminal"])
            probe_events_list.append(ep["observed_final_events"])

            # 分析探针中 agent 的最终位置
            final_a = find_entity(ep["observed_final_full_grid"], 'A')
            if final_a and info["initial_agent_pos"]:
                dx = final_a[0] - info["initial_agent_pos"][0]
                dy = final_a[1] - info["initial_agent_pos"][1]
                info.setdefault("probe_displacements", []).append((dx, dy))

        info["probe_terminals"] = probe_terminals
        info["probe_events"] = probe_events_list

        # 推断可能的 terminal
        terminal_counter = Counter(probe_terminals)
        info["most_common_terminal"] = terminal_counter.most_common(1)[0][0]

        # 推断事件概率
        event_prob = {k: 0.0 for k in EVENT_KEYS}
        for ev in probe_events_list:
            for k in EVENT_KEYS:
                if ev.get(k):
                    event_prob[k] += 1
        for k in EVENT_KEYS:
            event_prob[k] /= max(len(episodes), 1)
        info["probe_event_probs"] = event_prob

        return info

    def _parse_pos(self, pos_str: str | None) -> tuple[int, int] | None:
        if pos_str is None:
            return None
        parts = pos_str.split(",")
        return (int(parts[0]), int(parts[1]))

    def _min_dist_to(self, grid: list[str], ax: int, ay: int, char: str) -> float:
        min_dist = float('inf')
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if grid[y][x] == char:
                    dist = abs(x - ax) + abs(y - ay)
                    if dist < min_dist:
                        min_dist = dist
        return min_dist if min_dist != float('inf') else 999


def predict_with_simulator(context: dict) -> list[dict]:
    """使用物理模拟器进行预测"""
    initial_grid = context["initial_full_grid"]
    episodes = context["context_episodes"]
    analyzer = ContextAnalyzer()
    info = analyzer.analyze(context)

    # 使用默认参数运行模拟器
    sim = WorldSimulator(initial_grid)
    results = []

    for query in context["queries"]:
        result = sim.run(query["future_actions"], query["query_horizon"])

        # 后处理：根据 probe 数据调整预测
        terminal = result["terminal"]

        # 如果模拟器给出 timeout 但 probe 都是 blocked，改为 blocked
        if terminal == "timeout":
            if info["most_common_terminal"] != "timeout":
                terminal = info["most_common_terminal"]

        # 如果模拟器没有 collision 但大多数 probe 都有
        if not result["events"].get("collision") and info["probe_event_probs"].get("collision", 0) > 0.8:
            result["events"]["collision"] = True
            result["event_timeline"]["collision"] = "late"
            if "collision" not in result["event_order"] and len([x for x in result["event_order"] if x != "none"]) < 3:
                result["event_order"] = [x for x in result["event_order"] if x != "none"]
                result["event_order"].append("collision")
                while len(result["event_order"]) < 3:
                    result["event_order"].append("none")

        results.append({
            "id": query["query_id"],
            "final_grid": result["final_grid"],
            "events": result["events"],
            "event_timeline": result["event_timeline"],
            "event_order": result["event_order"][:3],
            "terminal": terminal,
        })

    return results


def predict_statistical_baseline(context: dict) -> list[dict]:
    """改进的统计基线预测器"""
    initial_grid = context["initial_full_grid"]
    analyzer = ContextAnalyzer()
    info = analyzer.analyze(context)

    results = []
    for query in context["queries"]:
        # 从 initial_observation 的 local_view 重建基本网格
        local_view = query["initial_observation"].get("local_view", [])
        if local_view:
            # 使用基线方法投影 local_view
            view_grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            for i in range(GRID_SIZE):
                view_grid[0][i] = "#"
                view_grid[GRID_SIZE - 1][i] = "#"
                view_grid[i][0] = "#"
                view_grid[i][GRID_SIZE - 1] = "#"
            cx = cy = GRID_SIZE // 2
            for vy, line in enumerate(local_view):
                for vx, ch in enumerate(line):
                    if ch in {"A", "B", "O", "K", "D", "G", "H", "I", "P", "^", "v", "<", ">"}:
                        x = cx + vx - 2
                        y = cy + vy - 2
                        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                            view_grid[y][x] = ch
            final_grid = ["".join(row) for row in view_grid]
        else:
            final_grid = initial_grid

        # 使用 probe 数据推断事件
        probe_events = info["probe_events"]
        events = {k: False for k in EVENT_KEYS}
        for ev in probe_events:
            for k in EVENT_KEYS:
                if ev.get(k):
                    events[k] = True

        # 也运行模拟器来获取更好的网格
        sim = WorldSimulator(initial_grid)
        sim_result = sim.run(query["future_actions"], query["query_horizon"])
        # 使用模拟器的网格（比 local_view 更准确）
        final_grid = sim_result["final_grid"]

        # 合并事件：如果任何 probe 触发了某事件，保留它
        for k in EVENT_KEYS:
            if sim_result["events"].get(k):
                events[k] = True

        # Timeline
        timeline = {}
        horizon = query["query_horizon"]
        for k in EVENT_KEYS:
            if events[k]:
                # 根据 probe 推断时间
                probe_timelines = [
                    ep.get("observed_final_event_timeline", {}).get(k, "never")
                    for ep in context["context_episodes"]
                ]
                non_never = [t for t in probe_timelines if t != "never"]
                if non_never:
                    timeline[k] = Counter(non_never).most_common(1)[0][0]
                else:
                    timeline[k] = "early"
            else:
                timeline[k] = "never"

        # Event order: 按照 probe 中常见的事件顺序
        probe_orders = [
            ep.get("observed_final_event_order", [])
            for ep in context["context_episodes"]
        ]
        # 取第一个 probe 的顺序（它们应该相似）
        event_order = list(probe_orders[0]) if probe_orders else ["none", "none", "none"]
        while len(event_order) < 3:
            event_order.append("none")

        # Terminal
        terminal = info["most_common_terminal"]
        # 使用模拟器结果（如果更具体）
        if sim_result["terminal"] != "timeout":
            terminal = sim_result["terminal"]

        results.append({
            "id": query["query_id"],
            "final_grid": final_grid,
            "events": events,
            "event_timeline": timeline,
            "event_order": event_order[:3],
            "terminal": terminal,
        })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="test.jsonl")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--mode", default="predict",
                        choices=["predict", "validate", "stats"])
    args = parser.parse_args()

    results = []
    for context in read_jsonl(args.tasks):
        predictions = predict_statistical_baseline(context)
        results.extend(predictions)

    if args.mode in ("predict", "stats"):
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(results)} predictions to {args.out}")

    if args.mode == "validate":
        correct_events = 0
        correct_terminal = 0
        correct_grid = 0
        correct_entity = 0
        total = 0
        idx = 0
        for context in read_jsonl(args.tasks):
            predictions = predict_statistical_baseline(context)
            for pred, query in zip(predictions, context["queries"]):
                total += 1
                label = query["label"]

                if all(pred["events"].get(k) == label["events"].get(k) for k in EVENT_KEYS):
                    correct_events += 1
                if pred["terminal"] == label["terminal"]:
                    correct_terminal += 1

                # Entity comparison
                pred_e = {}
                for y in range(GRID_SIZE):
                    for x in range(GRID_SIZE):
                        ch = pred["final_grid"][y][x]
                        if ch in 'ABOK':
                            pred_e[ch] = f"{x},{y}"
                label_e = label.get("entities", {})
                if pred_e == label_e:
                    correct_entity += 1

        print(f"Events: {correct_events}/{total} = {100*correct_events/total:.1f}%")
        print(f"Terminal: {correct_terminal}/{total} = {100*correct_terminal/total:.1f}%")
        print(f"Entity: {correct_entity}/{total} = {100*correct_entity/total:.1f}%")


if __name__ == "__main__":
    main()
