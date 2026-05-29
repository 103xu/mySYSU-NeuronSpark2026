"""
网格世界物理模拟器 - NS-2026-09 观测之环
支持所有已知物理机制，包括可配置的隐藏参数。
"""

from __future__ import annotations

import json
from typing import Optional

GRID_SIZE = 12
EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]


class Simulator:
    """参数化的网格世界模拟器"""

    def __init__(self, grid: list[str], profile: dict | None = None):
        self.initial_grid = [list(row) for row in grid]
        self.grid = [list(row) for row in grid]
        self.profile = profile or self._default_profile()
        self._parse_entities()

        # 状态
        self.agent_dir = self._infer_initial_dir()
        self.keys_collected = set()
        self.doors_opened = set()
        self.box_on_goal_triggered = False
        self.key_collected_this_step = False
        self.portal_used_this_step = False
        self.terminal = "active"
        self.step_count = 0
        self.total_steps = 0

        # 事件追踪
        self.events: dict[str, bool] = {k: False for k in EVENT_KEYS}
        self.event_first_step: dict[str, int | None] = {k: None for k in EVENT_KEYS}
        self.event_order: list[str] = []
        self.first_seen: set[str] = set()

        # 用于 portal 配对的缓存
        self._portal_graph: dict[tuple[int,int], tuple[int,int]] = {}
        self._infer_portal_pairs()

    @staticmethod
    def _default_profile() -> dict:
        return {
            "ice_mode": "slide_until_blocked",  # slide_one, slide_until_blocked, slide_none
            "field_mode": "push_one",  # push_one, push_continuous, push_none
            "field_affects_boxes": True,
            "field_affects_orbs": True,
            "field_priority": "before_move",  # before_move, after_move
            "portal_mode": "paired",  # paired, random
            "portal_bidirectional": True,
            "door_requires_key": True,
            "door_consumes_key": True,
            "box_pushable": True,
            "key_pickup_mode": "step_on",  # step_on, adjacent
            "goal_mode": "step_on",  # step_on, adjacent
            "hazard_mode": "step_on",  # step_on, adjacent
            "ice_direction": "momentum",  # momentum (continue direction), reverse, random
            "ice_max_slide": 10,
            "agent_faces_move_direction": True,
            "entity_update_order": "agent_first",  # agent_first, fields_first
        }

    def _parse_entities(self):
        """解析初始实体位置"""
        self.agent_pos = None
        self.boxes: dict[tuple[int,int], str] = {}  # pos -> type (B)
        self.keys: dict[tuple[int,int], str] = {}
        self.orbs: dict[tuple[int,int], str] = {}
        self.goals: set[tuple[int,int]] = set()
        self.hazards: set[tuple[int,int]] = set()
        self.portals: set[tuple[int,int]] = set()
        self.doors: dict[tuple[int,int], bool] = {}  # pos -> locked
        self.ice: set[tuple[int,int]] = set()
        self.fields: dict[tuple[int,int], str] = {}  # pos -> direction char

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                ch = self.grid[y][x]
                if ch == 'A':
                    self.agent_pos = (x, y)
                elif ch == 'B':
                    self.boxes[(x, y)] = 'B'
                elif ch == 'K':
                    self.keys[(x, y)] = 'K'
                elif ch == 'O':
                    self.orbs[(x, y)] = 'O'
                elif ch == 'G':
                    self.goals.add((x, y))
                elif ch == 'H':
                    self.hazards.add((x, y))
                elif ch == 'P':
                    self.portals.add((x, y))
                elif ch == 'D':
                    self.doors[(x, y)] = True  # locked by default
                elif ch == 'I':
                    self.ice.add((x, y))
                elif ch in '^v<>':
                    self.fields[(x, y)] = ch

    def _find_agent_in_grid(self) -> tuple[int, int] | None:
        """从网格中找到 agent 位置"""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == 'A':
                    return (x, y)
        return None

    def _infer_initial_dir(self) -> str:
        """从周围环境推断 agent 初始方向，默认 D"""
        if self.agent_pos is None:
            return 'D'
        x, y = self.agent_pos
        # 查看四个方向是否有特殊标记
        for dx, dy, d in [(0, -1, 'U'), (0, 1, 'D'), (-1, 0, 'L'), (1, 0, 'R')]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                ch = self.grid[ny][nx]
                # 如果对面有 field 方向指向 agent，则 agent 面朝相反方向
                if d == 'U' and ch == 'v':
                    return 'U'
                if d == 'D' and ch == '^':
                    return 'D'
                if d == 'L' and ch == '>':
                    return 'L'
                if d == 'R' and ch == '<':
                    return 'R'
        return 'D'

    def _infer_portal_pairs(self):
        """推断传送门配对 —— 默认按位置配对"""
        portal_list = sorted(self.portals)
        n = len(portal_list)
        for i in range(0, n - 1, 2):
            a, b = portal_list[i], portal_list[i + 1]
            self._portal_graph[a] = b
            if self.profile.get("portal_bidirectional", True):
                self._portal_graph[b] = a

    def _is_blocked(self, x: int, y: int) -> bool:
        """检查位置是否被阻挡"""
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return True
        ch = self.grid[y][x]
        if ch == '#':
            return True
        if ch == 'D' and self.doors.get((x, y), False):
            return True
        if ch == 'B' and not self.profile.get("box_pushable", True):
            return True
        return False

    def _is_occupied(self, x: int, y: int) -> bool:
        """检查位置是否有实体占据"""
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return True
        return self.grid[y][x] in '#BD'

    def _move_dir(self, direction: str) -> tuple[int, int]:
        return {'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0)}.get(direction, (0, 0))

    def _try_push_box(self, bx: int, by: int, dx: int, dy: int) -> bool:
        """尝试推动箱子"""
        nx, ny = bx + dx, by + dy
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
            return False
        if self.grid[ny][nx] not in '.IOGKPH^v<>' and not (self.grid[ny][nx] == 'D' and not self.doors.get((nx, ny), True)):
            return False
        # 箱子不能推到另一个箱子或其他实体上
        if self.grid[ny][nx] in 'B':
            return False
        # 移动箱子
        old_pos = (bx, by)
        new_pos = (nx, ny)
        old_char = self.grid[by][bx]
        self.grid[by][bx] = '.'
        self.grid[ny][nx] = old_char
        del self.boxes[old_pos]
        self.boxes[new_pos] = old_char

        # 检查 box_on_goal
        if new_pos in self.orbs:
            self._trigger_event("box_on_goal")

        # 检查箱子是否落在冰/场上 (field effects may apply)
        return True

    def _get_field_at(self, x: int, y: int) -> str | None:
        return self.fields.get((x, y))

    def _get_ice_at(self, x: int, y: int) -> bool:
        return (x, y) in self.ice

    def _trigger_event(self, event: str):
        if not self.events.get(event, False):
            self.events[event] = True
            self.event_first_step[event] = self.step_count
            if event not in self.first_seen:
                self.first_seen.add(event)
                self.event_order.append(event)

    def _apply_field_effects(self, pos: tuple[int, int]) -> tuple[int, int]:
        """对位置上的实体应用方向场效果"""
        x, y = pos
        field = self._get_field_at(x, y)
        if field is None:
            return pos

        if self.profile.get("field_mode") == "push_none":
            return pos

        dx, dy = self._move_dir(field)
        if dx == 0 and dy == 0:
            return pos

        steps = 0
        max_steps = self.profile.get("field_max_steps", 10) if self.profile.get("field_mode") == "push_continuous" else 1

        while steps < max_steps:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                break
            if self.grid[ny][nx] == '#':
                break
            if self.grid[ny][nx] == 'D' and self.doors.get((nx, ny), True):
                break
            # 移动
            ch = self.grid[y][x]
            self.grid[y][x] = '.'
            self.grid[ny][nx] = ch
            x, y = nx, ny
            steps += 1
            # 检查是否还有 field
            if self._get_field_at(x, y) is None:
                break

        return (x, y)

    def step(self, action: str) -> dict:
        """执行一步动作"""
        self.step_count += 1
        self.portal_used_this_step = False
        self.key_collected_this_step = False

        if self.terminal != "active":
            return self._make_obs()

        if action == 'WAIT':
            # 场效应仍然可能影响 agent (传送门不激活)
            if self.agent_pos:
                self.agent_pos = self._apply_field_effects(self.agent_pos)
            return self._check_terminal_and_events()

        # 移动动作
        dx, dy = self._move_dir(action)
        if dx == 0 and dy == 0:
            return self._make_obs()

        # 更新方向
        if self.profile.get("agent_faces_move_direction", True):
            self.agent_dir = action

        ax, ay = self.agent_pos
        nx, ny = ax + dx, ay + dy

        # 边界/墙壁碰撞
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
            self._trigger_event("collision")
            return self._check_terminal_and_events()

        target = self.grid[ny][nx]

        if target == '#':
            self._trigger_event("collision")
            return self._check_terminal_and_events()

        # 上锁的门
        if target == 'D' and self.doors.get((nx, ny), True):
            self._trigger_event("collision")
            return self._check_terminal_and_events()

        # 箱子推动
        if target == 'B':
            if self.profile.get("box_pushable", True):
                if not self._try_push_box(nx, ny, dx, dy):
                    self._trigger_event("collision")
                    return self._check_terminal_and_events()
            else:
                self._trigger_event("collision")
                return self._check_terminal_and_events()

        # 移动 agent
        if self.agent_pos:
            ox, oy = self.agent_pos
            self.grid[oy][ox] = '.'
        self.grid[ny][nx] = 'A'
        self.agent_pos = (nx, ny)

        # 检查当前位置的交互
        self._check_position_interactions(nx, ny)

        # 冰面滑动
        if self._get_ice_at(nx, ny) and self.profile.get("ice_mode") != "slide_none":
            self._apply_ice_slide(dx, dy)

        # 方向场效果
        if self.agent_pos and self._get_field_at(*self.agent_pos):
            if self.profile.get("field_priority") == "after_move":
                self.agent_pos = self._apply_field_effects(self.agent_pos)

        # 传送门
        if self.agent_pos in self.portals and self.profile.get("portal_mode") == "paired":
            dest = self._portal_graph.get(self.agent_pos)
            if dest and self.grid[dest[1]][dest[0]] not in '#BD':
                ox, oy = self.agent_pos
                self.grid[oy][ox] = '.'
                self.grid[dest[1]][dest[0]] = 'A'
                self.agent_pos = dest
                self.portal_used_this_step = True
                self._trigger_event("portal_used")
                # 检查传送目标
                self._check_position_interactions(*dest)

        return self._check_terminal_and_events()

    def _apply_ice_slide(self, dx: int, dy: int):
        """处理冰面滑动"""
        mode = self.profile.get("ice_mode", "slide_until_blocked")
        ice_dir = self.profile.get("ice_direction", "momentum")
        max_slide = self.profile.get("ice_max_slide", 10)

        if ice_dir == "momentum":
            sdx, sdy = dx, dy
        elif ice_dir == "reverse":
            sdx, sdy = -dx, -dy
        else:
            sdx, sdy = dx, dy

        steps = 0
        x, y = self.agent_pos

        while steps < max_slide:
            nx, ny = x + sdx, y + sdy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                break
            target = self.grid[ny][nx]
            if target == '#':
                break
            if target == 'D' and self.doors.get((nx, ny), True):
                break
            if target == 'B':
                if not self.profile.get("box_pushable", True):
                    break
                if not self._try_push_box(nx, ny, sdx, sdy):
                    break
            # 滑动
            self.grid[y][x] = '.'
            self.grid[ny][nx] = 'A'
            x, y = nx, ny
            self.agent_pos = (x, y)
            steps += 1
            self._check_position_interactions(x, y)

            # 检查传送门
            if (x, y) in self.portals:
                dest = self._portal_graph.get((x, y))
                if dest and self.grid[dest[1]][dest[0]] not in '#BD':
                    self.grid[y][x] = '.'
                    self.grid[dest[1]][dest[0]] = 'A'
                    x, y = dest
                    self.agent_pos = dest
                    self.portal_used_this_step = True
                    self._trigger_event("portal_used")
                    self._check_position_interactions(*dest)

            if mode == "slide_one":
                break
            if not self._get_ice_at(x, y):
                break

    def _check_position_interactions(self, x: int, y: int):
        """检查位置上的交互"""
        # 钥匙
        if (x, y) in self.keys:
            self._trigger_event("key_collected")
            self.keys_collected.add((x, y))
            del self.keys[(x, y)]
            self.grid[y][x] = 'A'
            self.key_collected_this_step = True

        # 门 —— 如果有钥匙则解锁
        if (x, y) in self.doors and self.doors[(x, y)]:
            if self.profile.get("door_requires_key", True):
                if self.keys_collected:
                    self.doors[(x, y)] = False
                    if self.profile.get("door_consumes_key", True):
                        self.keys_collected = set()  # 消耗钥匙
            else:
                self.doors[(x, y)] = False

        # 目标
        if (x, y) in self.goals:
            self._trigger_event("goal_reached")

        # 危险
        if (x, y) in self.hazards:
            self._trigger_event("hazard")

        # 检查 box_on_goal
        for box_pos in list(self.boxes.keys()):
            if box_pos in self.orbs:
                self._trigger_event("box_on_goal")

    def _check_terminal_and_events(self) -> dict:
        """检查终局条件"""
        if self.events.get("goal_reached"):
            self.terminal = "goal"
        elif self.events.get("hazard"):
            self.terminal = "hazard"
        elif self.events.get("collision"):
            self.terminal = "blocked"
        else:
            self.terminal = "active"
        return self._make_obs()

    def _make_obs(self) -> dict:
        return {
            "agent_pos": self.agent_pos,
            "agent_dir": self.agent_dir,
            "events": dict(self.events),
            "terminal": self.terminal,
        }

    def run(self, actions: list[str], horizon: int | None = None) -> dict:
        """执行动作序列"""
        limit = horizon if horizon is not None else len(actions)
        for i in range(min(limit, len(actions))):
            if self.terminal != "active":
                break
            self.step_count = i
            self.step(actions[i])

        self.total_steps = min(limit, len(actions))

        # 如果仍然 active 且没有更多动作，标记为 timeout
        if self.terminal == "active" and self.step_count >= self.total_steps - 1:
            if len(actions) == 0 or self.step_count >= len(actions):
                self.terminal = "timeout"

        # 修正：如果动作已全部执行完而仍然是 active
        if self.terminal == "active" and self.step_count >= len(actions):
            self.terminal = "timeout"

        return self.get_result()

    def get_result(self) -> dict:
        """获取当前世界的完整状态"""
        final_grid = ["".join(row) for row in self.grid]

        # 获取实体位置
        entities = {}
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                ch = self.grid[y][x]
                if ch in 'ABOK':
                    entities[ch] = f"{x},{y}"

        return {
            "final_grid": final_grid,
            "events": dict(self.events),
            "terminal": self.terminal,
            "entity_positions": entities,
            "event_first_step": dict(self.event_first_step),
            "event_order": list(self.event_order),
        }

    def get_event_timeline(self, horizon: int) -> dict[str, str]:
        """计算事件时间线"""
        timeline = {}
        for key in EVENT_KEYS:
            step = self.event_first_step.get(key)
            if step is None:
                timeline[key] = "never"
            else:
                ratio = step / max(horizon, 1)
                if ratio < 0.33:
                    timeline[key] = "early"
                elif ratio < 0.67:
                    timeline[key] = "mid"
                else:
                    timeline[key] = "late"
        return timeline

    def get_event_order_padded(self, n: int = 3) -> list[str]:
        """获取填充后的事件顺序"""
        order = list(self.event_order)
        while len(order) < n:
            order.append("none")
        return order[:n]


def grid_diff(initial: list[str], final: list[str]) -> list[dict]:
    """比较两个网格的差异"""
    diffs = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if initial[y][x] != final[y][x]:
                diffs.append({
                    "pos": f"{x},{y}",
                    "from": initial[y][x],
                    "to": final[y][x],
                })
    return diffs


if __name__ == "__main__":
    print("Simulator module loaded.")
