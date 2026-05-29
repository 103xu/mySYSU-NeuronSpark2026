"""
NS-2026-09 网格世界物理引擎
基于对训练数据的分析实现准确的物理模拟
"""

from __future__ import annotations

GRID_SIZE = 12
EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]
DIR_VEC = {'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0)}
FIELD_VEC = {'^': (0, -1), 'v': (0, 1), '<': (-1, 0), '>': (1, 0)}


class Entity:
    __slots__ = ('x', 'y', 'type')
    def __init__(self, x: int, y: int, etype: str):
        self.x = x
        self.y = y
        self.type = etype

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)


class GridWorld:
    """网格世界状态管理"""

    def __init__(self, grid: list[str]):
        self.static = [list(row) for row in grid]  # 静态地图（墙壁、冰、场、门等不动）
        self.dynamic = [['.' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 动态实体
        self.doors_locked: dict[tuple[int,int], bool] = {}
        self.removed_keys: set[tuple[int,int]] = set()
        self.removed_fields: set[tuple[int,int]] = set()
        self._init_grid(grid)

    def _init_grid(self, grid: list[str]):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                ch = grid[y][x]
                if ch in 'ABOK':
                    self.dynamic[y][x] = ch
                elif ch == 'D':
                    self.doors_locked[(x, y)] = True

    def get(self, x: int, y: int) -> str:
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return '#'
        # 动态实体优先
        if self.dynamic[y][x] != '.':
            return self.dynamic[y][x]
        # 被移除的钥匙/场
        if (x, y) in self.removed_keys or (x, y) in self.removed_fields:
            return '.'
        # 门
        if (x, y) in self.doors_locked and self.doors_locked[(x, y)]:
            return 'D'
        return self.static[y][x]

    @property
    def static_ch(self):
        """获取静态地图的字符（忽略动态实体）"""
        return self.static

    def is_static(self, x: int, y: int, ch: str) -> bool:
        """检查静态地图中某位置是否为特定字符"""
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return False
        if (x, y) in self.removed_keys:
            return ch == 'K'  # 已被收集的钥匙在静态意义上是 K
        if (x, y) in self.removed_fields:
            return False
        return self.static[y][x] == ch

    def move_entity(self, x: int, y: int, nx: int, ny: int) -> bool:
        """移动动态实体，返回是否成功"""
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
            return False
        ch = self.dynamic[y][x]
        if ch == '.':
            return False
        self.dynamic[y][x] = '.'
        self.dynamic[ny][nx] = ch
        return True

    def place_entity(self, x: int, y: int, ch: str):
        self.dynamic[y][x] = ch

    def remove_entity(self, x: int, y: int):
        self.dynamic[y][x] = '.'

    def to_full_grid(self) -> list[str]:
        result = []
        for y in range(GRID_SIZE):
            row = []
            for x in range(GRID_SIZE):
                row.append(self.get(x, y))
            result.append("".join(row))
        return result

    def find_entity(self, etype: str) -> tuple[int, int] | None:
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.dynamic[y][x] == etype:
                    return (x, y)
        return None

    def find_all(self, etype: str) -> list[tuple[int, int]]:
        result = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.dynamic[y][x] == etype:
                    result.append((x, y))
        return result


class GameEngine:
    """游戏引擎：执行动作序列并返回结果"""

    def __init__(self, grid: list[str]):
        self.world = GridWorld(grid)
        self.agent_dir = 'D'
        self.keys_held = 0
        self.action_count = 0
        self.events: dict[str, bool] = {k: False for k in EVENT_KEYS}
        self.event_step: dict[str, int] = {k: -1 for k in EVENT_KEYS}
        self.event_order: list[str] = []
        self.portal_cooldown = 0
        self.terminal = "active"
        self._infer_initial_dir()

    def _infer_initial_dir(self):
        """从观测推断初始方向"""
        agent_pos = self.world.find_entity('A')
        if agent_pos is None:
            return

    def _trigger(self, event: str):
        if not self.events[event]:
            self.events[event] = True
            self.event_step[event] = self.action_count
            self.event_order.append(event)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def step(self, action: str):
        """执行一步"""
        self.action_count += 1

        if self.terminal != "active":
            return

        # Step 0: 先处理所有方向场上的实体（全局场效应）
        self._process_global_fields()

        if action == 'WAIT':
            self._check_agent_position()
            self._check_box_on_goal()
            return

        dx, dy = DIR_VEC.get(action, (0, 0))
        if dx == 0 and dy == 0:
            return

        self.agent_dir = action
        agent_pos = self.world.find_entity('A')
        if agent_pos is None:
            return

        ax, ay = agent_pos
        tx, ty = ax + dx, ay + dy

        # 边界检查
        if not self._in_bounds(tx, ty):
            self._trigger('collision')
            self._check_agent_position()
            return

        target = self.world.get(tx, ty)

        # 墙壁
        if target == '#':
            self._trigger('collision')
            self._check_agent_position()
            return

        # 锁住的门
        if target == 'D' and self.world.doors_locked.get((tx, ty), True):
            if self.keys_held > 0:
                self.keys_held -= 1
                self.world.doors_locked[(tx, ty)] = False
            else:
                self._trigger('collision')
                self._check_agent_position()
                return

        # 箱子推动
        if target == 'B':
            bnx, bny = tx + dx, ty + dy
            can_push = (
                self._in_bounds(bnx, bny) and
                self.world.get(bnx, bny) not in '#BD' and
                self.world.dynamic[bny][bnx] not in 'B'
            )
            if can_push:
                self.world.move_entity(tx, ty, bnx, bny)
                if self.world.is_static(bnx, bny, 'O'):
                    self._trigger('box_on_goal')
            else:
                self._trigger('collision')
                self._check_agent_position()
                return

        # 移动 agent
        self.world.move_entity(ax, ay, tx, ty)

        # 检查传送门
        if self.world.is_static(tx, ty, 'P') and self.portal_cooldown <= 0:
            dest = self._find_portal_dest(tx, ty)
            if dest:
                dx2, dy2 = dest
                target2 = self.world.get(dx2, dy2)
                if target2 not in '#DB' and self.world.dynamic[dy2][dx2] not in 'B':
                    self.world.move_entity(tx, ty, dx2, dy2)
                    self._trigger('portal_used')
                    self.portal_cooldown = 3
                    tx, ty = dx2, dy2

        if self.portal_cooldown > 0:
            self.portal_cooldown -= 1

        # 冰面滑动
        if self.world.is_static(tx, ty, 'I'):
            self._slide_on_ice(tx, ty, dx, dy)

        # 检查位置交互
        self._check_agent_position()

        # 检查方向场（agent 现在在的位置）
        self._apply_field_to_agent()

        self._check_box_on_goal()

    def _slide_on_ice(self, x: int, y: int, dx: int, dy: int):
        """冰面滑动逻辑"""
        for _ in range(10):  # 最多滑 10 步
            if not self.world.is_static(x, y, 'I'):
                break
            nx, ny = x + dx, y + dy
            if not self._in_bounds(nx, ny):
                break
            target = self.world.get(nx, ny)
            if target == '#':
                break
            if target == 'D' and self.world.doors_locked.get((nx, ny), True):
                break
            if target == 'B':
                bnx, bny = nx + dx, ny + dy
                can_push = (
                    self._in_bounds(bnx, bny) and
                    self.world.get(bnx, bny) not in '#BD' and
                    self.world.dynamic[bny][bnx] not in 'B'
                )
                if not can_push:
                    break
                self.world.move_entity(nx, ny, bnx, bny)
                if self.world.is_static(bnx, bny, 'O'):
                    self._trigger('box_on_goal')

            self.world.move_entity(x, y, nx, ny)
            x, y = nx, ny

            # 冰上的传送门
            if self.world.is_static(x, y, 'P') and self.portal_cooldown <= 0:
                dest = self._find_portal_dest(x, y)
                if dest:
                    dx2, dy2 = dest
                    if self.world.get(dx2, dy2) not in '#DB':
                        self.world.move_entity(x, y, dx2, dy2)
                        self._trigger('portal_used')
                        self.portal_cooldown = 3
                        x, y = dx2, dy2
                        break

            self._check_agent_position()
            if self.events.get('hazard') or self.events.get('goal_reached'):
                break

    def _find_portal_dest(self, x: int, y: int) -> tuple[int, int] | None:
        """找到最近的其他传送门"""
        best = None
        best_dist = float('inf')
        for py in range(GRID_SIZE):
            for px in range(GRID_SIZE):
                if self.world.is_static(px, py, 'P') and (px, py) != (x, y):
                    dist = abs(px - x) + abs(py - y)
                    if dist < best_dist:
                        best_dist = dist
                        best = (px, py)
        return best

    def _apply_field_to_agent(self):
        """对 agent 应用方向场"""
        agent_pos = self.world.find_entity('A')
        if agent_pos is None:
            return
        x, y = agent_pos

        if not self.world.is_static(x, y, None):
            static_ch = self.world.static[y][x]
            if static_ch in '^v<>' and (x, y) not in self.world.removed_fields:
                dx, dy = FIELD_VEC[static_ch]
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny) and self.world.get(nx, ny) not in '#D':
                    if self.world.get(nx, ny) == 'B':
                        bnx, bny = nx + dx, ny + dy
                        if self._in_bounds(bnx, bny) and self.world.get(bnx, bny) not in '#BD':
                            self.world.move_entity(nx, ny, bnx, bny)
                        else:
                            return
                    self.world.move_entity(x, y, nx, ny)
                    self._check_agent_position()

    def _process_global_fields(self):
        """处理全局场效应：方向场影响上面的实体"""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                static_ch = self.world.static[y][x]
                if static_ch in '^v<>' and (x, y) not in self.world.removed_fields:
                    # 检查该位置上是否有实体（除了 agent）
                    dyn_ch = self.world.dynamic[y][x]
                    if dyn_ch in 'BO':  # 箱子和 orb 受场影响
                        dx, dy = FIELD_VEC[static_ch]
                        nx, ny = x + dx, y + dy
                        if (self._in_bounds(nx, ny) and
                            self.world.get(nx, ny) not in '#DB' and
                            self.world.dynamic[ny][nx] not in 'BO'):
                            self.world.move_entity(x, y, nx, ny)
                            # 检查 box_on_goal
                            if dyn_ch == 'B' and self.world.is_static(nx, ny, 'O'):
                                self._trigger('box_on_goal')

    def _check_agent_position(self):
        """检查 agent 位置的各种事件"""
        agent_pos = self.world.find_entity('A')
        if agent_pos is None:
            return
        x, y = agent_pos

        # 目标
        if self.world.is_static(x, y, 'G'):
            self._trigger('goal_reached')

        # 危险
        if self.world.is_static(x, y, 'H'):
            self._trigger('hazard')

        # 钥匙
        if self.world.is_static(x, y, 'K') and (x, y) not in self.world.removed_keys:
            self._trigger('key_collected')
            self.keys_held += 1
            self.world.removed_keys.add((x, y))

    def _check_box_on_goal(self):
        """检查 box 是否在 orb 上"""
        boxes = self.world.find_all('B')
        orbs = self.world.find_all('O')
        for bpos in boxes:
            for opos in orbs:
                if bpos == opos:
                    self._trigger('box_on_goal')

    def run(self, actions: list[str], horizon: int = 0) -> dict:
        limit = horizon if horizon > 0 else len(actions)
        max_steps = min(limit, len(actions))

        for i in range(max_steps):
            if self.terminal != "active":
                break
            self.step(actions[i])

        # 判定终局
        if self.events.get('goal_reached'):
            self.terminal = 'goal'
        elif self.events.get('hazard'):
            self.terminal = 'hazard'
        elif self.events.get('collision'):
            self.terminal = 'blocked'
        elif self.action_count >= max_steps:
            self.terminal = 'timeout'
        else:
            self.terminal = 'timeout'

        return self._result(max(horizon, max_steps))

    def _result(self, horizon: int) -> dict:
        final_grid = self.world.to_full_grid()

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

        order = list(self.event_order)
        while len(order) < 3:
            order.append("none")

        return {
            "final_grid": final_grid,
            "events": dict(self.events),
            "event_timeline": timeline,
            "event_order": order[:3],
            "terminal": self.terminal,
        }


def run_simulation(grid: list[str], actions: list[str], horizon: int = 0) -> dict:
    engine = GameEngine(grid)
    return engine.run(actions, horizon)
