# NS-2026-08 星云值班台：黑箱遥测世界模型闭环处置

## 赛题简介

凌晨的星云业务集群开始抖动：部分服务延迟上升，队列积压，缓存命中率异常。作为值班 SRE，你需要从缺失、延迟和丢包的脱敏 telemetry 中估计事故状态，并在事故尚未完全暴露时决定下一步处置动作。

每步返回一个运维动作（扩缩容、回滚、重启、熔断、限流、索引维护、缓存预热、排队疏导、诊断探针或空动作），在 48 步内控制事故、最小化 SLO 损失。

## 架构

10 个服务，固定拓扑：

```
svc_00 (edge) ──→ svc_01 (api) ──→ svc_07 (cache)
              ├──→ svc_02 (api) ──→ svc_06 (store)
              │                 └─→ svc_07 (cache)
              └──→ svc_03 (api) ──→ svc_04 (api) ──→ svc_05 (store)
                              ├──→ svc_05 (store)
                              ├──→ svc_08 (queue) ←── svc_09 (worker)
                              └──→ svc_09 (worker) ──→ svc_05 (store)
```

8 种故障模式，全部为多故障场景（每任务 2-4 个并发故障）。

## 策略设计

### 核心思路：纯探针驱动

`DIAGNOSTIC_PROBE` 是唯一永远安全的动作（不触发 wrong_action_penalty），其返回的 `action_hint` 直接映射到正确处置：

| action_hint | 主动作 | 次动作 |
|---|---|---|
| `rollback_release` | ROLLBACK_RELEASE on source | CIRCUIT_BREAK on svc_00 |
| `rollback_config` | ROLLBACK_CONFIG on source | SET_TIMEOUT on source |
| `restart_scale` | RESTART on source | SCALE on source |
| `scale_throttle` | SCALE on svc_00 | THROTTLE on svc_00 |
| `storage_index` | REBUILD_INDEX on dependency | SET_TIMEOUT on service |
| `queue_drain` | DRAIN_QUEUE on svc_08 | SCALE on svc_09 |
| `cache_warm` | WARM_CACHE on svc_07 | THROTTLE on svc_00 |
| `timeout_circuit` | CIRCUIT_BREAK on service | SET_TIMEOUT on dependency |

### 控制流

1. 收到探针结果 → 按优先级施加治疗（主动作 + 次动作）
2. global_health ≥ 4 且无可用探针 → 紧急 SCALE svc_00
3. 健康趋势上升且有异常服务 → 对最差服务发射探针
4. 默认 → NOOP

### 关键决策

- **不用猜测性动作**：SCALE/THROTTLE/DRAIN_QUEUE 在不知道故障类型时可能是错误动作（每个罚 5.0 分），只通过探针诊断后精确治疗
- **区分 service 和 dependency**：`dependency_timeout` 的 CIRCUIT_BREAK 应作用于故障服务（probe target），而 SET_TIMEOUT 作用于依赖方（source_hint）
- **事件引导探针**：event 指向的服务优先探针，加速故障发现

## 结果

| 评估集 | 得分 | vs Oracle |
|--------|------|-----------|
| Public (16 tasks) | 1429.81 / 1500 | 96.4% |
| Valid (64 tasks) | 1292.07 / 1500 | 90.6% |

## 文件结构

```
.
├── agent.py              # 提交文件
├── action_schema.json    # 动作定义
├── agent_api.md          # 接口说明
├── public_tasks.jsonl    # 16 个调试任务
├── valid_tasks.jsonl     # 64 个验证任务
├── test_tasks.jsonl      # 32 个正式任务（仅 public view）
├── train_traces.jsonl    # 480 条训练轨迹
├── checker/              # 评分器和可视化
│   ├── aiops_world.py
│   ├── checker.py
│   └── visualize_trace.py
├── tools/                # 格式检查和评估脚本
│   ├── check_format.py
│   └── run_public_eval.py
└── example_submission/   # 示例提交
```

## 本地评估

```bash
# 格式检查
python tools/check_format.py <submission_dir>

# 公开评估（16 tasks）
python tools/run_public_eval.py <submission_dir>

# 验证评估（64 tasks）
python checker/checker.py --tasks valid_tasks.jsonl --submission <submission_dir>
```
