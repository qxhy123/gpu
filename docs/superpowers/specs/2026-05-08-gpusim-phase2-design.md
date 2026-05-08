# gpusim Phase 2 — L1/L2 Cache + HBM Bandwidth 模型设计文档

**日期**：2026-05-08
**状态**：设计阶段（待实现）
**作者**：与 Claude 协同 brainstorm
**前置依赖**：Phase 1 完成（tag `phase1-complete`，HEAD `6b0ee5e` + 后续 fix），见 `2026-05-07-gpusim-phase1-design.md`
**范围**：仅 Phase 2。Phase 3+ 仅作为愿景列出。

---

## 1. 愿景与 Phase 2 范围

### 1.1 项目背景

Phase 1 交付了一个单 SM、cycle-approximate 的教学 GPU 模拟器。其内存模型只到"global memory = 固定 400 cycle"——足以教 SIMT、coalescing、bank conflict 等概念，但**不能解释为什么实际 kernel cycle 数远低于这个上限**（cache 命中），也**无法可视化 bandwidth saturation**。

Phase 2 把"global memory 黑盒"换成完整的真实 cache+HBM 层级。

### 1.2 Phase 2 一句话目标

> 让 Phase 1 的 `ld.global` 不再是固定 400 cycle 的黑盒，而是流过真实的 L1（128KB tag-precise + 16 MSHR）→ L2（4MB tag-precise + write-back）→ HBM（8 channel × 16 bank + row buffer），把 working set、line-level coalescing、bandwidth saturation、row-buffer locality 四个核心现象在 HTML 报告里可视化。

### 1.3 路线图回顾

| Phase | 范围 | 状态 |
|---|---|---|
| Phase 1 | 单 SM、cycle-approximate、PTX 子集、shared/global memory 无 cache、6 example、8 讲义 | ✅ 已完成 |
| **Phase 2** | **L1/L2 cache（tag-precise + MSHR）+ HBM channel/bank/row buffer + 4 example + 4 讲义** | **本文档** |
| Phase 3 | Tensor Core、FP16/BF16/FP8、wgmma | 后续 |
| Phase 4 | 多 SM、CTA→SM 调度、L2 跨 SM 共享 | 后续 |
| Phase 5 | 多 GPU、NVLink、NCCL collective | 后续 |

### 1.4 已锁定决策

| 维度 | 决策 |
|---|---|
| 建模深度 | Tag-precise L1/L2 + 详细 HBM（channel + bank + row buffer） |
| L1 ↔ shared 关系 | 共享 256 KB SRAM，启动时按比例切分（Hopper 模式） |
| L1 容量默认 | 128 KB（剩 128 KB 给 shared，可调） |
| L2 容量 | 4 MB（教学缩放；真机 H100 = 60 MB） |
| Cache line size | 128 B（与 Phase 1 sector 统一） |
| 替换策略 | LRU per-set；L1 4-way；L2 16-way |
| 写策略 | L1 write-through + no-write-allocate；L2 write-back |
| MSHR | L1 配 16 个 MSHR slot；line 级 coalescing；满则 `MSHR_FULL` stall |
| HBM 粒度 | 8 channels × 16 banks，row size 4 KB，每 channel 100 GB/s |
| Atomics、prefetch、fence | 不实现（推到后续 phase 或永久 out-of-scope） |
| 新增 examples | 4 个：l1_thrash / smem_vs_l1 / bw_saturation / row_buffer |
| 新增 tutorial 章节 | 4 章 |
| Phase 1 兼容 | parity 不变（数值正确）；cycle 数会变（带 cache 路径），微基准断言可能要松动 |

---

## 2. 架构总图与模块改动

### 2.1 数据流变化

**Phase 1**：`ld.global` → 固定 400 cycle 后 register 写回。

**Phase 2**：

```
warp issue ld.global
  ├─ functional value 立即从 backing buffer 读出（不变）
  └─ timing path（新）：
       L1 lookup（并行 set）
         ├─ HIT  → ready_at = now + 25 (L1_hit_latency)
         └─ MISS
             ├─ MSHR check：line 已在 fetch？合并到现有 MSHR
             ├─ MSHR 满  → STALL: MSHR_FULL
             └─ 新 MSHR：issue L2 lookup
                  ├─ HIT  → ready_at = now + L1_miss_check + L2_hit (~200)
                  └─ MISS → HBM request
                       │   1) 地址 → (channel, bank, row)
                       │   2) 入 channel queue（建模 bandwidth saturation）
                       │   3) 等待出队后服务：
                       │        ├─ row HIT  → ~10 cycle
                       │        └─ row MISS → ~30 cycle (ACT + DATA)
                       └─ ready_at = now + 全程累加
```

### 2.2 关键不变量

- **Functional value 直接读 backing buffer**（与 Phase 1 一致）
- Cache + HBM **只影响 timing**，不影响数值
- **Tag-only simulation**：cache line 不存数据，只存 tag/valid/dirty/lru_pos
- 所有 latency 同步预测式计算（在 issue cycle 一次性算出 ready_at），与 Phase 1 风格一致
- Trace 是防火墙：cache/HBM 事件全经 Recorder；analysis/viz 只消费 trace

### 2.3 模块拓扑

```
                 ┌─────────────────────────────────────────┐
                 │                  Device                  │  ← 新顶层
                 │  ┌─────────────┐  ┌─────────────────┐   │
                 │  │     SM      │  │       L2        │   │  ← 新模块
                 │  │             │  │  4 MB / 16-way  │   │
                 │  │ ┌─────────┐ │  │  write-back     │   │
                 │  │ │   L1    │◄┼──┤  (Phase 4 跨SM共享) │
                 │  │ │ 128 KB  │ │  └────────┬────────┘   │
                 │  │ │  4-way  │ │           │            │
                 │  │ │ + 16    │ │           ▼            │
                 │  │ │  MSHRs  │ │  ┌─────────────────┐   │  ← 新模块
                 │  │ └─────────┘ │  │       HBM       │   │
                 │  │             │  │ 8 ch × 16 bk    │   │
                 │  │  4 sub-cores│  │ row buffer      │   │
                 │  └─────────────┘  │ 100 GB/s/ch     │   │
                 │                   └────────┬────────┘   │
                 │                            ▼            │
                 │                   ┌─────────────────┐   │  ← 不变（Phase 1）
                 │                   │  numpy buffers  │   │
                 │                   │ (functional     │   │
                 │                   │  backing store) │   │
                 │                   └─────────────────┘   │
                 └─────────────────────────────────────────┘
```

### 2.4 文件改动

**新增**：
```
gpusim/core/cache/
  __init__.py
  line.py          # CacheLine dataclass: tag/valid/dirty/lru_pos
  l1.py            # L1Cache: lookup/install/evict/MSHR-merge
  l2.py            # L2Cache: same shape, write-back + dirty bit
  mshr.py          # MSHR pool: allocate/merge/release
gpusim/core/hbm.py # HBM: 8 channels × 16 banks, row buffer state, queue model
gpusim/core/device.py  # Device: 拥有 SM + L2 + HBM；公共 entry point
```

**修改**：

| 文件 | 改动 |
|---|---|
| `gpusim/core/sm.py` | 持有 L1（per-SM）；run() 改为 device 入口委托过来 |
| `gpusim/core/sub_core.py` | `_issue` 中 gmem 路径改为经 cache 层；不再自己算固定 latency |
| `gpusim/core/exec.py` | `GlobalMemory` 仍是 functional backing，不直接被 SubCore 访问 |
| `gpusim/core/warp.py` | + `StallReason.MSHR_FULL` |
| `gpusim/config/schema.py` | + `CacheConfig`, `HBMConfig` dataclass |
| `gpusim/config/default_hopper.yaml` | + cache + hbm 两节 |
| `gpusim/trace/events.py` | + `L1Event`, `L2Event`, `HBMEvent` |
| `gpusim/trace/recorder.py` | + `l1_access`, `l2_access`, `hbm_access` |
| `gpusim/trace/writer.py` | + 3 个新 parquet 文件 |
| `gpusim/analysis/metrics.py` | + cache hit rate / bandwidth / row buffer 等 |
| `gpusim/viz/html_report.py` | + 5 个新报告节 |
| `gpusim/api.py` | `Result` 加 `cache_metrics` + 3 个 events_df 属性 |

### 2.5 边界原则

1. **Functional vs timing 分离**：cache 子系统**只读 tag、不读 data**；data 永远从 numpy backing 读
2. **Device 作为顶层 entry**：Phase 4 multi-SM 时，Device 持有多个 SM 共享同一 L2 + HBM——拓扑天然扩展
3. **Trace 是防火墙**：cache/HBM 事件全经 Recorder
4. **API 兼容**：`gpusim.run(...)` 签名不变；Phase 1 例子参数不动

---

## 3. L1 Cache + MSHR 设计

### 3.1 数据结构

```python
@dataclass
class CacheLine:
    tag: int        # high bits of line address
    valid: bool
    lru_pos: int    # 0..ways-1; 0 = MRU
    # tag-only：不存 data
```

**L1 cache 结构**：
- 容量 = 128 KB，line size = 128 B，4-way → 1024 lines / 4 = **256 sets**
- 地址解码：`line_addr = phys_addr >> 7`；`set_idx = line_addr & 0xFF`；`tag = line_addr >> 8`
- 用 `dict[set_idx, list[CacheLine]]` 存（4 ways/set 静态分配）
- L1 大小可配置（受限于 L1+shared = 256 KB 共享 SRAM 池）

**MSHR pool**：

```python
@dataclass
class MSHREntry:
    line_addr: int
    issued_at: int                    # cycle when miss was first detected
    expected_complete: int            # cycle when L1 will install line
    waiters: list[Waiter]             # all warps waiting on this line

@dataclass
class Waiter:
    warp_id: int
    dst_regs: tuple[str, ...]         # scoreboard entries to release on completion
```

- Pool 大小 = 16 slots（可配置）
- Allocate：新 line miss → 检查是否已有 MSHR 命中此 line（**line 级 coalescing**）
- Merge：命中 → 把当前 warp/dst_regs 添加到现有 MSHR 的 waiters
- Release：MSHR 完成 → 安装 line 到 L1（驱逐 LRU）→ 通知所有 waiters → 释放 MSHR slot
- 满 → 上层（SubCore）回滚 issue，状态记 `MSHR_FULL`

### 3.2 Lookup 算法

```python
def access(self, line_addr: int, *, warp_id: int, dst_regs: tuple[str, ...],
           mode: Literal["load", "store"], now: int) -> AccessResult:
    """
    Returns one of:
      - Hit(ready_at = now + L1_hit_latency)
      - MissNewMSHR(ready_at = expected_complete)
      - MissMergeMSHR(ready_at = expected_complete)
      - Reject()                                   # MSHR pool full → caller stalls
    """
    set_idx = line_addr & SET_MASK
    tag = line_addr >> SET_BITS
    line = self._sets[set_idx].find(tag)

    if line is not None:                            # HIT
        self._touch_lru(set_idx, line)
        return Hit(ready_at = now + L1_HIT_LATENCY)

    # Miss + write-through-no-write-allocate: store-miss bypass L1 entirely
    if mode == "store":
        return Hit(ready_at = now + 1)              # 1-cycle "issued" for stores

    # Load miss: try MSHR merge
    if mshr := self._find_mshr(line_addr):
        mshr.add_waiter(warp_id, dst_regs)
        return MissMergeMSHR(ready_at = mshr.expected_complete)

    if self._mshr_pool.full():
        return Reject()                              # → STALL: MSHR_FULL

    expected = self._issue_downstream_fetch(line_addr, now)
    self._mshr_pool.allocate(line_addr, now, expected, warp_id, dst_regs)
    return MissNewMSHR(ready_at = expected)
```

### 3.3 LRU 更新规则

- per-set，每 set 4 ways
- 每条 way 持 `lru_pos ∈ [0, 3]`，0 = MRU，3 = LRU
- **Hit / install**：选中 way `lru_pos = 0`；其他 ways `lru_pos += 1`（最大裁到 3）
- **Eviction**：选 `lru_pos == 3` 的 way

### 3.4 写策略详细行为

| 场景 | L1 行为 |
|---|---|
| **store hit** | LRU 更新（hit 算 touch）；line 不变 |
| **store miss** | **不分配 line**（no-write-allocate）；直传 L2 |
| **load hit** | LRU 更新；返回 ready=now+25 |
| **load miss new** | 分配 MSHR；触发 L2 lookup |
| **load miss merge** | 加入现有 MSHR 等待者 |
| **eviction** | 静默丢弃（write-through 无脏数据） |

**关键不变量**：L1 永远不需要 dirty bit（write-through 保证 line 与 L2 一致）。

### 3.5 与 Phase 1 的衔接

- Phase 1 的 `gmem_latency` 默认 400 → Phase 2 拆成：L1 hit 25 / L2 hit 200 / HBM serve 100~130
- `coalescing_info(addresses)` 还在用：每个 transaction = 一条 cache line，每条 line 都进 L1 access pipeline
- 所以 Phase 1 的"32 lane → N transaction"逻辑不变；Phase 2 在其上再加"N transaction → M unique cache lines → MSHR merge 后实际 K 次下游 fetch"

教学话术：**warp coalescing** 是 lane 层面的合并；**cache-line MSHR coalescing** 是 transaction 层面的合并。两者叠加才是真实 GPU 的 effective bandwidth 故事。

---

## 4. L2 Cache 设计

### 4.1 数据结构

容量 = 4 MB，line size = 128 B，16-way → **2048 sets**。

```python
@dataclass
class L2Line:
    tag: int
    valid: bool
    dirty: bool          # write-back 才需要
    lru_pos: int
```

地址解码：`line_addr = phys_addr >> 7`；`set_idx = line_addr & 0x7FF`；`tag = line_addr >> 11`。

### 4.2 不做 MSHR 的简化

**L2 没有自己的 MSHR**。理由：
- L2 唯一上游来源是当前 SM 的 L1 MSHR
- L1 MSHR 已经做了 line 级 coalescing → L2 看到的 fetch 请求都是 unique line
- → L2 多 MSHR 是冗余

Phase 4 multi-SM 时 L2 会需要加 MSHR（不同 SM 的 L1 不互相协调），spec 里写明 Phase 2 单 SM 的合理简化。

### 4.3 写策略

L2 是 **write-allocate + write-back**，与 L1 形成"write-through 上层 + write-back 下层"的标准组合：

| 场景 | L2 行为 |
|---|---|
| **load miss** | 触发 HBM read → 安装 line |
| **store miss**（来自 L1 no-write-allocate） | 触发 HBM read（**write-allocate**：拉 line）→ 安装 → 标 dirty |
| **store hit** | 标 dirty；**不**立即写 HBM |
| **eviction**（替换 LRU） | 若 `dirty`：触发 HBM write（占 channel 写带宽）；否则静默丢弃 |

**关键**：dirty L2 line 在 eviction 时才写 HBM；store 流量不一定立即冲击 HBM 带宽，但被驱逐时会"延迟"地占用——bandwidth 教学的一个微妙点。

### 4.4 Lookup 算法

```python
def access(self, line_addr: int, *, mode: Literal["load","store"], now: int
           ) -> tuple[bool, int]:  # (hit, ready_at)
    set_idx = line_addr & 0x7FF
    tag = line_addr >> 11
    line = self._sets[set_idx].find(tag)

    if line is not None:                              # HIT
        self._touch_lru(set_idx, line)
        if mode == "store":
            line.dirty = True
        return (True, now + L2_HIT_LATENCY)

    # MISS — fetch from HBM
    hbm_complete = self._hbm.request(line_addr, now)
    self._install_line(line_addr, dirty=(mode == "store"), install_at=hbm_complete)
    return (False, hbm_complete + L2_MISS_INSTALL)


def _install_line(self, line_addr, dirty: bool, install_at: int):
    set_idx = line_addr & 0x7FF
    victim = self._sets[set_idx].pick_lru()
    if victim.valid and victim.dirty:
        # write-back: queue HBM write, doesn't block install
        victim_addr = (victim.tag << 11) | set_idx
        self._hbm.write_request(victim_addr, install_at)
    self._sets[set_idx].install(line_addr, dirty=dirty)
```

---

## 5. HBM 模型设计

### 5.1 物理结构

- **8 channels**（每 channel 独立 bus）
- 每 channel **16 banks**（不同 bank 可独立 row 状态）
- 每 bank 有 **row buffer**：当前打开的 row 索引；row size = **4 KB**

### 5.2 地址解码

```
addr 64-bit 布局（低位到高位）：
  [6:0]    byte offset within 128 B cache line
  [9:7]    channel    (3 bits → 8)
  [14:10]  col-in-row (5 bits → 32 lines/row)
  [18:15]  bank       (4 bits → 16)
  [30:19]  row        (12 bits → 4096 rows/bank)
  [63:31]  unused
```

总寻址空间 = 8 ch × 16 bank × 4096 rows × 32 lines × 128 B = 2 GB。

**为什么 channel 紧贴 line offset**：连续 cache line（addr += 128）会循环切换 channel——这是教学的关键。一个 warp 的 32 lane 连续访问（coalesced）天然分散到所有 8 channel；而 stride > 8 line 的访问会落在同一 channel，触发 channel 队列竞争。

**为什么 col-in-row 在 channel 之上**：8 个 channel 各自看到 stride-1024 的访问序列（每个 channel 一行 line × 32 col-in-row = 4 KB row，全在 row 0 内）。这意味着 sequential streaming 既享受 8-way channel 并行，又享受 row buffer hit。

**Trade-off 与 row-miss 的非直觉性**：在这种 layout 下，**stride = row_size (4 KB) 不会触发 row miss**——4 KB 增量只会让 col-in-row 跳跃，仍在同 row 内。要让每次访问都 row miss，stride 必须跳过 (channel × col × bank) = 8 × 32 × 16 × 128 B = **512 KB**。`row_buffer_demo` README 必须把这点讲清。

### 5.3 状态变量

```python
class HBM:
    channel_busy_until: list[int]                # [8]; 各 channel bus 下次空闲的 cycle
    bank_open_row: list[list[int | None]]        # [8][16]; 各 bank 当前打开的 row
```

### 5.4 服务算法（教学简化版）

```python
def service(self, addr: int, kind: str, now: int) -> int:
    """Return the cycle when this request completes (data ready)."""
    c    = (addr >> 7)  & 0x7    # channel  bits [9:7]
    b    = (addr >> 15) & 0xF    # bank     bits [18:15]
    row  = (addr >> 19) & 0xFFF  # row      bits [30:19]

    start = max(now, self._channel_busy_until[c])    # channel queue serialization

    if self._bank_open_row[c][b] == row:
        latency = ROW_HIT_LATENCY       # ~10 cycles
        row_kind = "ROW_HIT"
    else:
        latency = ROW_MISS_LATENCY      # ~30 cycles (ACT + DATA)
        self._bank_open_row[c][b] = row
        row_kind = "ROW_MISS"

    end = start + latency
    self._channel_busy_until[c] = end                # bus busy for full latency

    self._recorder.hbm_access(
        cycle=now, served_at=end, addr=addr, channel=c, bank=b, row=row,
        kind=kind,                  # "READ" | "WRITE_BACK"
        row_kind=row_kind,
        queue_wait=start - now,     # 教学指标
    )
    return end
```

### 5.5 教学简化与真机的差异

| 真机行为 | Phase 2 简化 | 影响 |
|---|---|---|
| 同 channel 不同 bank 可有部分 ACT/DATA 并行 | channel bus 完整序列化所有 latency | 丢失 bank 级 ACT/DATA overlap；换得清晰的 channel queue 故事 |
| ACT/PRE/RD/WR 是独立命令 | 单一 latency = ROW_HIT 或 ROW_MISS | 不教 DRAM command-level 行为 |
| DRAM refresh 周期性占带宽 | 不建模 | 影响 < 5% 真机带宽 |
| Bank 间 ACT 受 tFAW 约束 | 不建模 | 同上 |

---

## 6. Trace + 分析 + 可视化

### 6.1 完整事件清单

| 事件类型 | 来源 | 频率 |
|---|---|---|
| `WARP_STATE`（Phase 1） | SubCore 每 cycle 每 warp | 高（RLE 压缩） |
| `INSTR_ISSUE`（Phase 1） | SubCore on issue | 中 |
| `SMEM_ACCESS`（Phase 1） | SubCore on shared op | 低 |
| `GMEM_ACCESS`（Phase 1） | SubCore on global op | 低 |
| `DIV_PUSH/POP`（Phase 1） | SubCore on bra | 低 |
| `BAR_REACH/RELEASE`（Phase 1） | SM | 低 |
| `CTA_LAUNCH/RETIRE`（Phase 1） | SM | 低 |
| `L1_ACCESS` ⭐ | L1 cache | 高（每 cache line 一次） |
| `L2_ACCESS` ⭐ | L2 cache | 中（仅 L1 miss） |
| `HBM_ACCESS` ⭐ | HBM | 低（仅 L2 miss + WB） |

⭐ = Phase 2 新增。

### 6.2 新事件 schema

```python
@dataclass(frozen=True)
class L1Event:
    kind: str               # "HIT" | "MISS_NEW" | "MISS_MERGE"
    cycle: int
    warp_id: int
    line_addr: int
    set_idx: int
    way: int                # for HIT: way that hit; for MISS_NEW: way evicted
    mshr_slot: int | None   # for MISS_*

@dataclass(frozen=True)
class L2Event:
    kind: str               # "HIT" | "MISS_LOAD" | "MISS_STORE" | "EVICT_CLEAN" | "EVICT_DIRTY"
    cycle: int
    line_addr: int
    set_idx: int
    way: int
    victim_addr: int = -1   # for EVICT_*

@dataclass(frozen=True)
class HBMEvent:
    kind: str          # "READ" | "WRITE_BACK"
    row_kind: str      # "ROW_HIT" | "ROW_MISS"
    cycle: int
    served_at: int
    addr: int
    channel: int
    bank: int
    row: int
    queue_wait: int
```

### 6.3 新增分析指标

`gpusim/analysis/metrics.py` 加：

| 函数 | 输出 |
|---|---|
| `l1_hit_rate(l1_df)` | scalar |
| `l2_hit_rate(l2_df)` | scalar |
| `mshr_merge_rate(l1_df)` | scalar |
| `cache_hierarchy_breakdown(l1_df, l2_df)` | dict (l1_hit/l2_hit/hbm percentages) |
| `bandwidth_per_channel(hbm_df, cycles)` | DataFrame[8] |
| `channel_utilization(hbm_df, cycles)` | DataFrame[8] |
| `row_buffer_hit_rate(hbm_df)` | scalar |
| `queue_wait_distribution(hbm_df)` | pd.Series |
| `wb_traffic_fraction(hbm_df)` | scalar |
| `eviction_per_set(l1_df, l2_df)` | DataFrame |

### 6.4 HTML 报告新增节

在 Phase 1 现有 5 节之下：

| 节 | 内容 |
|---|---|
| **§6 Cache hierarchy hit rate** | Stacked bar：L1 hit / L2 hit / HBM；MSHR merge rate；MSHR_FULL 总数 |
| **§7 HBM channel utilization** | 时序图：8 条线每 channel busy 占比；queue_wait 分布 |
| **§8 Row buffer locality** | Pie：ROW_HIT vs ROW_MISS；时序：row miss 密度 |
| **§9 Write-back traffic** | Bar：READ vs WRITE_BACK 字节；dirty evictions per kernel |
| **§10 Eviction heatmap**（仅 thrash 类 kernel） | 每 set eviction 计数 heatmap |

### 6.5 Result API 扩展

```python
@dataclass
class Result:
    # Phase 1 fields...
    outputs: dict
    mode: str
    metrics: dict

    # Phase 2 additions
    @property
    def l1_events_df(self) -> pd.DataFrame: ...
    @property
    def l2_events_df(self) -> pd.DataFrame: ...
    @property
    def hbm_events_df(self) -> pd.DataFrame: ...
    @property
    def cache_metrics(self) -> dict: ...
    @property
    def bandwidth_df(self) -> pd.DataFrame: ...

    def cache_summary(self) -> str:
        """One-line: L1 hit X% / L2 hit Y% / HBM Z%, BW util A%, row hit B%."""
```

`Result.summary()` 也升级一行带 cache breakdown。

### 6.6 Perfetto 集成

L1/L2/HBM 事件作为 instant events 加进 Perfetto trace（每条 event 一条 instant 标记）；warp track 不变。

---

## 7. 测试策略

延续 Phase 1 三层金字塔。

### 7.1 单元测试（pytest）

| 模块 | 关键测试 |
|---|---|
| `core/cache/line` | tag/valid/dirty/lru_pos 字段 round-trip |
| `core/cache/l1` | 32 种 stride 模式的 hit/miss 序列、LRU eviction、write-through 行为 |
| `core/cache/mshr` | 多 warp 同 line 合并、pool 满时拒绝、release 后 slot 回收 |
| `core/cache/l2` | write-back dirty bit 状态、dirty eviction 触发 HBM write、no-MSHR 假设下的正确性 |
| `core/hbm` | 8 channel 地址 hash 正确、row_buffer 状态机、queue serialization |
| `analysis/metrics` | hit rate / bw / row hit 等所有新指标的 fixture |
| `viz/html_report` | 5 个新节都正确插入、Plotly 数据非空 |

### 7.2 Functional Parity（numpy）

4 个新 example 各有 numpy 参考实现，rtol=1e-5 对比。Phase 1 example 全部继续通过（数值不依赖 cache）。

### 7.3 Reference Fixture（真机对照）

`tests/reference/data/<name>.ref.json` 接口扩展：4 个新 kernel 的 schema。`gen_reference.py` 用户可在 H100 上跑产出。

新增对比维度（仅指标层；timing 层仍不对账）：
- `l1_hit_rate` ±10%
- `l2_hit_rate` ±15%
- `bandwidth_per_channel` ±20%（缩放后的 4MB L2 vs 真机 60MB 会有显著差距，按比例对账）

### 7.4 微基准（教科书事实）

新增固化断言：

```
- 数据集 fit L1 → l1_hit_rate ≥ 0.95（首轮 cold miss 后）
- 数据集 > L1 fit L2 → l1_hit_rate < 0.5 AND l2_hit_rate ≥ 0.95
- 单 channel saturation → 后续 queue_wait 显著 > 0
- sequential addr → row_buffer_hit_rate ≥ 0.95
- stride = 512 KB → row_buffer_hit_rate ≤ 0.1（layout 决定的 row-miss 触发 stride，见 §5.2）
```

Phase 1 微基准 `test_one_warp_kernel_ipc_le_1` 等如果断言精确 cycle 数 → 改为"≥"或区间。

---

## 8. 项目结构改动

### 8.1 目录新增

```
gpusim/core/cache/
  __init__.py
  line.py
  l1.py
  l2.py
  mshr.py
gpusim/core/hbm.py
gpusim/core/device.py

tests/unit/cache/
  __init__.py
  test_line.py
  test_l1.py
  test_l2.py
  test_mshr.py
tests/unit/core/test_hbm.py
tests/unit/core/test_device.py
```

### 8.2 配置文件迁移

`default_hopper.yaml` 加：

```yaml
cache:
  l1_size_bytes: 131072        # 128 KB; 与 smem 共享 256 KB SRAM
  l1_ways: 4
  l1_line_bytes: 128
  l1_hit_latency: 25
  l1_miss_check_latency: 5
  mshr_slots: 16
  l2_size_bytes: 4194304       # 4 MB
  l2_ways: 16
  l2_hit_latency: 200
  l2_miss_install_latency: 10

hbm:
  channels: 8
  banks_per_channel: 16
  row_size_bytes: 4096
  row_hit_latency: 10
  row_miss_latency: 30
```

Phase 1 配置文件加这两节后即可在 Phase 2 下跑；不加时使用默认值。

### 8.3 依赖

无新增 runtime 依赖。`pyproject.toml` 不变。

---

## 9. 教学示例与讲义

### 9.1 4 个新 example kernels

每个独立目录、可一键复现（Phase 1 风格）：`kernel.ptx + kernel.cu + reference.py + run.py + README.md`。

| # | Kernel | 教学意图 | 关键现象 |
|---|---|---|---|
| 1 | **l1_thrash_demo** | L1 容量与 working set | 三种数据规模配置，逐步突破 L1/L2/HBM；hit-rate 直方图阶跃可见 |
| 2 | **smem_vs_l1_demo** | 手动 caching vs 自动 caching | 同一 64×64 matmul 两个变体；smem 版 HBM 流量低，L1 版 HBM 流量随 thrash 上升 |
| 3 | **bw_saturation_demo** | Channel 并行 vs 序列化 | 低/高并发 launch 对比；高并发下 channel utilization 接近 1.0，queue_wait 上升 |
| 4 | **row_buffer_demo** | DRAM row locality | sequential (stride=128 B) vs row-miss stride (= **512 KB**，原因见 §5.2 layout)；row_buffer_hit_rate ≈1.0 vs ≈0 |

### 9.2 4 个新讲义章节

| # | 标题 | 关联 example |
|---|---|---|
| 08 | Cache hierarchy：L1、L2、working set | l1_thrash_demo |
| 09 | Shared memory vs cache：什么时候要手动管理 | smem_vs_l1_demo |
| 10 | HBM bandwidth：channel 并行与排队 | bw_saturation_demo |
| 11 | Row-buffer locality：DRAM 的隐藏维度 | row_buffer_demo |

每章结尾固定栏目：**看模拟器** / **改一改** / **真机对照**（如果 fixture 存在）。讲义专注 Phase 2 内的概念。

---

## 10. 与 Phase 1 兼容性

### 10.1 不会破坏的部分

| 维度 | 状态 |
|---|---|
| `gpusim.run(...)` 函数签名 | 不变 |
| Phase 1 example PTX 文件 | 不动 |
| Phase 1 example parity 测试（数值正确） | 全部继续通过 |
| `gpusim.cli` CLI 命令集 | 不变（新增 `--cache-config` 是可选 flag） |
| Result 旧字段 | 不变 |
| 配置 yaml 旧 section | 不变 |
| Stall token 前 10 类 | 不变 |

### 10.2 会变的部分

| 维度 | 变化 |
|---|---|
| **Cycle 数** | Phase 1 example 在 Phase 2 下 cycle 数会变（L1 hit 快路径）。微基准断言可能要松动 |
| **Stall 分类直方图** | 多了一类 `MSHR_FULL` |
| **HTML 报告** | 多 5 节；旧 5 节位置不变 |
| **Perfetto trace** | 多了 L1/L2/HBM instant 事件；旧 warp track 不变 |
| **Trace parquet schema** | 多 3 个文件（l1.parquet / l2.parquet / hbm.parquet） |
| **`Result` 类** | 多 5 个属性 |

---

## 11. 显式不在范围内（Phase 2）

记录在此以避免未来误解：

- Atomics（`atom.*`）
- Memory fence / coherency
- L1 / L2 prefetching
- TMA（Tensor Memory Accelerator）
- Tensor Core / MMA / wgmma
- FP16 / BF16 / FP8 数据通路
- Multi-SM L2 共享（Phase 4）
- DRAM command-level 时序（ACT/PRE/RD/WR 拆分）
- DRAM refresh
- Bank 间 tFAW 约束
- 多 GPU、NVLink、NCCL
- Volta+ ITS（与 Phase 1 一致继承）

---

## 12. 已知近似与简化

- **时序参数非官方**：所有 cycle 延迟为综合公开材料的合理近似值，不等同于 H100 真机
- **L2 大小缩放**：4 MB（真机 60 MB），在 default_hopper.yaml 标注
- **HBM 单 channel 带宽缩放**：100 GB/s（真机 H100 总带宽 ≈ 3 TB/s 分摊到 12-16 channel ≈ 200+ GB/s/channel）
- **HBM channel 完整序列化所有 latency**：丢失 bank-level ACT/DATA overlap，换教学清晰度
- **L2 不做 MSHR**：依赖 L1 MSHR 的 single-SM coalescing；Phase 4 加
- **Tag-only simulation**：不存 data，cache 内容 = numpy backing；这意味着不能模拟 cache 状态被 GPU side 操作"破坏"的情况（无 atomic/coherency 时这是合理假设）
- **预测式 latency**：在 issue cycle 一次性算 ready_at，不动态调整；与 Phase 1 一致

这些近似都是教学权衡——精度够展示对应现象，复杂度可控。

---

## 13. Phase 2 实施里程碑（高层）

详细任务由 writing-plans 阶段展开。

| 里程碑 | 交付 |
|---|---|
| **M1** | L1 cache + MSHR + 单 SM 集成（L2 mock 为固定 latency）；vector_add 仍 parity；新 stall token `MSHR_FULL` 工作 |
| **M2** | L2 cache（write-back + dirty eviction）+ 完整 L1↔L2 流水；smem_vs_l1_demo 第一版 |
| **M3** | HBM（channel + bank + row buffer + queue）+ 完整 L2↔HBM 衔接；row_buffer_demo + bw_saturation_demo 工作 |
| **M4** | Trace L1/L2/HBM 事件 + 5 个新分析指标 + HTML 5 个新节 + Perfetto 集成；l1_thrash_demo 工作 |
| **M5** | 4 章新讲义 + reference fixture 扩展 + 微基准断言松动 + README v2 |

Phase 1 的 13 处 deferred tech debt（hex `e` 已修；剩 12 项）继续推后。Phase 2 不引入这些 fix。

---

## 14. 设计协作记录

本文档由用户与 Claude（Opus 4.7, 1M context）通过 `superpowers:brainstorming` 流程逐节确认产出。所有关键决策均经用户显式确认（B/确认 回复）。

下一步：交由 `superpowers:writing-plans` 产出可执行的实施计划。
