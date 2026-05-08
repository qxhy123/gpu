# gpusim Phase 4 — Multi-SM + CTA 调度 + L2 跨 SM 共享 + TMA store 设计文档

**日期**：2026-05-08
**状态**：设计阶段（待实现）
**作者**：与 Claude 协同 brainstorm
**前置依赖**：Phase 1 完成（tag `phase1-complete`）+ Phase 2 完成（tag `phase2-shipped`）+ Phase 3 完成（tag `phase3-complete`，HEAD `1277337`）
**范围**：仅 Phase 4。Phase 5+ 仅作为愿景列出。

---

## 1. 愿景与 Phase 4 范围

### 1.1 项目背景

Phase 1-3 交付了**单 SM**、cycle-approximate、含完整 cache (L1/L2/HBM) + Tensor Core (sync mma + wgmma) + TMA load + mbarrier 的教学 GPU 模拟器。但所有 work 都跑在一个 SM 里——学生看不到 GPU 真正的"并行靠多 SM、共享靠 L2"的两级结构，也看不到 production matmul 的"TMA load → wgmma → TMA store"完整闭环。

Phase 4 把模拟器从"单 SM 教学板"扩展为**"N SM Device"**：multi-SM + CTA→SM 调度 + 全局共享 L2（含 cross-SM MSHR）+ TMA store + commit/wait_group。

### 1.2 Phase 4 一句话目标

> 在 Phase 1-3 的单 SM 之上加 **Device 顶层**：N 个 SM 共享 L2 + HBM，CTA scheduler（RR + greedy 可配）派发 CTA 到 SM，L2 加 MSHR 协调跨 SM 请求，补 Phase 3 deferred 的 TMA store（含 commit/wait_group）。3 个新 example + 3 章新讲义 让学生动手感受 multi-SM contention、scheduler 选择、L2 跨 SM 共享、production pipeline 的真实样貌。

### 1.3 路线图回顾

| Phase | 范围 | 状态 |
|---|---|---|
| Phase 1 | 单 SM、cycle-approximate、PTX 子集 | ✅ 已完成 |
| Phase 2 | L1/L2 cache + HBM channel/bank/row buffer | ✅ 已完成 |
| Phase 3 | Tensor Core + 6 精度 + wgmma + TMA-lite (load) | ✅ 已完成 |
| **Phase 4** | **Multi-SM + CTA scheduler + shared L2 + L2 MSHR + TMA store + 3 examples** | **本文档** |
| Phase 5 | Hopper Cluster (CGA) / distributed shared memory / atomics / multi-stream | 后续 |
| Phase 6 | 多 GPU、NVLink、NCCL collective | 后续 |

### 1.4 已锁定决策

| 维度 | 决策 |
|---|---|
| 范围 | multi-SM + CTA scheduler + shared L2 + L2 MSHR + TMA store + 3 examples + 3 tutorials（B 选项） |
| SM 数 | 默认 8，可配置（`DeviceConfig.n_sm`） |
| CTA 调度策略 | RR + greedy，默认 RR（`DeviceConfig.scheduler.cta_policy = "rr"｜"greedy"`） |
| L2 共享 | 单一全局 L2（4 MB，与 Phase 2 同），所有 SM 共享 |
| L2 MSHR | 默认 32 slot，可配置 |
| TMA store | 完整 async + commit/wait_group：`cp.async.bulk.tensor.2d.global.shared::cta` + `cp.async.bulk.commit_group` + `cp.async.bulk.wait_group N` |
| 顶层架构 | 新 `Device` class 持有 N 个 SM + 共享 L2 + HBM；`gpusim.run(...)` API 不变 |
| 新 stall token | `L2_MSHR_FULL` + `BULK_STORE_QUEUE_FULL` + `BULK_STORE_WAIT`（Phase 3 含 14 类 → 17 类） |
| 新 trace 事件 | `CtaDispatchEvent` + `L2MshrEvent` + `BulkStoreEvent`（Phase 3 含 13 类 → 16 类） |
| 现有 trace 事件加 `sm_id` 字段 | 默认 -1（兼容 Phase 1-3 single-SM 路径） |
| 新分析指标 | 6 个（per_sm_util / cta_to_sm / cta_dispatch_latency / l2_cross_sm_hit_rate / l2_mshr_pressure / bulk_store_async_overlap_ratio） |
| 新 HTML 节 | 4 节（§15–§18） |
| 新 examples | 3：multi_sm_scheduler / l2_sharing_demo / tma_store_matmul |
| 新 tutorials | 3：chapter 16-18 |

---

## 2. 架构总图与模块改动

### 2.1 数据流变化

**单 SM → multi-SM 拓扑**：
```
Device (NEW)
├─ DeviceConfig: n_sm, scheduler, cache (shared), hbm (shared)
├─ HBM (1 instance, shared)
├─ L2Cache (1 instance, shared, 加 MSHR)
├─ CtaScheduler (NEW: RR | greedy)
└─ SM × n_sm
   └─ L1Cache (per-SM, 接外部 L2 引用)
   └─ SubCore × 4
   └─ MbarrierPool (per-CTA)
   └─ TensorDescriptorPool (per-SM 不变)
```

**CTA 派发流程**：
```
Device.run(kernel, grid):
    cta_queue = enumerate(grid)
    while cta_queue or any_sm_busy:
        for sm in sms: sm.step_cycle(cycle)
        l2.tick(cycle)        # drain L2 MSHR
        if cta_queue and any_sm_has_capacity:
            cta = cta_queue.pop_front()
            sm = scheduler.pick(sms)   # RR | greedy
            sm.activate_cta(cta, cycle)
            recorder.cta_dispatch(cycle, cta_id, sm.id, queue_pos)
        cycle += 1
```

**TMA store 流程**：
```
warp issue cp.async.bulk.tensor.2d.global.shared::cta [gmem_desc], [smem_src]
  ├─ functional：smem → gmem 立即拷贝（功能正确）
  ├─ 入 BulkStoreQueue per warp-group：(commit_group_id, completion_at)
  └─ 后续 cp.async.bulk.commit_group 把 in-flight 划成 group
       cp.async.bulk.wait_group N 等到 ≤ N 个 group
```

### 2.2 关键不变量（继承自 Phase 1-3）

- **Functional 与 timing 分离**：所有 functional execution 立即在 numpy 层做对，timing 只管 cycle
- **Trace 是防火墙**：所有 multi-SM / scheduler / L2 MSHR / BulkStore 事件经 Recorder
- **Device 顶层但 SM 自治**：`SM.step_cycle(cycle)` 接口不变；Device 不侵入 SM 内部，只协调 CTA 派发 + 共享 L2 / HBM
- **L1 → L2 接口微调**：L1 仍调 `l2.fetch(line_addr, sm_id, now) → ready_at`；L2 实例共享
- **API 兼容**：`gpusim.run(...)` 签名不变；Phase 1-3 example 全部继续可跑（n_sm 默认 8 时这些 single-CTA example 只在 SM 0 上跑）

### 2.3 模块拓扑

```
gpusim/core/
├── device.py                NEW: Device + run + SM 协调
├── scheduler.py             MODIFY: + CtaScheduler (RR + greedy + factory)
├── cache/
│   ├── l2.py                MODIFY: + MSHR + origin_sm + tick
│   └── l2_mshr.py           NEW: L2 MSHR 池
├── tma_store.py             NEW: BulkStoreQueue + do_bulk_store_2d
├── sm.py                    MODIFY: 不再持有 L2/HBM；加 sm_id；外部 l2 引用
├── sub_core.py              MODIFY: + cp.async.bulk store / commit / wait_group
└── warp.py                  MODIFY: + bulk_store_pending_pc + 3 stall token

gpusim/frontend/parser.py    MODIFY: + cp.async.bulk store / commit / wait
gpusim/config/
├── schema.py                MODIFY: + DeviceConfig + CtaSchedulerConfig
└── default_hopper.yaml      MODIFY: 顶层重构，加 device 节

gpusim/trace/
├── events.py                MODIFY: + 3 events; sm_id 字段加到现有事件
├── recorder.py              MODIFY: + 3 recorder methods
└── writer.py                MODIFY: + 3 parquet writers

gpusim/analysis/metrics.py   MODIFY: + 6 metrics
gpusim/viz/                  MODIFY: + 4 HTML 节 + Perfetto per-SM swimlane
gpusim/api.py                MODIFY: + 3 events_df + device_metrics + device_summary()
```

### 2.4 Phase 1+2+3 carry-over

仅当 Phase 4 example 真撞到才修：
- **`0f` PTX float literal**（Phase 1 deferred）：multi_sm 例子若需 fp32 立即数零值，仍可用 `0` 整数 → 不阻塞
- **parser hex `e` digit**（Phase 1 deferred）：tensor 例子不太可能用 hex `e` 立即数 → 不阻塞
- **IPDOM 启发式**（Phase 1 deferred）：tma_store_matmul 有循环 + commit/wait，**可能撞到**——预留 budget
- **INSTR_COMPLETE event**（Phase 1 deferred）：与 Phase 4 无关
- **Phase 2 的 L2 容量边界 / MSHR 边界微基准**（Phase 2 边角）：Phase 4 microbench 自然覆盖

### 2.5 边界原则

1. Functional vs timing 分离 —— 继承
2. Trace 是防火墙 —— 继承
3. Tag-only / 数值 layer-bypass —— 继承
4. **Device 不持有 functional state** —— gmem/smem 仍是 SM/CTA 级；Device 只持有 timing-relevant 共享资源（L2/HBM/scheduler）
5. **API 兼容** —— `gpusim.run(...)` 签名不变；Phase 1-3 example 全部可继续运行

---

## 3. PTX 子集扩展 + IR 改动

Phase 4 仅新增 TMA store 相关的 3 条指令；其他改动在 sm/scheduler/L2 内部，不动 PTX 表面。

### 3.1 新增指令

| 指令 | 用途 |
|---|---|
| `cp.async.bulk.tensor.2d.global.shared::cta [gmem_desc], [smem_src]` | async 2D tensor store smem→gmem |
| `cp.async.bulk.commit_group` | 把当前 in-flight bulk store 划成新 group |
| `cp.async.bulk.wait_group N` | 阻塞至 in-flight bulk store group 数 ≤ N |

注：Phase 4 不新增标量/向量算术。

### 3.2 IR 改动

**无新 PtxType / 新 IR 节点**。复用 Phase 3 的 `TensorDescriptor`。

`gpusim.tma_desc` 伪指令（Phase 3）也用于构造 store side 的 gmem descriptor —— 同一 pool 同 handle 类型，store 时表示"目标 gmem 区域"。

### 3.3 Parser 改动

复用 Phase 3 已建好的 `cp.async.bulk.tensor.*` 通配分支。`_parse_operands` 改动：

```python
if op.startswith("cp.async.bulk.tensor."):
    # 现有：3 个 [bracketed] 操作数（load）
    # 新增：检查 op 是否含 "mbarrier" → load (3 args)；否则 store (2 args)
    n_args = 3 if "mbarrier" in op else 2
    srcs: list = []
    for _ in range(n_args):
        self.eat("LBRACK")
        addr = self._parse_operand(PtxType.u64)
        self.eat("RBRACK")
        srcs.append(addr)
        if not self.accept("COMMA"):
            break
    return [], srcs

if op == "cp.async.bulk.commit_group":
    return [], []
if op == "cp.async.bulk.wait_group":
    n_imm = self._parse_operand(PtxType.s32)
    return [], [n_imm]
```

注意：`cp.async.bulk.commit_group` 与 `wgmma.commit_group.sync.aligned` 是**不同**的 commit_group，分别由 `BulkStoreQueue` 和 `WgmmaQueue` 管理，互不干扰。

### 3.4 FUSet.classify 改动

```python
if op.startswith("cp.async.bulk.commit_group") or op.startswith("cp.async.bulk.wait_group"):
    return FUKind.LSU
```

---

## 4. Multi-SM + Device 详细设计

### 4.1 DeviceConfig + 配置 yaml 重构

**新顶层 `DeviceConfig`**：
```python
@dataclass
class DeviceConfig:
    n_sm: int = 8
    sm: SMConfig = field(default_factory=SMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    hbm: HBMConfig = field(default_factory=HBMConfig)
    scheduler: CtaSchedulerConfig = field(default_factory=CtaSchedulerConfig)


@dataclass
class CtaSchedulerConfig:
    cta_policy: str = "rr"   # "rr" | "greedy"
```

**`SMConfig` 减负**：移除 `cache` 和 `hbm` 字段（搬到 Device 顶层）。`SMConfig` 仅保留 sub_cores / regs / smem / fu / tensor_core 等 SM-内部配置。

**default_hopper.yaml 顶层重构**：
```yaml
device:
  n_sm: 8
  scheduler:
    cta_policy: rr

sm:
  sub_cores: 4
  warps_per_sm: 64
  threads_per_sm: 2048
  ... (其他 SM 字段不变)
  fu: { ... }
  tensor_core: { ... }

cache: { ... }    # 顶层（Phase 2 时本来在 sm 下）
hbm: { ... }      # 顶层（Phase 2 时本来在 sm 下）
```

`gpusim/config/loader.py` 调整为 device-first 解析；遗留 yaml（cache/hbm 在 sm 下）做向后兼容：若顶层无 `device:` 节但有 `sm:`，按 Phase 2 模式回退到单 SM 模式（`n_sm=1`）。

### 4.2 Device class

`gpusim/core/device.py`（新）：

```python
@dataclass
class DeviceRunResult:
    cycles: int
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    occupancy: dict | None = None


class Device:
    def __init__(self, cfg: DeviceConfig, recorder: object | None = None):
        self.cfg = cfg
        self.recorder = recorder

    def run(self, kernel, grid, block, params, regs_per_thread=16, smem_per_cta=0):
        # 1. allocate 共享 gmem + 共享 HBM + 共享 L2 (with MSHR)
        gmem = GlobalMemory()
        hbm = HBM(self.cfg.hbm, recorder=self.recorder)
        l2 = L2Cache(self.cfg.cache, hbm, recorder=self.recorder)

        # 2. 构造 N 个 SM；每个 SM 接受同一个 l2 引用
        sms = [SM(self.cfg.sm, sm_id=i, recorder=self.recorder, l2=l2, hbm=hbm)
                for i in range(self.cfg.n_sm)]

        # 3. 计算 occupancy（per-SM）
        occ = compute_occupancy(self.cfg.sm, threads_per_cta, regs_per_thread, smem_per_cta)

        # 4. CTA scheduler
        cta_queue = list(_enumerate_grid(grid))
        scheduler = make_cta_scheduler(self.cfg.scheduler.cta_policy)

        # 5. main loop
        cycle = 0
        while cta_queue or any(sm.has_active_warps() for sm in sms):
            for sm in sms:
                sm.step_cycle(cycle, kernel, gmem, params)
            l2.tick(cycle)
            while cta_queue:
                target_sm = scheduler.pick(sms, occ)
                if target_sm is None: break
                cta = cta_queue.pop(0)
                target_sm.activate_cta(cta, cycle)
                if self.recorder:
                    self.recorder.cta_dispatch(cycle=cycle, cta_id=cta.cid,
                                                 sm_id=target_sm.sm_id, queue_position=...)
            cycle += 1
            if cycle > 100_000_000: raise RuntimeError("runaway")

        return DeviceRunResult(cycles=cycle, outputs=..., occupancy=...)
```

**SM 改动**：
- `SM.__init__` 加 `sm_id: int`、`l2: L2Cache`、`hbm: HBM` 参数
- `SM.run(...)` 保留作单 SM 测试入口（内部构造 1-SM Device）
- 加 `SM.step_cycle(cycle, ...)` —— 一次 cycle 推进
- 加 `SM.activate_cta(cta, cycle)` + `SM.has_active_warps()` + `SM.can_admit_cta(occ)` + `SM.active_warp_count()` 接口

### 4.3 CtaScheduler

`gpusim/core/scheduler.py` 新增：

```python
def make_cta_scheduler(policy: str) -> "CtaScheduler":
    if policy == "rr": return RRCtaScheduler()
    if policy == "greedy": return GreedyCtaScheduler()
    raise ValueError(f"unknown cta_policy {policy!r}")


class RRCtaScheduler:
    def __init__(self):
        self._next = 0

    def pick(self, sms, occ) -> "SM | None":
        n = len(sms)
        for _ in range(n):
            sm = sms[self._next]
            self._next = (self._next + 1) % n
            if sm.can_admit_cta(occ):
                return sm
        return None


class GreedyCtaScheduler:
    def pick(self, sms, occ) -> "SM | None":
        eligible = [sm for sm in sms if sm.can_admit_cta(occ)]
        if not eligible:
            return None
        return min(eligible, key=lambda sm: sm.active_warp_count())
```

### 4.4 L2 加 MSHR

`gpusim/core/cache/l2_mshr.py`（新）：

```python
@dataclass
class L2MshrEntry:
    line_addr: int
    arrival_cycle: int
    completion_at: int
    waiters: list[tuple[int, int]] = field(default_factory=list)


class L2Mshr:
    def __init__(self, n_slots: int):
        self.n_slots = n_slots
        self._table: dict[int, L2MshrEntry] = {}

    def lookup_or_alloc(self, *, line_addr: int, sm_id: int, now: int) -> tuple[bool, L2MshrEntry | None]:
        if line_addr in self._table:
            return (False, self._table[line_addr])
        if len(self._table) >= self.n_slots:
            return (False, None)
        entry = L2MshrEntry(line_addr=line_addr, arrival_cycle=now, completion_at=-1)
        self._table[line_addr] = entry
        return (True, entry)

    def release(self, line_addr: int) -> None:
        self._table.pop(line_addr, None)
```

`L2Cache.fetch(line_addr, sm_id, now)` 改：
```python
def fetch(self, *, line_addr: int, sm_id: int, now: int) -> int:
    line = self._sets[set_idx].find(tag)
    if line is not None:                            # HIT
        if recorder:
            recorder.l2_access(..., origin_sm=line.origin_sm, hit_sm=sm_id)
        return now + l2_hit_latency

    allocated, entry = self.mshr.lookup_or_alloc(line_addr=line_addr, sm_id=sm_id, now=now)
    if entry is None:
        return -1   # signal MSHR_FULL → L1 stalls
    if not allocated:                                # MERGE
        if recorder: recorder.l2_mshr(kind="MERGE", ...)
        return entry.completion_at
    hbm_done = self._hbm.request(line_addr, now)
    entry.completion_at = hbm_done + l2_miss_install_latency
    if recorder: recorder.l2_mshr(kind="ALLOC", ...)
    line.origin_sm = sm_id   # install
    return entry.completion_at

def tick(self, now: int) -> None:
    done = [addr for addr, e in self._mshr._table.items() if e.completion_at <= now]
    for addr in done:
        if recorder: recorder.l2_mshr(kind="RELEASE", ...)
        self._mshr.release(addr)
```

`CacheLine` 加 `origin_sm: int = -1` 字段。

L1 `fetch` 调 L2 时若返回 -1 → 触发 `L2_MSHR_FULL` stall。

### 4.5 新 stall token

| Token | 触发 |
|---|---|
| `L2_MSHR_FULL` | L1 调 L2 fetch 时 L2 MSHR 满 |
| `BULK_STORE_QUEUE_FULL` | 发起 cp.async.bulk store 时队列满 |
| `BULK_STORE_WAIT` | cp.async.bulk.wait_group 等待中 |

总 stall 类数：14 → **17**。

---

## 5. TMA Store 详细设计

### 5.1 cp.async.bulk.tensor.2d.global.shared::cta 执行语义

**Functional**：
1. 解 gmem descriptor（来自 `gpusim.tma_desc` allocated handle）
2. 总字节 = `dim_x * dim_y * elem_bytes`
3. `do_bulk_store_2d(gmem, smem, cta_id, smem_src, desc)` 立即拷贝
4. 立即完成（数值正确）

**Timing**：
1. cache_lines = ceil(tile_bytes / 128)
2. **Bypass L1/L2** —— 与 TMA load 一致
3. 每 line 调 `hbm.write_request(line_addr, now)`；max(...) 作为 `completion_at`
4. 入 `BulkStoreQueue`（per warp-group）

`do_bulk_store_2d`（在 `gpusim/core/tma_store.py`）与 Phase 3 的 `do_bulk_copy_2d` 镜像，仅方向反：
```python
def do_bulk_store_2d(*, gmem, smem, cta_id: int, smem_src: int,
                      desc: TmaDescriptor) -> int:
    bytes_per_row = desc.dim_x * desc.elem_bytes
    dst_stride_bytes = desc.stride_y * desc.elem_bytes
    smem_buf = smem._cta[cta_id]
    for row in range(desc.dim_y):
        gmem_addr = desc.gmem_base + row * dst_stride_bytes
        src_off = smem_src + row * bytes_per_row
        chunk = bytes(smem_buf[src_off:src_off + bytes_per_row])
        gmem.store_bytes(gmem_addr, chunk)
    return desc.dim_y * bytes_per_row
```

### 5.2 BulkStoreQueue + commit_group / wait_group N

```python
@dataclass
class InflightBulkStore:
    issued_at: int
    completion_at: int
    bytes_total: int
    commit_group_id: int = -1


@dataclass
class BulkStoreQueue:
    capacity: int = 16
    in_flight: list[InflightBulkStore] = field(default_factory=list)
    committed_groups: list[int] = field(default_factory=list)
    next_group_id: int = 0

    def try_push(self, f: InflightBulkStore) -> bool: ...
    def commit_group(self) -> int: ...
    def must_wait(self, target_n: int) -> bool: ...
    def drain_completed_groups(self, now: int) -> list[int]: ...
```

语义与 `WgmmaQueue` 完全镜像。区别：
- 不持有 dst_regs（store 不写寄存器）
- per warp-group（与 wgmma 一致），让 4 warp 协同发

### 5.3 SubCore _issue 路由

```python
if op.startswith("cp.async.bulk.tensor.") and "global.shared" in op:
    # store 形式
    # _is_ready: 4 warp 都到才 issue (BARRIER 状态), QUEUE_FULL 时 BULK_STORE_QUEUE_FULL
    # 由 Device 主循环（或 SubCore 协调）发起 do_bulk_store_2d
    ...

if op == "cp.async.bulk.commit_group":
    # self.bulk_store_queues[w.warp_group_id].commit_group()
    # advance PC + record event
    ...

if op == "cp.async.bulk.wait_group":
    # _is_ready 已经处理 BULK_STORE_WAIT；这里 advance PC + record
    ...
```

### 5.4 _is_ready 改动

```python
if op.startswith("cp.async.bulk.tensor.") and "global.shared" in op:
    if self.bulk_store_queues is not None:
        q = self.bulk_store_queues.setdefault(
            w.warp_group_id, BulkStoreQueue(capacity=self.cfg.tensor_core.bulk_store_queue_capacity))
        if len(q.in_flight) >= q.capacity:
            return False, StallReason.BULK_STORE_QUEUE_FULL
    w.bulk_store_pending_pc = pc
    return False, StallReason.BARRIER

if op == "cp.async.bulk.wait_group":
    q = self.bulk_store_queues.get(w.warp_group_id)
    if q is None:
        return True, StallReason.ISSUED
    target_n = int(instr.src[0].value)
    drained = q.drain_completed_groups(now=now)
    if q.must_wait(target_n):
        return False, StallReason.BULK_STORE_WAIT
    return True, StallReason.ISSUED
```

`Warp` 加字段 `bulk_store_pending_pc: int = -1`，与 `wgmma_pending_pc` 对称。

### 5.5 SM warp-group 协调（与 wgmma 同套）

Device 主循环（或 SM.step_cycle）扫描 warp-group：
- 若 4 warp 都有 `bulk_store_pending_pc >= 0` 且 PC 一致 → 发起 store
- 由 warp 0 代理执行 `do_bulk_store_2d`（functional）
- 入 `BulkStoreQueue` 并记录 `BulkStoreEvent(kind="ISSUE", ...)`
- 4 warp 推进 PC，重置 pending

### 5.6 默认参数 / 配置

```python
@dataclass
class TensorCoreConfig:
    # ... 现有 5 字段 ...
    bulk_store_latency_per_line: int = 4   # NEW
    bulk_store_queue_capacity: int = 16    # NEW
```

---

## 6. Trace + 分析 + 可视化

### 6.1 完整事件清单（Phase 1+2+3+4）

| 类别 | 事件 | 频率 |
|---|---|---|
| Phase 1 (8) | WARP_STATE, INSTR_ISSUE, SMEM_ACCESS, GMEM_ACCESS, DIV_PUSH/POP, BAR_REACH/RELEASE, CTA_LAUNCH/RETIRE | 高（RLE） |
| Phase 2 (3) | L1_ACCESS, L2_ACCESS, HBM_ACCESS | 中-低 |
| Phase 3 (4) | MmaEvent, WgmmaEvent, TmaEvent, MbarrierEvent | 中-低 |
| **Phase 4 (3)** | **CtaDispatchEvent, L2MshrEvent, BulkStoreEvent** | 中-低 |

所有现有事件加 `sm_id: int = -1` 字段（默认 -1，单 SM 兼容路径下保留）。

### 6.2 Phase 4 新事件 schema

```python
@dataclass(frozen=True)
class CtaDispatchEvent:
    cycle: int
    cta_id: int
    sm_id: int
    queue_position: int
    active_warps_at_dispatch: int


@dataclass(frozen=True)
class L2MshrEvent:
    kind: str          # "ALLOC" | "MERGE" | "RELEASE" | "FULL"
    cycle: int
    line_addr: int
    sm_id: int
    n_waiters: int = 0


@dataclass(frozen=True)
class BulkStoreEvent:
    kind: str          # "ISSUE" | "COMMIT_GROUP" | "WAIT_GROUP" | "DRAIN"
    cycle: int
    warp_group_id: int
    sm_id: int
    pc: int
    smem_src: int = 0
    gmem_base: int = 0
    bytes_total: int = 0
    completion_at: int = -1
    commit_group_id: int = -1
    wait_n: int = -1
```

### 6.3 现有事件 schema 变更

加 `sm_id: int = -1` 字段：
- WarpStateSegment, InstrIssueEvent, SmemEvent, GmemEvent, DivEvent, BarEvent
- L1Event, L2Event, HBMEvent
- MmaEvent, WgmmaEvent, TmaEvent, MbarrierEvent

向后兼容：默认 -1 表示 single-SM 模式（Phase 1-3 路径）。

`CtaEvent` 加 `sm_id` 字段（替代默认 0）。

### 6.4 新增分析指标（6 个）

| 函数 | 输出 | 教学用途 |
|---|---|---|
| `per_sm_utilization(warp_state_df, total_cycles, n_sm)` | DataFrame[n_sm] | per-SM 忙碌 % |
| `cta_to_sm_mapping(dispatch_df)` | DataFrame[cta_id, sm_id, dispatch_cycle] | 调度记录表 |
| `cta_dispatch_latency(dispatch_df, cta_launch_df)` | Series（cycle 直方图） | CTA 等待入 SM 的 cycle 分布 |
| `l2_cross_sm_hit_rate(l2_events_df)` | scalar 0..1 | L2 hit 中 origin_sm ≠ hit_sm 的比例 |
| `l2_mshr_pressure(l2_mshr_events_df, total_cycles)` | Series | 每 cycle in-flight L2 MSHR 数 |
| `bulk_store_async_overlap_ratio(bulk_store_df, warp_state_df)` | scalar 0..1 | TMA store in-flight 时 warp 是否在干活 |

### 6.5 HTML 报告新增节（4 节）

| 节 | 内容 |
|---|---|
| **§15 Per-SM utilization** | n_sm bar chart + 文字说明热点 SM |
| **§16 CTA → SM mapping + dispatch latency** | 表格 + Plotly stacked bar by SM |
| **§17 L2 cross-SM hit + MSHR pressure** | cross-SM hit % + L2 MSHR pressure 时序图 |
| **§18 BulkStore timeline** | Plotly Gantt：每 warp-group BulkStore in-flight + commit + wait |

### 6.6 Result API 扩展

```python
@dataclass
class Result:
    # ... Phase 1-3 fields ...

    @property
    def cta_dispatch_events_df(self) -> pd.DataFrame: ...
    @property
    def l2_mshr_events_df(self) -> pd.DataFrame: ...
    @property
    def bulk_store_events_df(self) -> pd.DataFrame: ...

    @property
    def device_metrics(self) -> dict: ...

    def device_summary(self) -> str: ...
```

`Result.summary()` 在 Phase 4 模式下追加 `device_summary()` 行。

### 6.7 Perfetto 集成

| 事件 | Perfetto track | 视觉 |
|---|---|---|
| `CtaDispatchEvent` | per-SM "CTA" track | 蓝色 instant + cta_id 标签 |
| `L2MshrEvent("ALLOC")` | global "L2 MSHR" track | 绿色 instant |
| `L2MshrEvent("MERGE")` | global "L2 MSHR" track | 黄色 instant |
| `L2MshrEvent("FULL")` | global "L2 MSHR" track | 红色 instant |
| `BulkStoreEvent("ISSUE")` | per-warp-group "TMA Store" track | 紫色 X event(duration) |
| `BulkStoreEvent("WAIT_GROUP")` | per-warp-group "TMA Store" track | 灰色 instant |

所有 per-SM events（含 Phase 1-3 既有事件）现在 pid 形如 `SM{n}` 而非 single track。Phase 1-3 example 路径下 sm_id=0，pid=`SM0`，与 multi-SM 一致。

### 6.8 Parquet 落盘

新增 3 个 parquet：`cta_dispatch.parquet`、`l2_mshr.parquet`、`bulk_store.parquet`。

---

## 7. 测试策略

### 7.1 单元测试

| 模块 | 关键测试 |
|---|---|
| `core/scheduler` (CTA) | RR 顺序、greedy 选最少 active warp、capacity 满返回 None |
| `core/cache/l2_mshr` | alloc / merge / release / 满返回 None；cross-SM 同 line 触发 MERGE |
| `core/cache/l2` | `origin_sm` 字段；cross-SM hit 元数据；MSHR full → fetch 返回 -1 |
| `core/tma_store` | BulkStoreQueue lifecycle；`do_bulk_store_2d` 字节正确 |
| `core/device` | `Device.run` 单 CTA 退化到单 SM 行为；多 CTA 派遣到多 SM；超出 occupancy 等待 |
| `core/sm` | `SM.activate_cta + step_cycle`；外部 L2 注入 |
| `frontend/parser` | cp.async.bulk store / commit_group / wait_group 解析 |
| `analysis/metrics` | 6 个新指标 fixture |
| `viz/html_report` | 4 个新节正确插入 |
| `config/loader` | 顶层 device 节解析；遗留 yaml 回退兼容 |
| `trace/recorder` | 3 个新事件 + 现有事件加 sm_id 字段 |

### 7.2 Functional Parity

3 个新 example 各有 numpy 参考实现：
- `multi_sm_scheduler`：grid=(8 或 16,1,1)，每 CTA 写不同段，rtol=0
- `l2_sharing_demo`：grid=(8,1,1)，所有 CTA 都读同一段 read-only 输入，rtol=0
- `tma_store_matmul`：grid=(2,1,1) 或 (4,1,1) 的 matmul，rtol=1e-2

Phase 1-3 example 全部继续通过。

### 7.3 Reference Fixture（真机对照）

`tests/reference/data/` 加 3 个 stub。容忍度：
- `per_sm_utilization` ±15%
- `l2_cross_sm_hit_rate` ±10%
- `l2_mshr_pressure_peak` ±20%

`gen_reference.py` `SUPPORTED_KERNELS` 加 3 项。

### 7.4 微基准

新增 `tests/microbench/test_phase4_facts.py`：

```
- 8 个独立 CTA 跑 vector_add 在 8 SM 下 cycles ≤ 1.5 × 单 CTA cycles（≥ 5× 加速）
- 同一只读输入被 8 CTA 读：l2_cross_sm_hit_rate ≥ 0.6
- 高强度 + 8 SM：L2 MSHR pressure peak ≥ 16
- multi_sm_scheduler 在 irregular workload 下：greedy cycles < RR cycles × 0.85
- tma_store_matmul：bulk_store_async_overlap_ratio ≥ 0.3
```

### 7.5 兼容性测试

新增 `tests/parity/test_phase1_3_examples_unchanged.py`：
- Phase 1-3 的 10 个 example 在新 Device 路径下数值结果不变
- cycles 浮动 ≤ 5%
- 既有 metrics 不变

### 7.6 Memory budget 测试

新增 `tests/microbench/test_phase4_runtime.py`（@pytest.mark.slow）：
- 默认 8 SM 下 multi_sm_scheduler < 30 秒
- tma_store_matmul（n_sm=8, 4 CTA）< 60 秒

---

## 8. 项目结构改动

### 8.1 目录变化

```
gpusim/core/
├── device.py                NEW
├── tma_store.py             NEW
├── cache/
│   ├── l2.py                MODIFY
│   └── l2_mshr.py           NEW
├── scheduler.py             MODIFY
├── sm.py                    MODIFY
├── sub_core.py              MODIFY
└── warp.py                  MODIFY

gpusim/frontend/parser.py    MODIFY
gpusim/config/{schema.py, default_hopper.yaml}    MODIFY
gpusim/trace/{events.py, recorder.py, writer.py}  MODIFY
gpusim/analysis/metrics.py   MODIFY
gpusim/viz/                  MODIFY
gpusim/api.py                MODIFY

tests/unit/core/test_{device,cta_scheduler,tma_store}.py     NEW
tests/unit/cache/test_l2_mshr.py                              NEW
tests/parity/test_{multi_sm_scheduler,l2_sharing_demo,tma_store_matmul,phase1_3_examples_unchanged}.py    NEW
tests/microbench/test_{phase4_facts,phase4_runtime}.py        NEW
tests/reference/data/{multi_sm_scheduler,l2_sharing_demo,tma_store_matmul}.ref.json    NEW
```

### 8.2 配置 yaml 迁移示例

```yaml
# default_hopper.yaml v4
device:
  n_sm: 8
  scheduler:
    cta_policy: rr

sm:
  sub_cores: 4
  warps_per_sm: 64
  threads_per_sm: 2048
  max_ctas_per_sm: 32
  regs_per_sm: 65536
  smem_per_sm_bytes: 49152
  smem_banks: 32
  scheduler:
    policy: gto
  regfile: { banks: 4, regs_per_subcore: 16384 }
  fu: { ... }
  tensor_core:
    tc_mma_latency: 8
    tc_mma_occupancy: 1
    tc_wgmma_latency: 32
    tc_wgmma_occupancy: 4
    wgmma_queue_capacity: 16
    bulk_store_queue_capacity: 16    # NEW
    bulk_store_latency_per_line: 4   # NEW

cache:
  l1_size_bytes: 131072
  l1_ways: 4
  l1_line_bytes: 128
  l1_hit_latency: 25
  l1_miss_check_latency: 5
  mshr_slots: 16
  l2_size_bytes: 4194304
  l2_ways: 16
  l2_line_bytes: 128
  l2_hit_latency: 200
  l2_miss_install_latency: 10
  l2_mshr_slots: 32      # NEW

hbm:
  channels: 8
  banks_per_channel: 16
  row_size_bytes: 4096
  row_hit_latency: 10
  row_miss_latency: 30
```

### 8.3 依赖

无新依赖。

---

## 9. 教学示例与讲义

### 9.1 3 个新 example

| # | Example | 教学意图 | grid |
|---|---|---|---|
| 1 | **multi_sm_scheduler** | 多 SM 并发 + RR vs greedy 调度对比 | (16,1,1)（CTA 间 work 不均匀） |
| 2 | **l2_sharing_demo** | 多 CTA 共享只读输入 → cross-SM L2 hit | (8,1,1) |
| 3 | **tma_store_matmul** | TMA load + wgmma + TMA store 完整 production matmul | (2-4,1,1)，每 CTA 64×128 tile |

multi_sm_scheduler 用单一 PTX；run.py 用两份 config 切 `cta_policy` 比对。

每个目录：`{kernel*.ptx}` + `reference.py` + `run.py` + `README.md` + `__init__.py`。

### 9.2 3 章新讲义（chapters 16-18）

| # | 标题 | 关联 example |
|---|---|---|
| 16 | Multi-SM 与 CTA 调度：从单 SM 到 N SM 的并行 | multi_sm_scheduler |
| 17 | L2 共享与 cross-SM coalescing | l2_sharing_demo |
| 18 | TMA store 与端到端生产 matmul pipeline | tma_store_matmul |

每章固定栏目：**看模拟器** / **改一改** / **真机对照**。

---

## 10. 与 Phase 1+2+3 兼容性

### 10.1 不会破坏的部分

| 维度 | 状态 |
|---|---|
| `gpusim.run(...)` 函数签名 | 不变 |
| Phase 1-3 example PTX | 不动 |
| Phase 1-3 parity 测试 | 全部继续通过（默认 8 SM，单 CTA example 落在 SM 0） |
| `gpusim.cli` 命令集 | 不变 |
| Result 旧字段 | 不变 |
| HTML 报告既有 §1–§14 | 位置 + 内容不变 |
| Perfetto trace 既有 track | 行为兼容（per-SM swimlane 现在是 SM0..SM7） |
| Stall token 既有 14 类 | 不变 |

### 10.2 会变的部分

| 维度 | 变化 |
|---|---|
| Cycle 数（Phase 1-3 example） | 几乎不变（多 SM 时只用 SM 0） |
| Stall 直方图 | 多 3 类 |
| HTML 报告 | 多 4 节（§15–§18） |
| Perfetto | 新 track（每 SM 一行） |
| Trace parquet | 多 3 文件 |
| Result | 多 3 properties + `device_metrics` + `device_summary()` |
| 配置 yaml schema | 顶层重构（cache/hbm 上移），增加 `device:` 节；遗留 yaml 经 loader 兼容回退 |
| `SMConfig` | 移除 `cache` + `hbm` 字段（搬到 Device 顶层）；代码内部全经 loader 走 yaml 路径 |
| `Instr.type` | 不变（Phase 3 已 Optional） |
| Trace 既有事件 | 加 `sm_id: int = -1` 字段（默认值兼容） |

### 10.3 配置兼容回退

`gpusim/config/loader.py` 检测顶层 `device:` 节是否存在：
- 存在 → Phase 4 路径，按新 schema 解析
- 不存在 → Phase 1-3 路径（单 SM），把现有 `sm.cache` / `sm.hbm` 上拉到顶层，构造 `DeviceConfig(n_sm=1, ...)`

---

## 11. 显式不在范围内（Phase 4）

记录以避免误解：

- **多 SM 同步原语**：`atom.*`、`red.*`（Phase 5+）
- **Hopper Cluster (CGA) / distributed shared memory**：`shared::cluster` 在 Phase 3 PTX 名义已支持，但 cluster 内多 SM 真共享 smem 的语义未实现
- **Multi-stream / 多 kernel 并发**：仍是单 kernel
- **多 GPU、NVLink、NCCL**：Phase 5+
- **L2 partition / banking**：单一全局 L2，不分 slice
- **CTA priority / preemption**：CTA 一旦上 SM 就跑到完
- **TMA store 反向 mbarrier**：仅 commit/wait_group
- **TMA descriptor 真机字节级 encoding**：继续 `gpusim.tma_desc` 伪指令
- **Persistent kernel**：每个 grid 都是 fresh launch
- **Compiler-level passes**：模拟器 PTX 走原样

---

## 12. 已知近似与简化

- **CTA scheduler 简化**：RR + greedy 是教学性两端；真机调度涉及 priority、stream、CGA、dependency
- **L2 single shared instance**：真机 H100 是 60 MB 拆 12 slice 的复杂结构。本模拟器是单 4 MB 全局 L2 + MSHR 32 slot
- **L2 MSHR 简化**：仅按 line_addr coalesce；真机 MSHR 还有 priority、age、address-conflict 处理
- **TMA store 简化**：functional 立即写 gmem；timing 上仅 `latency = max(per-line HBM serve)` 估算
- **per-SM 资源限制不变**：Phase 1 设的 max_ctas_per_sm = 32, threads = 2048, regs = 65536；Phase 4 不动
- **不建模 SM 异步退役**：CTA 退役在 main loop 同步处理，所有 SM 在同一 cycle tick

---

## 13. Phase 4 实施里程碑

| 里程碑 | 交付 |
|---|---|
| **M1** | 配置 schema 重构：DeviceConfig + 顶层 yaml + loader 双路径（device/legacy）+ SMConfig 减负 + 单元测试。无运行时行为变化 |
| **M2** | Device class + SM 重构（外部 L2 注入）+ CtaScheduler（RR + greedy）+ multi_sm_scheduler example + 微基准 |
| **M3** | L2 MSHR：l2_mshr.py + L2.fetch 改造 + origin_sm 字段 + cross-SM 元数据 + L2_MSHR_FULL stall + l2_sharing_demo example |
| **M4** | TMA store：tma_store.py + BulkStoreQueue + parser 扩展 + SubCore + Device 协调 + 2 个新 stall token + tma_store_matmul example |
| **M5** | Trace + 分析 + viz + 收尾：3 trace events + sm_id 注入既有事件 + 6 分析指标 + 4 HTML 节 + Perfetto per-SM swimlane + Result API + 3 章新讲义 + reference fixture 扩展 + Phase 4 微基准 + Phase 1-3 兼容性测试 + README v4 + tag `phase4-complete` |

预估总任务数：**32-37**（与 Phase 3 的 33 接近）。

每 milestone 后打 git tag (`M{1..5}-phase4-complete`) 作为 review checkpoint。

---

## 14. 设计协作记录

本文档由用户与 Claude（Opus 4.7, 1M context）通过 `superpowers:brainstorming` 流程逐节确认产出。所有关键决策均经用户显式确认（A/B/C/D 选择或"ok"回复）。

下一步：交由 `superpowers:writing-plans` 产出可执行的实施计划。
