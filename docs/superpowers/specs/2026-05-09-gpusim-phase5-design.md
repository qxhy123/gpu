# gpusim Phase 5 — Hopper Cluster (CGA) + Distributed Shared Memory 设计文档

**日期**：2026-05-09
**状态**：设计阶段（待实现）
**作者**：与 Claude 协同 brainstorm
**前置依赖**：Phase 1-4 完成（tag `phase4-complete`，HEAD `e5270b6`）
**范围**：仅 Phase 5。Phase 6+ 仅作为愿景列出。

---

## 1. 愿景与 Phase 5 范围

### 1.1 项目背景

Phase 1-4 交付了 multi-SM、共享 L2、完整 cache 层级、Tensor Core (sync mma + wgmma)、TMA load + store、mbarrier、CTA scheduler。Phase 3 PTX 已名义解析 `shared::cluster` 命名空间，Phase 4 仍把它当作 `shared::cta`——cluster 是模拟器的技术债，没有真正的 cluster 拓扑/语义。

Phase 5 完成 Hopper "cluster + dsmem" 闭环：`DeviceConfig.cluster_size` 配置 cluster 大小（默认 1 = Phase 4 行为）；`mapa.shared::cluster` + `ld/st.shared::cluster` 让 CTA 可访问同 cluster 远程 CTA's smem；`barrier.cluster.{arrive,wait}` 提供 two-phase async 同步；mbarrier + TMA load 扩到 `shared::cluster`，让一个 CTA 可代理 fetch 后写入远程 cluster 成员的 smem 池；3 个新 example + 3 章新讲义让学生学完后能读 cutlass Hopper persistent matmul 的核心结构。

### 1.2 Phase 5 一句话目标

> 在 Phase 1-4 的 multi-SM Device 之上加 cluster 拓扑：cluster_size 个 CTA 协同驻留、跨 SM 共享 distributed shared memory、cluster-wide barrier + mbarrier 同步、cluster TMA load 写到 remote CTA's smem。3 个新 example + 3 章讲义补完 Hopper 教学闭环。

### 1.3 路线图回顾

| Phase | 范围 | 状态 |
|---|---|---|
| Phase 1 | 单 SM、cache-less、PTX 子集 | ✅ |
| Phase 2 | L1/L2 + HBM | ✅ |
| Phase 3 | Tensor Core + wgmma + TMA load + mbarrier | ✅ |
| Phase 4 | Multi-SM + Device + L2 MSHR + TMA store | ✅ |
| **Phase 5** | **Hopper Cluster (CGA) + dsmem + cluster mbarrier + cluster TMA load** | **本文档** |
| Phase 6 | Atomics (`atom.*`/`red.*` 全空间) + cooperative epilogue (cluster TMA store) | 后续 |
| Phase 7 | Multi-stream / 多 kernel 并发 | 后续 |
| Phase 8 | Multi-GPU + NVLink + NCCL | 后续 |

### 1.4 已锁定决策

| 维度 | 决策 |
|---|---|
| 范围 | Cluster + dsmem core（A 选项）；无 atomics/multi-stream/cluster TMA store |
| Cluster size | 可配置，默认 2，必须整除 grid_size（`DeviceConfig.cluster_size`） |
| dsmem 实现 | 指针编码：`mapa.shared::cluster` 返回 `(rank << 24) \| offset` |
| Cluster barrier | Two-phase async：`barrier.cluster.arrive` + `barrier.cluster.wait` |
| Cluster mbarrier | `mbarrier.{init,arrive,try_wait}.shared::cluster`；mbarrier 指针可 cluster 编码 |
| Cluster TMA load | `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` smem_dst 解码 → 写到 remote CTA's smem + 在 remote mbarrier arrive_tx |
| Cluster 调度 | Device 层批派（B 选项）：scheduler 加 peek/commit，Device 一次试派整 cluster |
| 顶层架构 | 复用 Phase 4 Device + SM；不新增类（仅新增 ClusterBarrierPool） |
| 新 stall token | `CLUSTER_BARRIER_WAIT`（17 → 18 类） |
| 新 trace 事件 | `ClusterDispatchEvent` + `ClusterBarrierEvent`（16 → 18 类） |
| 新分析指标 | 3：cluster_dispatch_latency / cluster_barrier_wait_distribution / dsmem_remote_access_rate |
| 新 HTML 节 | 2：§19 cluster timeline、§20 dsmem traffic |
| 新 examples | 3：cluster_basic / cluster_matmul_dsmem / cluster_tma_pipeline |
| 新 tutorials | 3：chapters 19-21 |
| Phase 1-4 兼容 | cluster_size=1 默认 → Phase 4 行为 byte-for-byte 不变 |

---

## 2. 架构总图与模块改动

### 2.1 数据流变化

**Cluster 拓扑**（cluster_size = 2 时）：
```
Device (Phase 4 unchanged)
├─ DeviceConfig: + cluster_size (default 1)
├─ HBM, L2 (shared, unchanged)
├─ CtaScheduler (unchanged: RR | greedy)
└─ SM × n_sm
   └─ each SM hosts CTAs from various clusters

Cluster mapping:
  cluster_id  = cta_id // cluster_size
  cluster_rank= cta_id %  cluster_size
  Cluster K's CTAs:  cta_id ∈ [K*cluster_size, (K+1)*cluster_size)
  Each Cluster CTA lives on a different SM.
```

**Cluster 派发（Device.run）**：
```
_try_dispatch():
    while clusters_remaining:
        next_cluster = clusters[ptr]               # K CTAs
        target_sms = scheduler.peek(sms, occ, k=cluster_size)
        if target_sms is None:
            return                                 # wait next cycle
        scheduler.commit(k=cluster_size)
        for cta, sm in zip(next_cluster, target_sms):
            sm.activate_cta(cta, cluster_id, cluster_rank, ...)
            recorder.cta_dispatch(...)
        recorder.cluster_dispatch(cycle, cluster_id, [sm_id ...])
        ptr += cluster_size
```

**dsmem 访问**：
```
mapa.shared::cluster  %dst, %smem_ptr, %rank
  → %dst = (rank << 24) | smem_ptr     # 指针编码

ld.shared::cluster.f32  %r, [%cluster_ptr]
  rank   = (%cluster_ptr >> 24) & 0xFF
  offset = %cluster_ptr & 0xFFFFFF
  cluster_id    = current_warp.cluster_id
  target_cta_id = cluster_id * cluster_size + rank
  return SharedMemory._cta[target_cta_id][offset:offset+4]

st.shared::cluster.f32  [%cluster_ptr], %r → 同样路由到目标 CTA
```

**Cluster barrier**：
```
barrier.cluster.arrive      # CTA 内所有 warp 到达此 PC → CTA arrive 在 cluster barrier
barrier.cluster.wait        # 等到 cluster 内所有 CTA 都 arrive → 全 cluster 放行

ClusterBarrierPool (per cluster_id, owned by Device):
  expected = cluster_size
  arrived_mask = bitmask of arrived ranks
  phase = 0 / 1
  arrive(rank) -> bool:
    arrived_mask |= (1 << rank)
    if popcount(arrived_mask) == expected:
      arrived_mask = 0; phase ^= 1; return True
    return False
  is_released(captured_phase) -> bool:
    return phase != captured_phase
```

**Cluster mbarrier**：复用 Phase 3 `MbarrierPool` 数据结构；`mbarrier.{init,arrive,try_wait}.shared::cluster` 在 SubCore 路径上：
- 解码 mbar 指针的 cluster 部分（rank | offset）
- 找 target CTA 的 MbarrierPool（cluster_id * cluster_size + rank）
- 调用该 pool 的 init/arrive/try_wait

**Cluster TMA load**：`cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` 已经在 Phase 3 解析；Phase 5 实质化：
- smem_dst 指针可以是 cluster 编码 → 解码到 remote CTA's smem 区域
- mbar 指针同样可 cluster 编码 → 在 remote CTA's mbarrier arrive_tx
- functional 层：`do_bulk_copy_2d` 增加 `target_cta_id` 参数；其余路径不变

### 2.2 关键不变量（继承 Phase 1-4）

- **Functional vs timing 分离**：cluster 拓扑只影响 timing（dispatch + barrier）+ functional 路由（dsmem ld/st 解码到目标 CTA）
- **Trace 防火墙**：`ClusterDispatchEvent` + `ClusterBarrierEvent` 经 Recorder
- **API 不变**：`gpusim.run(...)` 签名不变；cluster_size=1 默认下行为同 Phase 4
- **Device 不持有 functional state**：cluster smem 仍是 SharedMemory 的 per-CTA buffer 字典，cluster 只是访问路由

### 2.3 模块拓扑

```
gpusim/core/
├── cluster.py                       NEW: ClusterBarrierPool + helpers
├── device.py                        MODIFY: cluster-aware _try_dispatch + _cluster_barriers
├── sm.py                            MODIFY: activate_cta 加 cluster_id/rank；step_cycle 协调 cluster barrier
├── sub_core.py                      MODIFY: + ld/st.shared::cluster + mapa + barrier.cluster.* + mbarrier.shared::cluster + cluster TMA load
├── tma.py                           MODIFY: do_bulk_copy_2d 接受 target_cta_id
├── warp.py                          MODIFY: + cluster_id, cluster_rank, cluster_barrier_arrived, cluster_barrier_wait_pc, cluster_barrier_phase_at_wait
├── exec.py                          MODIFY: InstrExecutor + cluster_id/cluster_size; mapa + dsmem ld/st + getctarank
└── scheduler.py                     MODIFY: RR/Greedy 加 peek(k) + commit(k) 接口

gpusim/frontend/parser.py            MODIFY: + barrier.cluster.{arrive,wait} + mapa.shared::cluster + ld/st.shared::cluster + mbarrier.shared::cluster + getctarank.u32

gpusim/config/
├── schema.py                        MODIFY: + DeviceConfig.cluster_size: int = 1
└── default_hopper.yaml              MODIFY: + device.cluster_size: 1

gpusim/trace/
├── events.py                        MODIFY: + ClusterDispatchEvent + ClusterBarrierEvent
├── recorder.py                      MODIFY: + 2 methods
└── writer.py                        MODIFY: + 2 parquet writers

gpusim/analysis/metrics.py           MODIFY: + 3 metrics
gpusim/viz/                          MODIFY: + 2 HTML 节 + Perfetto cluster track
gpusim/api.py                        MODIFY: + cluster_*_events_df + cluster_metrics + cluster_summary
```

### 2.4 Phase 1-4 carry-over

仅当 Phase 5 example 真撞到才修：
- 早 phases deferred items (`0f` literal、IPDOM)：cluster examples 用 numpy 构造数据，不阻塞
- Phase 4 deferred items：Device 已 ship，无未完结项

### 2.5 边界原则

1-5 (从 Phase 1-4 继承)
6. **Cluster 是 Device 层概念** —— SM 只持有 CTA + ClusterBarrierPool 引用；Device 协调 cluster 派发 + barrier 状态机
7. **dsmem 解码在 InstrExecutor 层** —— SubCore 不知道 cluster，只把指针交给 InstrExecutor 解码

---

## 3. PTX 子集扩展 + IR 改动

### 3.1 新增指令

| 指令 | 用途 |
|---|---|
| `mapa.shared::cluster %dst, %src_smem_ptr, %rank` | 把本地 smem 指针 + cluster rank 编码为 cluster 远程指针 |
| `ld.shared::cluster.<ty> %dst, [%cluster_ptr]` | 从 cluster 远程 CTA's smem 加载 |
| `st.shared::cluster.<ty> [%cluster_ptr], %src` | 写到 cluster 远程 CTA's smem |
| `barrier.cluster.arrive` | CTA-wide arrive（不阻塞），通知 cluster |
| `barrier.cluster.wait` | CTA-wide wait（阻塞），等 cluster 全 arrive |
| `mbarrier.init.shared::cluster [%mbar], %expected` | 用 cluster 指针指向远程 mbarrier 时初始化 |
| `mbarrier.arrive.shared::cluster [%mbar]` | 在 cluster 远程 mbarrier arrive |
| `mbarrier.try_wait.parity.shared::cluster %p, [%mbar], %phase` | 在 cluster 远程 mbarrier try_wait |
| `getctarank.u32 %r` | 读取本 CTA 的 cluster rank |

注：Phase 3 已解析 `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`——Phase 5 不新增 op，但运行时把 smem_dst + mbar 指针解码为 cluster 远程指针时实际生效。

### 3.2 IR 改动

**无新 PtxType / 新 IR 节点**。复用 Phase 3 的 `Reg / Imm / RegGroup`。

新增 special register：`%clusterrank`（per-warp）。Lexer 已识别 `%`-前缀；只需 InstrExecutor `_resolve_special` 加分支返回 `cluster_rank`。

### 3.3 Parser 改动

(a) `mapa.shared::cluster %dst, %src, %rank`：标准 3-operand 形式；现有通用解码可处理。

(b) `ld.shared::cluster.<ty>` / `st.shared::cluster.<ty>`：现有 ld/st 路径要识别 `shared::cluster` 命名空间。当前 `_parse_operands` 对 `ld.` 前缀走通用 `[addr]` 解析，`shared` vs `shared::cluster` 区别在 op 字符串。运行时（SubCore）按 op 字符串区分。

(c) `barrier.cluster.arrive` / `barrier.cluster.wait`：无操作数，加专用分支返回 `[], []`。

(d) `mbarrier.init.shared::cluster [%mbar], %expected` 等：现有 `mbarrier.init.shared::cta` 等已有 parser 路径；`shared::cluster` 形式同结构（`[addr], imm`），现有路径基本能复用——只需把 op 前缀 prefix-match 改宽，让 `mbarrier.init.shared::cluster` 也匹配。

(e) `getctarank.u32 %r`：单 dst 的 sreg 读取。加专用分支或当 mov.u32 处理。

### 3.4 FUSet.classify 改动

```python
        if op.startswith("mapa."):
            return FUKind.INT
        if op.startswith("barrier.cluster."):
            return FUKind.SYNC
        if op == "getctarank.u32":
            return FUKind.INT
        # ld.shared::cluster / st.shared::cluster / mbarrier.shared::cluster
        # 已经被现有 ld./st./mbarrier. 通用规则路由到对应 FU，不需新增
```

### 3.5 Special register 扩展

`InstrExecutor._resolve_special` 加：

```python
        if sreg == "clusterrank":
            return self.cluster_rank   # 新增 InstrExecutor 字段
```

`InstrExecutor.__init__` 加 `cluster_rank: int = -1` 字段（默认 -1 = 不在 cluster 内），由 SM.activate_cta 注入。

---

## 4. Cluster + dsmem 详细设计

### 4.1 DeviceConfig 扩展

```python
@dataclass
class DeviceConfig:
    n_sm: int = 8
    cluster_size: int = 1            # NEW (Phase 5): default 1 = no clustering
    sm: SMConfig = field(default_factory=SMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    hbm: HBMConfig = field(default_factory=HBMConfig)
    scheduler: CtaSchedulerConfig = field(default_factory=CtaSchedulerConfig)
```

`default_hopper.yaml` 加 `device.cluster_size: 1`。

`Device.run` 启动时校验：`grid_size % cluster_size == 0`，否则 `ValueError`。

### 4.2 Cluster 派发

新 scheduler 接口：

```python
class RRCtaScheduler:
    def peek(self, sms, occ, k=1):
        """Try to find k admittable SMs without committing internal state.
        Returns list of SMs or None."""
        n = len(sms)
        candidates = []
        try_next = self._next
        for _ in range(n):
            sm = sms[try_next]
            if sm.can_admit_cta(occ):
                candidates.append(sm)
                if len(candidates) == k:
                    self._pending_advance = (try_next + 1) % n
                    return candidates
            try_next = (try_next + 1) % n
        return None

    def commit(self, k=1):
        self._next = self._pending_advance


class GreedyCtaScheduler:
    def peek(self, sms, occ, k=1):
        eligible = sorted(
            [sm for sm in sms if sm.can_admit_cta(occ)],
            key=lambda sm: sm.active_warp_count())
        if len(eligible) >= k:
            return eligible[:k]
        return None

    def commit(self, k=1):
        # Greedy is stateless; commit is a no-op
        pass
```

Device.run 改 `_try_dispatch`：

```python
def _try_dispatch():
    nonlocal cta_pointer
    cluster_size = self.cfg.cluster_size
    while cta_pointer < len(cta_queue):
        target_sms = scheduler.peek(sms, occ, k=cluster_size)
        if target_sms is None:
            return
        scheduler.commit(k=cluster_size)
        cluster_id = cta_pointer // cluster_size
        for i, sm in enumerate(target_sms):
            cid, ctaid_xyz = cta_queue[cta_pointer + i]
            sm.activate_cta(cid, ctaid_xyz, regs_per_thread, smem_per_cta,
                              threads_per_cta, warps_per_cta, cycle,
                              cluster_id=cluster_id, cluster_rank=i)
            if self.recorder is not None:
                self.recorder.cta_dispatch(...)
        # Init cluster barrier pool entry
        from gpusim.core.cluster import ClusterBarrierPool
        self._cluster_barriers[cluster_id] = ClusterBarrierPool(
            expected=cluster_size,
        )
        if self.recorder is not None:
            self.recorder.cluster_dispatch(
                cycle=cycle, cluster_id=cluster_id,
                cluster_size=cluster_size,
                sm_ids=tuple(sm.sm_id for sm in target_sms),
                cta_ids=tuple(cta_queue[cta_pointer + i][0]
                                for i in range(cluster_size)),
                queue_position=cta_pointer // cluster_size,
            )
        cta_pointer += cluster_size
```

### 4.3 Warp 字段 + SM.activate_cta 签名扩展

```python
@dataclass
class Warp:
    # ... existing fields ...
    cluster_id: int = -1                     # NEW
    cluster_rank: int = -1                   # NEW
    cluster_barrier_arrived: bool = False    # NEW
    cluster_barrier_wait_pc: int = -1        # NEW
    cluster_barrier_phase_at_wait: int = -1  # NEW
```

`SM.activate_cta(...)` 加 keyword 参数 `cluster_id`、`cluster_rank`，给该 CTA 的所有 warps 设置；构造 `InstrExecutor` 时也传入这两值。

### 4.4 ClusterBarrierPool

`gpusim/core/cluster.py`（新）：

```python
@dataclass
class ClusterBarrierPool:
    expected: int                                     # = cluster_size
    arrived_mask: int = 0
    phase: int = 0

    def arrive(self, cluster_rank: int) -> bool:
        """Returns True if this arrive completes the barrier."""
        self.arrived_mask |= (1 << cluster_rank)
        if bin(self.arrived_mask).count("1") >= self.expected:
            self.arrived_mask = 0
            self.phase ^= 1
            return True
        return False

    def is_released(self, captured_phase: int) -> bool:
        return self.phase != captured_phase
```

`Device` 持有 `_cluster_barriers: dict[int, ClusterBarrierPool]`，注入到每个 SM。SM 通过 `self._device_cluster_barriers` 引用。

### 4.5 barrier.cluster.{arrive,wait} 协调

**arrive 路径**（在 SM.step_cycle 协调，复用 Phase 1 的 bar.sync CTA-wide 协调机制）：

`_is_ready` 见 `barrier.cluster.arrive` → 设 `w.barrier_pc = pc`，return `BARRIER`。

SM.step_cycle 的 CTA-level 协调扩展：

```python
for cid, ws in by_cta.items():
    non_done = [w for w in ws if not w.finished]
    if non_done and all(w.barrier_pc >= 0 for w in non_done):
        instr = non_done[0].kernel.instrs[non_done[0].barrier_pc]
        if instr.op == "barrier.cluster.arrive":
            cluster_id = non_done[0].cluster_id
            rank = non_done[0].cluster_rank
            pool = self._device_cluster_barriers[cluster_id]
            pool.arrive(rank)
            if self.recorder is not None:
                self.recorder.cluster_barrier(
                    kind="ARRIVE", cycle=cycle,
                    cluster_id=cluster_id, cta_id=cid,
                    rank=rank, sm_id=self.sm_id,
                    arrived_count=bin(pool.arrived_mask).count("1"),
                )
            for w in non_done:
                w.stack.update_top_pc(w.barrier_pc + 1); w.stack.maybe_pop()
                w.barrier_pc = -1
        else:
            # bar.sync 既有逻辑
            for w in non_done:
                w.stack.update_top_pc(w.barrier_pc + 1); w.stack.maybe_pop()
                w.barrier_pc = -1
```

**wait 路径**：

`_is_ready` 见 `barrier.cluster.wait` → 第一次见时 snapshot phase + 设 `w.cluster_barrier_wait_pc = pc`，return `CLUSTER_BARRIER_WAIT`。后续 cycle 检查 phase delta。

```python
# In SM.step_cycle, after barrier coordination:
for w in self._active_warps:
    if w.cluster_barrier_wait_pc >= 0:
        pool = self._device_cluster_barriers.get(w.cluster_id)
        if pool is None:
            continue
        if pool.is_released(w.cluster_barrier_phase_at_wait):
            w.stack.update_top_pc(w.cluster_barrier_wait_pc + 1)
            w.stack.maybe_pop()
            w.cluster_barrier_wait_pc = -1
            if self.recorder is not None:
                self.recorder.cluster_barrier(
                    kind="WAIT_RELEASE", cycle=cycle,
                    cluster_id=w.cluster_id,
                    cta_id=w.cta_id, rank=w.cluster_rank,
                    sm_id=self.sm_id,
                )
```

### 4.6 dsmem ld/st 路由

InstrExecutor `_exec_lane`：

```python
if op.startswith("ld.shared::cluster.") or op.startswith("st.shared::cluster."):
    base_addr = self._read(t, instr.src[0], PtxType.u64)
    rank = (base_addr >> 24) & 0xFF
    offset = base_addr & 0xFFFFFF
    cluster_id = self.cluster_id
    target_cta_id = cluster_id * self.cluster_size + rank
    if op.startswith("ld."):
        ty = parse_type_from_op(op)
        if ty is PtxType.f32:
            v = self.smem.load_f32(target_cta_id, offset)
        # ... 其他 dtypes ...
        self._write(t, instr.dst[0], v, ty)
    else:
        ty = parse_type_from_op(op)
        v = self._read(t, instr.src[1], ty)
        if ty is PtxType.f32:
            self.smem.store_f32(target_cta_id, offset, v)
    return
```

`mapa.shared::cluster %dst, %src, %rank`：

```python
if op == "mapa.shared::cluster":
    smem_offset = self._read(t, instr.src[0], PtxType.u64)
    rank = self._read(t, instr.src[1], PtxType.u32)
    encoded = (rank << 24) | (smem_offset & 0xFFFFFF)
    self._write(t, instr.dst[0], encoded, PtxType.u64)
    return
```

### 4.7 Cluster mbarrier 路由

SubCore._issue 既有 `mbarrier.*` 分支扩展：

```python
if op.startswith("mbarrier.init."):
    addr = w.fn_state.threads[0].get_u64(instr.src[0].name)
    if "shared::cluster" in op:
        rank = (addr >> 24) & 0xFF
        offset = addr & 0xFFFFFF
        target_cta = w.cluster_id * self.cfg.cluster_size + rank
        pool = self.mbarrier_pools[target_cta]
        pool.init(smem_addr=offset, expected=int(instr.src[1].value))
    else:
        # Phase 3 path unchanged
        ...
```

`mbarrier.arrive.shared::cluster` / `mbarrier.try_wait.parity.shared::cluster` 同模式。

### 4.8 Cluster TMA load 路由

Phase 3 已建好的 `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` 路径在 SubCore._issue：Phase 5 让 smem_dst + mbar 指针都可 cluster 编码：

```python
if op.startswith("cp.async.bulk.tensor."):
    smem_dst_reg = instr.src[0]
    desc_reg = instr.src[1]
    mbar_reg = instr.src[2] if len(instr.src) > 2 else None
    smem_dst_ptr = w.fn_state.threads[0].get_u64(smem_dst_reg.name)

    if "shared::cluster" in op:
        rank = (smem_dst_ptr >> 24) & 0xFF
        smem_offset = smem_dst_ptr & 0xFFFFFF
        target_cta = w.cluster_id * self.cfg.cluster_size + rank
        if mbar_reg:
            mbar_ptr = w.fn_state.threads[0].get_u64(mbar_reg.name)
            mbar_rank = (mbar_ptr >> 24) & 0xFF
            mbar_offset = mbar_ptr & 0xFFFFFF
            target_mbar_cta = w.cluster_id * self.cfg.cluster_size + mbar_rank
        else:
            target_mbar_cta = target_cta
            mbar_offset = -1
        do_bulk_copy_2d(gmem=..., smem=..., cta_id=target_cta,
                          smem_dst=smem_offset, desc=desc)
        if mbar_reg:
            pool = self.mbarrier_pools[target_mbar_cta]
            pool.arrive_tx(smem_addr=mbar_offset, ...)
    else:
        # Phase 4 path: target = local CTA
        ...
```

### 4.9 新 stall token

| Token | 触发 |
|---|---|
| `CLUSTER_BARRIER_WAIT` | warp 在 `barrier.cluster.wait`，等 cluster 内所有 CTA 都 arrive |

总 stall 类数：17 → **18**。

---

## 5. Trace + 分析 + 可视化

### 5.1 完整事件清单

| 类别 | 事件 | 频率 |
|---|---|---|
| Phase 1 (8) | WARP_STATE, INSTR_ISSUE, SMEM_ACCESS, GMEM_ACCESS, DIV_PUSH/POP, BAR_REACH/RELEASE, CTA_LAUNCH/RETIRE | 高（RLE） |
| Phase 2 (3) | L1_ACCESS, L2_ACCESS, HBM_ACCESS | 中-低 |
| Phase 3 (4) | MmaEvent, WgmmaEvent, TmaEvent, MbarrierEvent | 中-低 |
| Phase 4 (3) | CtaDispatchEvent, L2MshrEvent, BulkStoreEvent | 中-低 |
| **Phase 5 (2)** | **ClusterDispatchEvent, ClusterBarrierEvent** | 中-低 |

### 5.2 Phase 5 新事件 schema

```python
@dataclass(frozen=True)
class ClusterDispatchEvent:
    cycle: int
    cluster_id: int
    cluster_size: int
    sm_ids: tuple[int, ...]
    cta_ids: tuple[int, ...]
    queue_position: int


@dataclass(frozen=True)
class ClusterBarrierEvent:
    kind: str                  # "ARRIVE" | "WAIT_BLOCK" | "WAIT_RELEASE"
    cycle: int
    cluster_id: int
    cta_id: int
    rank: int
    sm_id: int
    arrived_count: int = 0
```

### 5.3 现有 events 微调

无 schema 变化。Phase 5 不动 Phase 1-4 既有事件字段。

### 5.4 新增分析指标（3 个）

| 函数 | 输出 | 教学用途 |
|---|---|---|
| `cluster_dispatch_latency(cluster_dispatch_df, cta_launch_df)` | pd.Series（cycle 直方图） | Cluster 等所有 K 个 SM 都空才能整体派的等待分布 |
| `cluster_barrier_wait_distribution(cluster_barrier_df)` | pd.Series | 每个 cluster barrier 从首 ARRIVE 到 WAIT_RELEASE 的 cycle 数分布 |
| `dsmem_remote_access_rate(instr_issue_df)` | scalar 0..1 | INSTR_ISSUE 中 op 含 `shared::cluster` 的 ld/st 占所有 ld/st.shared 的比例（cluster 协作密度） |

### 5.5 HTML 报告新增节（2 节）

| 节 | 内容 |
|---|---|
| **§19 Cluster dispatch + barrier timeline** | 每 cluster 一行：dispatch 时间、各 CTA arrive 时间、wait release 时间。Plotly Gantt-style |
| **§20 dsmem cross-CTA traffic** | `dsmem_remote_access_rate` + 表格：每 cluster 内的 cross-CTA ld/st count |

### 5.6 Result API 扩展

```python
@dataclass
class Result:
    # ... Phase 1-4 fields ...

    @property
    def cluster_dispatch_events_df(self) -> pd.DataFrame: ...
    @property
    def cluster_barrier_events_df(self) -> pd.DataFrame: ...

    @property
    def cluster_metrics(self) -> dict: ...

    def cluster_summary(self) -> str: ...
```

`Result.summary()` 在 cluster_size > 1 时追加 `cluster_summary()` 行。

### 5.7 Perfetto 集成

| 事件 | Perfetto track | 视觉 |
|---|---|---|
| `ClusterDispatchEvent` | per-cluster "Cluster" track | 蓝色 instant + cluster_id 标签 |
| `ClusterBarrierEvent("ARRIVE")` | per-cluster "Cluster Barrier" track | 黄色 instant |
| `ClusterBarrierEvent("WAIT_BLOCK")` | per-cluster "Cluster Barrier" track | 红色 instant |
| `ClusterBarrierEvent("WAIT_RELEASE")` | per-cluster "Cluster Barrier" track | 绿色 instant |

PIDs 形如 `Cluster{cluster_id}` 让 N 个 cluster 在 Perfetto 上并行可见。

### 5.8 Parquet 落盘

新增 2 个 parquet：`cluster_dispatch.parquet`、`cluster_barrier.parquet`。

---

## 6. 测试策略

### 6.1 单元测试

| 模块 | 关键测试 |
|---|---|
| `core/cluster` | ClusterBarrierPool arrive 累计；popcount = expected 时 phase 翻转；is_released 正确 |
| `core/device` | cluster-aware `_try_dispatch`：peek 失败回退；K 个 SM 都 OK 才 commit；cluster_size=1 退化到 Phase 4 行为 |
| `core/sm` | SM.activate_cta 接受 cluster_id/cluster_rank 写入 warps；SM.step_cycle 协调 cluster barrier |
| `core/sub_core` | dsmem ld/st 解码路径；cluster mbarrier 路由到 target CTA's pool |
| `core/exec` | mapa.shared::cluster 编码；getctarank.u32 special reg 解析；cluster ld/st target CTA 路由 |
| `core/scheduler` | RR/Greedy peek + commit 接口正确（不预占用） |
| `frontend/parser` | barrier.cluster.{arrive,wait}、mapa.shared::cluster、ld/st.shared::cluster、mbarrier.shared::cluster、getctarank.u32 解析 |
| `analysis/metrics` | 3 个新指标 fixture |
| `viz/html_report` | §19/§20 在 cluster 事件存在时正确插入 |
| `config/loader` | DeviceConfig.cluster_size 默认 1；可从 yaml 读到 8/4/2 |

### 6.2 Functional Parity（numpy）

3 个新 example 各有 numpy 参考实现：
- `cluster_basic`：cluster_size=2，CTA 0 写入 dsmem，CTA 1 读取 + 写到 OUT；numpy 直对
- `cluster_matmul_dsmem`：4-CTA cluster，wgmma + dsmem 共享 A tile；numpy fp16 matmul，rtol=1e-2
- `cluster_tma_pipeline`：cluster_size=4，CTA 0 用 cluster TMA load 把数据写到所有 4 个 CTA 的 smem，每 CTA 各算一段输出；numpy 直对

Phase 1-4 example 全部继续通过。

### 6.3 Reference Fixture

`tests/reference/data/` 加 3 个 stub。容忍度：
- `cluster_dispatch_latency` ±20%
- `cluster_barrier_wait` ±15%
- `dsmem_remote_access_rate` ±5%

`gen_reference.py` SUPPORTED_KERNELS 加 3 项。

### 6.4 微基准

新增 `tests/microbench/test_phase5_facts.py`：

```
- cluster_basic 在 cluster_size=2 下 cycles 比 cluster_size=1 同 grid 的版本 small overhead（≤ 1.5×）
- cluster_matmul_dsmem 中 dsmem_remote_access_rate ≥ 0.4
- cluster_tma_pipeline 中 cluster TMA load 写到 remote CTA's smem：HBM 流量比 4 个独立 CTA 各拉一份 ≤ 0.4×
- cluster barrier wait < 50 cycle 在 8-SM/4-cluster 拓扑下
- cluster_size=1 下 Phase 4 既有 example 的 cycles 浮动 ≤ 5%
```

### 6.5 Phase 1-4 兼容性测试

扩展现有 Phase 1-3 regression test → 改名 `test_phase1_4_examples_unchanged.py`：
- 跑全部 Phase 1-4 的 17 个 example
- 输出数值不变，cycles 浮动 ≤ 5%

### 6.6 Memory budget 测试

新增 `tests/microbench/test_phase5_runtime.py`（@pytest.mark.slow）：
- cluster_matmul_dsmem 完整跑 < 60 秒
- cluster_tma_pipeline 完整跑 < 60 秒

---

## 7. 项目结构改动

### 7.1 目录变化

```
gpusim/core/
├── cluster.py                       NEW
├── device.py                        MODIFY
├── sm.py                            MODIFY
├── sub_core.py                      MODIFY
├── tma.py                           MODIFY
├── warp.py                          MODIFY
├── exec.py                          MODIFY
└── scheduler.py                     MODIFY

gpusim/frontend/parser.py            MODIFY
gpusim/config/{schema.py, default_hopper.yaml}    MODIFY
gpusim/trace/{events.py, recorder.py, writer.py}  MODIFY
gpusim/analysis/metrics.py           MODIFY
gpusim/viz/                          MODIFY
gpusim/api.py                        MODIFY

tests/unit/core/test_{cluster,device_cluster,sm_cluster,sub_core_cluster,exec_cluster}.py   NEW
tests/unit/frontend/test_parser_phase5.py    NEW
tests/parity/test_{cluster_basic,cluster_matmul_dsmem,cluster_tma_pipeline}.py    NEW
tests/parity/test_phase1_4_examples_unchanged.py    RENAME from phase1_3
tests/microbench/test_{phase5_facts,phase5_runtime}.py    NEW
tests/reference/data/{cluster_basic,cluster_matmul_dsmem,cluster_tma_pipeline}.ref.json    NEW
```

### 7.2 配置 yaml 变化

`default_hopper.yaml` 顶层 `device:` 节加：

```yaml
device:
  n_sm: 8
  cluster_size: 1                    # NEW (Phase 5)
  scheduler:
    cta_policy: rr
```

向后兼容：`cluster_size` 缺省读取为 1 → Phase 4 行为。

### 7.3 依赖

无新依赖。

---

## 8. 教学示例与讲义

### 8.1 3 个新 example

| # | Example | grid / cluster | 教学意图 |
|---|---|---|---|
| 1 | **cluster_basic** | grid=(2,1,1), cluster_size=2 | 最小 2-CTA cluster：mapa + ld.shared::cluster + barrier.cluster.{arrive,wait} 基础语义 |
| 2 | **cluster_matmul_dsmem** | grid=(4,1,1), cluster_size=4, block=(128) | 4-CTA cluster + wgmma：CTA 0 拉 A tile 进自身 smem；其他 CTA 用 mapa 共享访问；barrier.cluster.wait 同步；wgmma 在共享数据上算 |
| 3 | **cluster_tma_pipeline** | grid=(4,1,1), cluster_size=4, block=(128) | Cluster TMA load + cluster mbarrier：CTA 0 issue cluster TMA load 写到所有 4 个 cluster CTA 的 smem；每 CTA 用 cluster mbarrier try_wait 等数据 |

每目录：`{kernel.ptx}` + `reference.py` + `run.py` + `README.md` + `__init__.py`。

### 8.2 3 章新讲义（chapters 19–21）

| # | 标题 | 关联 example |
|---|---|---|
| 19 | Hopper Cluster (CGA) 入门：dsmem + barrier.cluster | cluster_basic |
| 20 | Cluster + wgmma：协同 matmul 与 dsmem 共享数据 | cluster_matmul_dsmem |
| 21 | Cluster TMA 与 mbarrier：分布式 producer-consumer 流水线 | cluster_tma_pipeline |

每章固定栏目：**看模拟器** / **改一改** / **真机对照**。

---

## 9. 与 Phase 1-4 兼容性

### 9.1 不会破坏的部分

| 维度 | 状态 |
|---|---|
| `gpusim.run(...)` 函数签名 | 不变 |
| Phase 1-4 example PTX | 不动 |
| Phase 1-4 parity 测试 | 全部继续通过（cluster_size=1 默认 → Phase 4 行为 byte-for-byte） |
| Result 旧字段 | 不变 |
| HTML 报告 §1–§18 | 位置 + 内容不变 |
| Perfetto 既有 track | 不变 |
| Stall token 既有 17 类 | 不变 |
| 新依赖 | 无 |
| Trace 既有事件 schema | 不变 |

### 9.2 会变的部分

| 维度 | 变化 |
|---|---|
| Cycle 数（Phase 1-4 example） | cluster_size=1 默认下不变；Device.run 加 peek/commit 路径但等价于"K=1 时单 SM 试派"，行为相同 |
| Stall 直方图 | 多 1 类 `CLUSTER_BARRIER_WAIT`（仅 cluster kernel 出现） |
| HTML 报告 | 多 2 节（§19, §20） |
| Perfetto | 新 track（每 cluster 一行 dispatch + barrier） |
| Trace parquet | 多 2 文件 |
| Result | 多 2 properties + `cluster_metrics` + `cluster_summary()` |
| 配置 yaml | `device:` 节加 `cluster_size: 1` 字段（缺省读为 1） |
| `Warp` | 加 5 字段；默认值兼容 |
| `InstrExecutor` | 加 cluster_id / cluster_size 字段；默认 -1 / 1 兼容 |
| `SM.activate_cta` 签名 | + `cluster_id`、`cluster_rank` keyword 参数；现有调用兼容 |
| Scheduler 接口 | + `peek(sms, occ, k=1)` + `commit(k=1)`；默认 k=1 等价 Phase 4 `pick` |

### 9.3 Phase 4 既有 SM/Device path 兼容回退

`Device.run` 检测 `cluster_size`：
- `== 1`：走 Phase 4 单 CTA 派发路径（每 cycle 试派 1 个 CTA），所有 cluster_id/rank = -1
- `> 1`：走 cluster 批派路径，强制 `grid_size % cluster_size == 0`

`SM` 内部代码均检查 `w.cluster_id >= 0` 才走 cluster 路径；否则路径与 Phase 4 同。

---

## 10. 显式不在范围内（Phase 5）

记录以避免误解：

- **Atomics**：`atom.*`、`red.*`（Phase 6 主题）
- **Cluster TMA store**：`cp.async.bulk.tensor.2d.global.shared::cluster` 反向（smem→gmem 跨 cluster）—— Phase 6 配 cooperative epilogue
- **Multi-stream / 多 kernel 并发**：Phase 7
- **多 GPU、NVLink、NCCL**：Phase 8
- **Cluster smem 容量统一计算**：每 CTA 仍按 `smem_per_sm_bytes` 分配；不模拟"cluster 总 smem = K × smem_per_cta"作为单一 budget
- **GPC 拓扑**：cluster 不绑定到具体 GPC；任意 cluster_size 个 SM 都可以 host cluster（教学简化）
- **Distributed shared memory bank conflicts**：dsmem 跨 SM 访问当 1 cycle latency 算（不模拟 SM-to-SM 互连延迟）
- **Cluster preemption**：cluster 一旦派发就跑到全 cluster retire；不模拟 cluster 中途 stall 全 cluster
- **3D cluster grid**：Phase 5 仅 1D cluster（`cluster_size` 标量）；真机支持 cluster.{x,y,z}

---

## 11. 已知近似与简化

- **Cluster topology 简化**：任意 cluster_size 个 SM 形成 cluster；不模拟 GPC 边界。真机 H100 cluster 必须 fit 在单 GPC（9 SMs）内
- **dsmem 访问无延迟差**：Phase 5 把 cluster smem 远程访问当本地访问算（同 smem_latency = 20 cycle）。真机 cluster smem 跨 SM 通过 NoC，延迟略高于本地 smem
- **Cluster barrier 简化**：仅 popcount-based 计数；真机用专用 cluster barrier 硬件
- **Cluster mbarrier 复用 per-CTA pool**：通过指针解码路由；真机 cluster mbarrier 是物理上的 dsmem 区域
- **No SM-pair affinity**：cluster CTA 落到哪个 SM 由 scheduler 决定；真机的"GPC 内 SM 间互连有距离差异"未建模
- **Single-cluster wave only**：当 cluster_size 个 SM 不够时整 cluster 等；不模拟 GPC 内多 cluster 同时调度

---

## 12. Phase 5 实施里程碑

| 里程碑 | 交付 |
|---|---|
| **M1** | Frontend + 配置：parser (5 新 op + cluster mbarrier + cluster ld/st)、DeviceConfig.cluster_size、yaml + loader、Warp 字段、InstrExecutor cluster_id 字段。无运行时行为变化（cluster_size=1 兼容） |
| **M2** | Cluster 派发 + barrier：Device 批派 + scheduler peek/commit + ClusterBarrierPool + barrier.cluster.{arrive,wait} 协调 + cluster_basic example + 1 stall token |
| **M3** | dsmem + cluster mbarrier：mapa.shared::cluster + ld/st.shared::cluster + getctarank + mbarrier cluster 路由 + cluster_matmul_dsmem example |
| **M4** | Cluster TMA load：cp.async.bulk.tensor.shared::cluster 实质化（remote CTA 解码） + cluster_tma_pipeline example |
| **M5** | Trace + 分析 + viz + 收尾：2 trace events + 3 分析指标 + 2 HTML 节 + Perfetto cluster track + Result API + 3 章新讲义 + Phase 1-4 兼容性测试 + Phase 5 microbench + reference fixture + README v5 + tag `phase5-complete` |

预估总任务数：**26-30**（小于 Phase 3/4 的 33-35，因为：无新算术、无新 FU、复用既有 mbarrier/TMA 路径）。

每 milestone 后打 git tag (`M{1..5}-phase5-complete`)。

---

## 13. 设计协作记录

本文档由用户与 Claude（Opus 4.7, 1M context）通过 `superpowers:brainstorming` 流程逐节确认产出。所有关键决策均经用户显式确认（A/B/C 选择或"ok"回复）。

下一步：交由 `superpowers:writing-plans` 产出可执行的实施计划。
