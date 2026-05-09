# gpusim Phase 6 — Atomics + Cluster TMA Store + Cooperative Epilogue 设计文档

**日期**：2026-05-09
**状态**：设计阶段（待实现）
**作者**：与 Claude 协同 brainstorm
**前置依赖**：Phase 1-5 完成（tag `phase5-complete`，HEAD `044b42d`）
**范围**：仅 Phase 6。Phase 7+ 仅作为愿景列出。

---

## 1. 愿景与 Phase 6 范围

### 1.1 项目背景

Phase 1-5 交付了完整 multi-SM Hopper 模拟器（Tensor Core + wgmma + TMA load/store + mbarrier + Cluster + dsmem）。但所有路径都是"无竞争路径"——同一 line 多 SM 同时访问要么 hit cache 要么 miss serialize via L2 MSHR。GPU 真实生产中 atomic 是另一个核心维度：硬件 atomic ALU + 跨 SM 串行 + lock-free 模式。Phase 5 cluster_matmul_dsmem 的 wgmma + cluster TMA store cooperative epilogue 也因缺少 cluster TMA store 实现 deferred 到这里。

### 1.2 Phase 6 一句话目标

> 把 atomic 子系统加到模拟器：gmem atomic 走 L2 atomic ALU per-line 串行 + smem atomic 复用 bank conflict 路径；补 cluster TMA store 完成 cluster cooperative epilogue 故事。5 个新 example + 5 章新讲义。

### 1.3 路线图回顾

| Phase | 范围 | 状态 |
|---|---|---|
| Phase 1-5 | 单 SM → multi-SM Hopper full stack | ✅ |
| **Phase 6** | **Atomics (atom.* / red.*) gmem+smem + cluster TMA store + cooperative epilogue** | **本文档** |
| Phase 7 | Multi-stream / 多 kernel 并发 | 后续 |
| Phase 8 | Multi-GPU + NVLink + NCCL | 后续 |

### 1.4 已锁定决策

| 维度 | 决策 |
|---|---|
| 范围 | 窄（A 选项）：atomic 子集 + cluster TMA store；不做 cluster atomic / 64-bit atomic / fp16 atomic |
| Atomic ops | 5 个 `atom.*`：add, min, max, cas, exch + 3 个 `red.*`：add, min, max（共 8 op） |
| Spaces | `.global` (走 L2 atomic ALU) + `.shared` (走 smem bank)；不做 `.shared::cluster` |
| Dtypes | u32 / s32 / f32 |
| gmem atomic 模型 | 真实 L2 ALU per-line queue（A 选项）：每 line 一个 atomic FIFO；多 SM 同 line 串行 |
| smem atomic 模型 | 复用 Phase 1 bank conflict + `atomic_op_extra_latency = 4` |
| Cluster TMA store | 完整：`cp.async.bulk.tensor.2d.global.shared::cluster` smem_src 可 cluster 编码 → 从 remote CTA's smem 读 → store gmem |
| 顶层架构 | 复用 Phase 4-5；L2 加 atomic queue dict；SubCore 加 atomic 路由 |
| 新 stall token | **0**（atomic 不引 warp-level stall，仅 latency 增加） |
| 新 trace 事件 | `AtomicEvent`（18 → 19 类） |
| 新分析指标 | 4：atomic_throughput_per_line / atomic_serialization_overhead / atom_vs_red_ratio / cooperative_epilogue_overlap |
| 新 HTML 节 | 2：§21 atomic contention timeline、§22 cooperative epilogue overlap |
| 新 examples | 5：atom_histogram / atom_reduction_smem / cluster_cooperative_epilogue / atom_cas_spinlock / red_min_max |
| 新 tutorials | 5：chapters 22-26 |
| Phase 1-5 兼容 | 既有 example 不破；atomic config 默认仅添加新字段 |

---

## 2. 架构总图与模块改动

### 2.1 数据流变化

**gmem atomic 流程**：
```
warp issue atom.global.add %old, [addr], %val
  ├─ functional：每 lane 串行 atomic-update gmem[addr] → return old
  ├─ 计算 line_addr = addr / 128
  ├─ L1 atomic 路由 (Phase 6 加)：每 lane 转发到 L2 atomic ALU
  ├─ L2.atomic_op(line_addr, op, val_per_lane, sm_id, now):
  │     pool.enqueue(line_addr, request) — 入 per-line atomic queue
  │     完成 cycle = max(arrival, queue_head_complete + atomic_op_latency)
  │     更新 queue_head_complete
  │     return completion_at + L2_install_latency
  └─ scoreboard 标 dst 寄存器 (atom 才有 dst；red 没有) ready_at = max(per-lane completion)
```

**smem atomic 流程**：
```
warp issue atom.shared.add %old, [smem_addr], %val
  ├─ 计算 per-lane addr → bank conflict degree（复用 Phase 1）
  ├─ functional：每 lane 串行 atomic-update smem[cta_id][offset]
  ├─ latency = smem_latency + bank_conflict_degree * atomic_op_extra_latency
  └─ scoreboard 标 dst (atom only) ready_at
```

**Cluster TMA store 流程**：
```
warp issue cp.async.bulk.tensor.2d.global.shared::cluster [gmem_dst], [smem_src]
  ├─ 解 smem_src 指针：rank = (ptr >> 24) & 0xFF；offset = ptr & 0xFFFFFF
  ├─ target_cta = cluster_id * cluster_size + rank
  ├─ functional：do_bulk_store_2d(gmem, smem, cta_id=target_cta, smem_src=offset, desc)
  ├─ 入 BulkStoreQueue（与 Phase 4 同套 queue / commit_group / wait_group）
  └─ Phase 4 的 cp.async.bulk.commit_group / wait_group N 直接复用
```

### 2.2 关键不变量（继承 Phase 1-5）

- **Functional vs timing 分离**：atomic 数值在 numpy / Python int 上立即算；timing 只管 cycle
- **Trace 防火墙**：`AtomicEvent` 经 Recorder
- **API 不变**：`gpusim.run(...)` 签名不变
- **L2 atomic queue 是 L2 内部细节**：L1 不感知；SubCore 不感知；仅 L2.atomic_op 实现 + Recorder 看见

### 2.3 模块拓扑

```
gpusim/core/
├── cache/
│   └── l2.py                        MODIFY: + atomic_op + L2AtomicQueue per line
├── atomic.py                        NEW: AtomicOp helpers + L2AtomicQueue dataclass
├── smem.py                          MODIFY: + atomic_op (per-cta) helpers
├── tma_store.py                     MODIFY: do_bulk_store_2d 接受 cluster 编码 smem_src
├── exec.py                          MODIFY: GlobalMemory + SharedMemory atomic_op helpers (functional)
├── sub_core.py                      MODIFY: + atom.{global,shared}.* + red.{global,shared}.* + cluster TMA store decode
└── functional_units.py              MODIFY: + classify atom/red ops → LSU

gpusim/frontend/parser.py            MODIFY: + atom.{global,shared}.<op>.<ty> + red.{global,shared}.<op>.<ty>

gpusim/config/
├── schema.py                        MODIFY: CacheConfig + atomic_op_latency / atomic_queue_capacity / smem_atomic_op_extra_latency
└── default_hopper.yaml              MODIFY: cache: + 3 fields

gpusim/trace/
├── events.py                        MODIFY: + AtomicEvent
├── recorder.py                      MODIFY: + atomic method
└── writer.py                        MODIFY: + atomic.parquet writer

gpusim/analysis/metrics.py           MODIFY: + 4 metrics
gpusim/viz/                          MODIFY: + 2 HTML 节 (§21/§22) + Perfetto atomic track
gpusim/api.py                        MODIFY: + atomic_events_df + atomic_metrics + atomic_summary
```

### 2.4 Phase 1-5 carry-over

仅当 Phase 6 example 真撞到才修：
- Phase 5 deferred 项目：`m64n32k16` wgmma —— `cluster_cooperative_epilogue` 用 `m64n128k16`（既有）+ cluster 拆分维度，**不需要** m64n32k16
- 早 phases deferred items：`0f` literal、IPDOM 等仍然不阻塞

### 2.5 边界原则

1-7 (从 Phase 1-5 继承)
8. **Atomic functional 在 InstrExecutor / GlobalMemory.atomic_op 层**：返回 old + 写 new
9. **Atomic timing 在 L2.atomic_op 层**：per-line queue，全 SM 共享 L2 queue 视图

---

## 3. PTX 子集扩展 + IR 改动

### 3.1 新增指令

| 指令 | 用途 |
|---|---|
| `atom.global.<op>.<ty> %old, [%addr], %val` | gmem atomic RMW；返回 old |
| `atom.shared.<op>.<ty> %old, [%addr], %val` | smem atomic RMW；返回 old |
| `red.global.<op>.<ty> [%addr], %val` | gmem reduction（无 dst） |
| `red.shared.<op>.<ty> [%addr], %val` | smem reduction（无 dst） |
| `atom.global.cas.<ty> %old, [%addr], %expected, %val` | compare-and-swap 特殊形（3 src） |
| `cp.async.bulk.tensor.2d.global.shared::cluster [gmem_dst], [smem_src]` | cluster TMA store；smem_src 可 cluster 编码 |

`<op>` 集合：`{add, min, max, exch}` for `atom`（plus `cas` separately）；`{add, min, max}` for `red`。
`<ty>` 集合：`{u32, s32, f32}`。

注：Phase 4 已解析 `cp.async.bulk.tensor.2d.global.shared::cta`（store with `shared::cta`）；Phase 5 解析 cluster TMA load。Phase 6 加 cluster TMA store 解析（与 cta store 唯一区别是 op 字符串含 `shared::cluster` 而非 `shared::cta`）。

### 3.2 IR 改动

**无新 PtxType / 新 IR 节点**。复用 Reg / Imm。

### 3.3 Parser 改动

**atom 通用形 `atom.{global,shared}.<op>.<ty>`**：

```python
        if op.startswith("atom.global.") or op.startswith("atom.shared."):
            is_cas = ".cas." in op
            dst = self._parse_operand(self._type_from_op(op))
            self.eat("COMMA")
            self.eat("LBRACK")
            addr = self._parse_operand(PtxType.u64)
            self.eat("RBRACK")
            self.eat("COMMA")
            srcs: list = [addr]
            srcs.append(self._parse_operand(self._type_from_op(op)))
            if is_cas:
                self.eat("COMMA")
                srcs.append(self._parse_operand(self._type_from_op(op)))
            return [dst], srcs
```

**red 通用形**（无 dst）：

```python
        if op.startswith("red.global.") or op.startswith("red.shared."):
            self.eat("LBRACK")
            addr = self._parse_operand(PtxType.u64)
            self.eat("RBRACK")
            self.eat("COMMA")
            val = self._parse_operand(self._type_from_op(op))
            return [], [addr, val]
```

**Cluster TMA store** —— 复用 Phase 4 cp.async.bulk.tensor 的现有 parser 分支（Phase 4 add 时已用 `n_args = 3 if "mbarrier" in op else 2` 判断 load vs store；Phase 6 的 cluster TMA store 形式 `cp.async.bulk.tensor.2d.global.shared::cluster [gmem], [smem]` 也是 2 args，无 mbarrier，自然走 store 分支。无新 parser 改动。

### 3.4 FUSet.classify

```python
        if op.startswith("atom.") or op.startswith("red."):
            return FUKind.LSU
```

### 3.5 `_type_from_op` 兼容

现有 `_type_from_op` 找 `op.split(".")` 的最后一个已知 PtxType。`atom.global.add.u32` 最后一段 `u32` 命中。`atom.global.cas.s32` 同理。无需改。

---

## 4. Atomic 详细设计

### 4.1 L2AtomicQueue + L2Cache.atomic_op

`gpusim/core/atomic.py`（新）—— per-line atomic queue：

```python
@dataclass
class L2AtomicEntry:
    line_addr: int
    arrival_cycle: int
    completion_at: int
    sm_id: int
    op: str               # "add" | "min" | "max" | "exch" | "cas"
    op_kind: str          # "atom" | "red"


class L2AtomicQueue:
    """Per-line atomic FIFO. Multiple SMs hitting the same line serialize.
    Each atomic op takes atomic_op_latency cycles after the previous one finishes."""

    def __init__(self, n_slots: int = 32):
        self.n_slots = n_slots
        self._queues: dict[int, list[L2AtomicEntry]] = {}

    def enqueue(self, *, line_addr: int, sm_id: int, op: str, op_kind: str,
                  arrival: int, atomic_op_latency: int,
                  l2_hit_latency: int) -> int:
        q = self._queues.setdefault(line_addr, [])
        q = [e for e in q if e.completion_at > arrival]
        prev_done = q[-1].completion_at if q else 0
        start = max(arrival + l2_hit_latency, prev_done)
        completion = start + atomic_op_latency
        entry = L2AtomicEntry(
            line_addr=line_addr, arrival_cycle=arrival, completion_at=completion,
            sm_id=sm_id, op=op, op_kind=op_kind,
        )
        q.append(entry)
        self._queues[line_addr] = q
        return completion

    def queue_depth(self, line_addr: int, now: int) -> int:
        q = self._queues.get(line_addr, [])
        return sum(1 for e in q if e.completion_at > now)
```

`L2Cache.atomic_op(line_addr, sm_id, op, op_kind, now) -> int`：
- 复用 existing tag lookup (touch line if hit, fetch on miss similar to write_through path)
- 调 `self._atomic_queue.enqueue(...)` 计算 completion
- 不区分 hit/miss for atomic latency in this simulation（atomic 总是先把 line "锁住"）—— 真机 atomic miss 会先 fetch；为简化 Phase 6 当 atomic always hits（spec §11 标注简化）
- Recorder.atomic 事件注入 line_addr / op / latency

### 4.2 GlobalMemory + SharedMemory atomic_op (functional)

`gpusim/core/exec.py`：

```python
class GlobalMemory:
    def atomic_op(self, addr: int, op: str, val, ty: PtxType):
        if ty is PtxType.f32:
            old = self.load_f32(addr)
            new = self._apply_op_f32(op, old, val)
            self.store_f32(addr, new)
            return old
        old = self.load_u32(addr)
        new = self._apply_op_int(op, old, val, ty)
        self.store_u32(addr, new)
        return old

    @staticmethod
    def _apply_op_int(op: str, old: int, val, ty) -> int:
        if op == "add": return (old + int(val)) & 0xFFFFFFFF
        if op == "min": return min(old, int(val))
        if op == "max": return max(old, int(val))
        if op == "exch": return int(val) & 0xFFFFFFFF
        if op == "cas":
            expected, new = val
            return new if old == expected else old
        raise ValueError(op)
```

`SharedMemory.atomic_op(cta_id, offset, op, val, ty)` 同模式但操作 `_cta[cta_id]`。

### 4.3 SubCore atom / red 路由

In `_issue`:

```python
        if op.startswith("atom.global.") or op.startswith("atom.shared.") \
                or op.startswith("red.global.") or op.startswith("red.shared."):
            is_global = ".global." in op
            is_atom = op.startswith("atom.")
            op_name = op.split(".")[2]
            ty = instr.type
            for lane in range(32):
                if not (w.fn_state.active_mask >> lane) & 1: continue
                t = w.fn_state.threads[lane]
                addr = t.get_u64(instr.src[0].name)
                val = self.executor._read(t, instr.src[1], ty)
                if op_name == "cas":
                    expected = val
                    new_val = self.executor._read(t, instr.src[2], ty)
                    val_passed = (expected, new_val)
                else:
                    val_passed = val
                if is_global:
                    old = self.executor.gmem.atomic_op(addr, op_name, val_passed, ty)
                else:
                    old = self.smem.atomic_op(w.cta_id, addr, op_name, val_passed, ty)
                if is_atom:
                    self.executor._write(t, instr.dst[0], old, ty)
            now = ...
            if is_global:
                lines = sorted(set(int(t.get_u64(instr.src[0].name)) // 128
                                     for lane in range(32)
                                     if (w.fn_state.active_mask >> lane) & 1
                                     for t in [w.fn_state.threads[lane]]))
                max_completion = now
                for ln in lines:
                    c = self.l2.atomic_op(line_addr=ln, sm_id=getattr(self, "sm_id", -1),
                                            op=op_name, op_kind="atom" if is_atom else "red",
                                            now=now)
                    max_completion = max(max_completion, c)
                completion = max_completion
            else:
                bank_conflict = ...   # reuse Phase 1
                completion = now + self.cfg.fu.smem_latency + bank_conflict * self.cfg.cache.smem_atomic_op_extra_latency
            if self.recorder is not None:
                self.recorder.atomic(
                    cycle=now, sm_id=getattr(self, "sm_id", -1),
                    warp_id=w.warp_id, kind="ATOM" if is_atom else "RED",
                    op=op_name, space="global" if is_global else "shared",
                    line_addr=ln if is_global else addr,
                    latency=completion - now,
                )
            if is_atom:
                w.scoreboard.mark_write(instr.dst[0].name, completion, origin="atomic")
            w.stack.update_top_pc(w.stack.top().pc + 1); w.stack.maybe_pop()
            return
```

### 4.4 CacheConfig 新增字段

```python
@dataclass
class CacheConfig:
    # ... existing ...
    atomic_op_latency: int = 10
    atomic_queue_capacity: int = 32
    smem_atomic_op_extra_latency: int = 4
```

---

## 5. Cluster TMA Store + Cooperative Epilogue

### 5.1 cp.async.bulk.tensor.2d.global.shared::cluster 执行语义

**Functional**：
1. 解 gmem descriptor (从 handle lookup TmaDescriptor)
2. 解 smem_src 指针：
   - 若 op 含 `shared::cluster` 且 `w.cluster_id >= 0` 且 `cluster_size > 1`：
     - rank = (smem_src_ptr >> 24) & 0xFF
     - smem_offset = smem_src_ptr & 0xFFFFFF
     - source_cta = w.cluster_id * cluster_size + rank
   - 否则：source_cta = w.cta_id, smem_offset = smem_src_ptr
3. `do_bulk_store_2d(gmem, smem, cta_id=source_cta, smem_src=smem_offset, desc)`
4. 入 BulkStoreQueue（与 Phase 4 同套）

**Timing**：与 Phase 4 cp.async.bulk.tensor.global.shared::cta 同：
- n_lines = ceil(tile_bytes / 128)
- completion_at = now + max(8, n_lines * `bulk_store_latency_per_line`)

### 5.2 SubCore 路由

In `gpusim/core/sub_core.py`，既有 `cp.async.bulk.tensor.` 分支扩展：

```python
        if op.startswith("cp.async.bulk.tensor."):
            smem_reg = instr.src[0]
            desc_reg = instr.src[1]
            mbar_reg = instr.src[2] if len(instr.src) > 2 else None
            smem_ptr = w.fn_state.threads[0].get_u64(smem_reg.name)
            handle = w.fn_state.threads[0].get_u64(desc_reg.name)
            desc = self.tma_descriptor_pool.lookup(handle)

            cluster_size = getattr(w.executor, "cluster_size", 1)
            is_cluster = ("shared::cluster" in op
                          and w.cluster_id >= 0
                          and cluster_size > 1)
            is_load = "global.shared" in op and "mbarrier" in op
            is_store = "global.shared" in op and "mbarrier" not in op

            if is_cluster:
                rank = (int(smem_ptr) >> 24) & 0xFF
                smem_offset = int(smem_ptr) & 0xFFFFFF
                target_cta = w.cluster_id * cluster_size + rank
            else:
                smem_offset = int(smem_ptr)
                target_cta = w.cta_id

            if is_load:
                from gpusim.core.tma import do_bulk_copy_2d
                tx_bytes = do_bulk_copy_2d(
                    gmem=self.executor.gmem, smem=self.smem,
                    cta_id=target_cta, smem_dst=smem_offset, desc=desc,
                )
                # ... mbarrier arrive_tx (Phase 5) ...
            elif is_store:
                from gpusim.core.tma_store import do_bulk_store_2d
                tx_bytes = do_bulk_store_2d(
                    gmem=self.executor.gmem, smem=self.smem,
                    cta_id=target_cta, smem_src=smem_offset, desc=desc,
                )
                # Push to BulkStoreQueue (Phase 4) ...
            ...
```

### 5.3 cooperative_epilogue example 设计

`examples/cluster_cooperative_epilogue/kernel.ptx`:

4-CTA cluster, M=64 N=128 K=16 fp16 matmul:
1. **Load A (CTA 0 broadcasts)**: CTA 0 TMA load A (64×16 fp16) into local smem; cluster mbarrier signals; all CTAs `mapa.shared::cluster` to read CTA 0's A
2. **Load B (each CTA loads its slice)**: CTA r loads B slice for cols [r*32, r*32+32) into local smem
3. **wgmma**: each CTA does `wgmma.mma_async.sync.aligned.m64n128k16` with full B... actually adapt: each CTA computes 32-col slice via m64n32k16 (if simulator supports) OR simplified version skips wgmma
4. **D writeback to cluster smem**: each CTA writes its 64×32 D slice to its local smem_D
5. **Cluster cooperative store**: CTA 0 issues 4 cluster TMA stores (one per cluster CTA's smem_D) to gmem 4 different offsets

If `m64n32k16` not supported, fallback simplified version skips wgmma and tests cluster TMA store mechanism alone.

### 5.4 BulkStoreQueue 复用 / Stall token 不变

Cluster TMA store 复用 Phase 4 `BulkStoreQueue` per warp-group。同样的 `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group N` 控制 in-flight cluster store。

无新 stall token（Phase 4 既有 `BULK_STORE_QUEUE_FULL` + `BULK_STORE_WAIT` 已覆盖）。

---

## 6. Trace + 分析 + 可视化

### 6.1 完整事件清单

| 类别 | 事件 | 频率 |
|---|---|---|
| Phase 1 (8) | WARP_STATE, INSTR_ISSUE, SMEM_ACCESS, GMEM_ACCESS, DIV_PUSH/POP, BAR_REACH/RELEASE, CTA_LAUNCH/RETIRE | 高 |
| Phase 2 (3) | L1_ACCESS, L2_ACCESS, HBM_ACCESS | 中-低 |
| Phase 3 (4) | MmaEvent, WgmmaEvent, TmaEvent, MbarrierEvent | 中-低 |
| Phase 4 (3) | CtaDispatchEvent, L2MshrEvent, BulkStoreEvent | 中-低 |
| Phase 5 (2) | ClusterDispatchEvent, ClusterBarrierEvent | 中-低 |
| **Phase 6 (1)** | **AtomicEvent** | 中 |

### 6.2 Phase 6 新事件 schema

```python
@dataclass(frozen=True)
class AtomicEvent:
    cycle: int
    sm_id: int
    warp_id: int
    kind: str                  # "ATOM" | "RED"
    op: str                    # "add" | "min" | "max" | "exch" | "cas"
    space: str                 # "global" | "shared"
    line_addr: int             # gmem: line_addr; smem: byte offset
    latency: int               # cycles from issue to completion
    n_lanes: int = 1
    queue_depth_before: int = 0
```

### 6.3 现有 events 不变

`SMEM_ACCESS` / `GMEM_ACCESS` 不重复事件；atomic 通过 `AtomicEvent` 单独记录。

### 6.4 新增分析指标（4 个）

| 函数 | 输出 | 教学用途 |
|---|---|---|
| `atomic_throughput_per_line(atomic_df, total_cycles)` | DataFrame[line_addr, atomic_count, throughput] | Hot spot 识别 |
| `atomic_serialization_overhead(atomic_df, total_cycles)` | scalar 0..1 | atomic 因 L2 queue 串行多花 cycles 比例 |
| `atom_vs_red_ratio(atomic_df)` | dict {"atom": pct, "red": pct} | atom vs red 占比 |
| `cooperative_epilogue_overlap(bulk_store_df, mma_df)` | scalar 0..1 | wgmma + cluster TMA store overlap |

### 6.5 HTML 报告新增节（2 节）

| 节 | 内容 |
|---|---|
| **§21 Atomic contention timeline** | per-line atomic queue depth 时序 + 表格：每 line atomic 数 / 平均 latency / hot lines |
| **§22 Cooperative epilogue overlap** | wgmma + cluster TMA store Plotly Gantt + overlap ratio |

### 6.6 Result API 扩展

```python
@dataclass
class Result:
    # ... Phase 1-5 fields ...

    @property
    def atomic_events_df(self) -> pd.DataFrame: ...

    @property
    def atomic_metrics(self) -> dict: ...

    def atomic_summary(self) -> str:
        m = self.atomic_metrics
        if not m or m.get("count", 0) == 0:
            return "no atomic ops"
        return (f"atomic count={m['count']} / "
                 f"hot line peak depth={m['peak_queue_depth']} / "
                 f"serial overhead={m['serialization_overhead']*100:.1f}%")
```

### 6.7 Perfetto 集成

| 事件 | Perfetto track | 视觉 |
|---|---|---|
| `AtomicEvent("ATOM")` | global "Atomic" track | 红色 instant + line_addr |
| `AtomicEvent("RED")` | global "Atomic" track | 橙色 instant |

### 6.8 Parquet 落盘

新增 1 个 parquet：`atomic.parquet`。

---

## 7. 测试策略

### 7.1 单元测试

| 模块 | 关键测试 |
|---|---|
| `core/atomic` | L2AtomicQueue: enqueue arrival vs prev_completion；多 line 不互相串行；queue_depth |
| `core/cache/l2` | atomic_op routes through L2AtomicQueue；同 line 多 SM 串行 latency；不同 line 并行 |
| `core/exec` | GlobalMemory.atomic_op all 5 ops × 3 dtypes；cas 正确语义；SharedMemory.atomic_op |
| `core/sub_core` | atom.global / atom.shared / red.global / red.shared 路由；scoreboard 标 atom dst（red 不标） |
| `core/tma_store` | do_bulk_store_2d 接受 cluster 编码 smem_src 后 read 正确 CTA's smem |
| `frontend/parser` | atom.global.add.u32, atom.global.cas.s32 (3 src), red.global.add.f32, atom.shared.* |
| `analysis/metrics` | 4 个新指标 fixture |
| `viz/html_report` | §21 / §22 在 atomic / bulk store events 存在时插入 |
| `config/loader` | CacheConfig 3 个新字段默认 |

### 7.2 Functional Parity（numpy）

5 个新 example：
- `atom_histogram`：32 thread × N CTA atomic.add 一个 random bin idx 的 counter；numpy histogram 对照，rtol=0
- `atom_reduction_smem`：每 thread atomic.add 1 到一个 smem 计数；assert OUT == n_threads
- `cluster_cooperative_epilogue`：4-CTA cluster 数据正确性 check
- `atom_cas_spinlock`：N thread CAS-loop critical section；assert counter == N
- `red_min_max`：N elements；与 numpy.min/max 对比

Phase 1-5 example 全部继续通过。

### 7.3 Reference Fixture

`tests/reference/data/` 加 5 个 stub。容忍度：
- `atomic_throughput_per_line` ±15%
- `serialization_overhead` ±10%
- `cooperative_epilogue_overlap` ±15%

`gen_reference.py` SUPPORTED_KERNELS 加 5 项。

### 7.4 微基准

`tests/microbench/test_phase6_facts.py`：

```
- 32 thread atomic.add 同一 line：cycles 比 32 thread 各自 atomic 不同 line 慢 ≥ 5×
- atomic_reduction_smem 在 single bank: cycles ≥ 32 × atomic_op_extra_latency
- atom_histogram 高碰撞 (4 bins) vs 低碰撞 (1024 bins) cycles 比 ≥ 3×
- cluster_cooperative_epilogue 中 cluster TMA store wait_group 0 可见 in-flight cycles > 0
- atom.add.u32 与 red.add.u32 cycles 几乎相同
```

### 7.5 Phase 1-5 兼容性

复用 `tests/parity/test_phase1_4_examples_unchanged.py` → 改名 `test_phase1_5_examples_unchanged.py`：
- 跑 Phase 1-5 全部 20 个 example
- atomic 默认未启用，cycles 浮动 ≤ 5%

### 7.6 Memory budget 测试

`tests/microbench/test_phase6_runtime.py`（@pytest.mark.slow）：
- atom_histogram < 30 秒
- cluster_cooperative_epilogue < 60 秒

---

## 8. 项目结构改动

### 8.1 目录变化

```
gpusim/core/
├── atomic.py                        NEW
├── cache/l2.py                      MODIFY
├── smem.py                          MODIFY
├── exec.py                          MODIFY
├── tma_store.py                     MODIFY
├── sub_core.py                      MODIFY
└── functional_units.py              MODIFY

gpusim/frontend/parser.py            MODIFY
gpusim/config/{schema.py, default_hopper.yaml}    MODIFY
gpusim/trace/{events.py, recorder.py, writer.py}  MODIFY
gpusim/analysis/metrics.py           MODIFY
gpusim/viz/                          MODIFY
gpusim/api.py                        MODIFY

tests/unit/core/test_{atomic,l2_atomic,exec_atomic,sub_core_atomic}.py    NEW
tests/unit/frontend/test_parser_phase6.py    NEW
tests/unit/analysis/test_phase6_metrics.py    NEW
tests/unit/viz/test_html_report_phase6.py    NEW
tests/parity/test_{atom_histogram,atom_reduction_smem,cluster_cooperative_epilogue,atom_cas_spinlock,red_min_max}.py    NEW
tests/parity/test_phase1_5_examples_unchanged.py    RENAME from phase1_4
tests/microbench/test_{phase6_facts,phase6_runtime}.py    NEW
tests/reference/data/{atom_histogram,atom_reduction_smem,cluster_cooperative_epilogue,atom_cas_spinlock,red_min_max}.ref.json    NEW
```

### 8.2 配置 yaml 变化

`default_hopper.yaml` `cache:` 节加：

```yaml
cache:
  # ... existing ...
  atomic_op_latency: 10
  atomic_queue_capacity: 32
  smem_atomic_op_extra_latency: 4
```

### 8.3 依赖

无新依赖。

---

## 9. 教学示例与讲义

### 9.1 5 个新 example

| # | Example | grid / config | 教学意图 |
|---|---|---|---|
| 1 | **atom_histogram** | grid=(8,1,1), block=(32,1,1), n_bins varies | gmem `atom.global.add.u32`；高竞争 vs 低竞争 cycles 对比；演示 L2 atomic ALU 串行 |
| 2 | **atom_reduction_smem** | grid=(1,1,1), block=(128,1,1) | smem `atom.shared.add.u32`；演示 smem bank conflict + atomic latency 复合 |
| 3 | **cluster_cooperative_epilogue** | grid=(4,1,1), cluster_size=4, block=(128,1,1) | wgmma + cluster TMA store；闭合 Phase 5 cluster 故事 |
| 4 | **atom_cas_spinlock** | grid=(8,1,1), block=(32,1,1) | gmem `atom.global.cas.u32`；CAS retry 模式 |
| 5 | **red_min_max** | grid=(8,1,1), block=(32,1,1) | gmem `red.global.{min,max}.f32`；red vs atom 硬件 cost 差 |

每目录：`{kernel.ptx}` + `reference.py` + `run.py` + `README.md` + `__init__.py`。

### 9.2 5 章新讲义（chapters 22–26）

| # | 标题 | 关联 example |
|---|---|---|
| 22 | gmem atomic 与 L2 ALU 串行 | atom_histogram |
| 23 | smem atomic 与 bank conflict | atom_reduction_smem |
| 24 | Cluster TMA store 与 cooperative epilogue | cluster_cooperative_epilogue |
| 25 | CAS 与 lock-free pattern | atom_cas_spinlock |
| 26 | red vs atom：reduction 原语的硬件区别 | red_min_max |

每章固定栏目：**看模拟器** / **改一改** / **真机对照**。

---

## 10. 与 Phase 1-5 兼容性

### 10.1 不会破坏的部分

| 维度 | 状态 |
|---|---|
| `gpusim.run(...)` 函数签名 | 不变 |
| Phase 1-5 example PTX | 不动 |
| Phase 1-5 parity 测试 | 全部继续通过 |
| Result 旧字段 | 不变 |
| HTML 报告 §1–§20 | 位置 + 内容不变 |
| Perfetto 既有 track | 不变 |
| Stall token 既有 18 类 | 不变 |
| 新依赖 | 无 |
| Trace 既有事件 schema | 不变 |
| BulkStoreQueue / commit_group / wait_group | 复用 Phase 4 |
| Cluster + dsmem | Phase 5 路径不变 |

### 10.2 会变的部分

| 维度 | 变化 |
|---|---|
| Cycle 数（Phase 1-5 example） | 不变 |
| Stall 直方图 | 不变 |
| HTML 报告 | 多 2 节（§21, §22） |
| Perfetto | 新 track（global "Atomic"） |
| Trace parquet | 多 1 文件 |
| Result | 多 1 property + `atomic_metrics` + `atomic_summary()` |
| 配置 yaml | `cache:` 节加 3 字段（缺省读为默认） |
| `CacheConfig` | 加 3 字段 |
| `L2Cache` | 加 atomic_op 方法 |
| `GlobalMemory` / `SharedMemory` | 加 atomic_op 方法 |
| `do_bulk_store_2d` | 接受 cluster 编码 smem_src |

### 10.3 Phase 4-5 既有路径兼容

- Phase 4 TMA store (`shared::cta`) 不变
- Phase 5 cluster TMA load 不变
- Phase 6 cluster TMA store 走同 SubCore 分支，按 op 字符串 `is_load` vs `is_store` + `is_cluster` 分发

---

## 11. 显式不在范围内（Phase 6）

- **Cluster atomics**：`atom.shared::cluster.*`（Phase 7+）
- **64-bit atomic**：`atom.global.add.u64` 等
- **fp16/bf16 atomic**：真机 Hopper 支持，模拟器不实现
- **inc/dec/and/or/xor atomic ops**
- **atomic miss 路径**：所有 atomic 当 L2 hit
- **atomic ordering / fence**：默认 acq_rel 语义
- **Multi-stream / 多 kernel 并发**：Phase 7
- **多 GPU、NVLink、NCCL**：Phase 8
- **GPU-wide atomic**（`atom.system.*`）：Phase 8

---

## 12. 已知近似与简化

- **Atomic 总是 L2 hit**：atomic line 假定常驻 L2，不模拟 atomic 触发的 HBM fetch
- **L2 atomic ALU 数量**：Phase 6 假定每 line 一个 ALU port；真机 ALU 数量有限，跨 line contention 未建模
- **smem atomic 简化**：复用 bank conflict + atomic_op_extra_latency = 4
- **f32 atomic.add 顺序敏感**：simulator lane 顺序固定，不模拟 race
- **CAS 单尝试**：simulator atomic.cas 单 op 完成；retry 由用户代码循环
- **red 与 atom 同 latency**：真机 red 略快（无返回路径）；spec 标注简化
- **Cluster TMA store latency 同 cta TMA store**：不模拟跨 SM 数据汇集额外 cost

---

## 13. Phase 6 实施里程碑

| 里程碑 | 交付 |
|---|---|
| **M1** | Frontend + 配置：parser (atom/red 全 8 op × 2 space × 3 dtype + cas 3-src 形)、CacheConfig + 3 字段、yaml + loader、AtomicEvent + recorder + parquet。无运行时行为变化 |
| **M2** | smem atomic：SharedMemory.atomic_op + SubCore atom/red shared 分支 + bank conflict 复用 + atom_reduction_smem example |
| **M3** | gmem atomic：GlobalMemory.atomic_op + L2AtomicQueue + L2.atomic_op + SubCore atom/red global 分支 + atom_histogram + atom_cas_spinlock + red_min_max examples |
| **M4** | Cluster TMA store：do_bulk_store_2d 接受 cluster 编码 smem_src + SubCore 路由扩展 + cluster_cooperative_epilogue example |
| **M5** | Trace + 分析 + viz + 收尾：4 metrics + 2 HTML 节 + Perfetto + Result API + 5 章新讲义 + Phase 1-5 兼容性测试 + Phase 6 microbench + reference fixture + README v6 + tag `phase6-complete` |

预估总任务数：**32-36**。

每 milestone 后打 git tag (`M{1..5}-phase6-complete`)。

---

## 14. 设计协作记录

本文档由用户与 Claude（Opus 4.7, 1M context）通过 `superpowers:brainstorming` 流程逐节确认产出。所有关键决策均经用户显式确认（A/B/C 选择或"OK"回复）。

下一步：交由 `superpowers:writing-plans` 产出可执行的实施计划。
