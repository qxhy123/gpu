# gpusim Phase 3 — Tensor Core + 多精度 + wgmma + TMA 设计文档

**日期**：2026-05-08
**状态**：设计阶段（待实现）
**作者**：与 Claude 协同 brainstorm
**前置依赖**：Phase 1 完成（tag `phase1-complete`）+ Phase 2 完成（tag `phase2-shipped`，HEAD `8ef4204`）
**范围**：仅 Phase 3。Phase 4+ 仅作为愿景列出。

---

## 1. 愿景与 Phase 3 范围

### 1.1 项目背景

Phase 1 + Phase 2 交付了一个单 SM、cycle-approximate、带完整 cache 层级（L1/L2/HBM）的教学 GPU 模拟器。但所有计算路径只走 CUDA Core（FP32/INT32 标量 ALU），无法演示 AI infra 的实际加速来源——**Tensor Core**。

Phase 3 把模拟器从"标量 GPU"扩展为"AI 加速器"：sync `mma`（3 shape × 6 精度）+ Hopper `wgmma`（async warp-group MMA）+ TMA-lite（async 张量搬运）+ 4 个新 example + 4 章新讲义。

### 1.2 Phase 3 一句话目标

> 在 Phase 1+2 的 SM/cache/HBM 之上加入 Tensor Core 子系统：sync `mma`（3 shape × 6 精度）+ `wgmma`（warp_id // 4 隐式分组、async + WgmmaQueue + commit/wait_group）+ TMA-lite（`cp.async.bulk.tensor.2d` + mbarrier）+ 4 个新 example。让学生能动手感受 FP32 → FP16 → FP8 的速度跃迁、accumulator 精度的必要性、以及 wgmma + TMA 的真实生产 pipeline。

### 1.3 路线图回顾

| Phase | 范围 | 状态 |
|---|---|---|
| Phase 1 | 单 SM、cycle-approximate、PTX 子集、shared/global memory 无 cache | ✅ 已完成 |
| Phase 2 | L1/L2 cache（tag-precise）+ HBM channel/bank/row buffer | ✅ 已完成 |
| **Phase 3** | **Tensor Core (sync mma) + 6 精度 + wgmma + TMA-lite + 4 example** | **本文档** |
| Phase 4 | 多 SM、CTA→SM 调度、L2 跨 SM 共享 | 后续 |
| Phase 5 | 多 GPU、NVLink、NCCL collective | 后续 |

### 1.4 已锁定决策

| 维度 | 决策 |
|---|---|
| 范围 | sync mma + wgmma + TMA-lite（B 选项，full） |
| 精度集 | FP16 + BF16 + FP8 (E4M3 + E5M2) + TF32 + INT8 + INT32 accum |
| 非原生 dtype | 加 `ml_dtypes>=0.4` 依赖（提供真实位宽 storage） |
| sync mma shape | 一形状/精度，反映 K-vs-bitwidth：m16n8k8 (TF32) / m16n8k16 (FP16/BF16) / m16n8k32 (FP8/INT8) |
| wgmma shape | 一 canonical/精度，n=128：m64n128k8 (TF32) / m64n128k16 (FP16/BF16) / m64n128k32 (FP8/INT8) |
| Async 数据搬运 | TMA-lite：`cp.async.bulk.tensor.2d` + mbarrier（Hopper 风格） |
| Warp-group | 隐式分组：`warp_id // 4`，`Warp` 加 `warp_group_id` 字段 |
| Tensor Core FU | 新加 `FUKind.TC`，每 sub-core 1 个；与 ALU 并行；新 `WgmmaQueue` per warp-group |
| 新 stall token | `WGMMA_QUEUE_FULL` + `WGMMA_WAIT`（Phase 2 含 MSHR_FULL 共 12 类 → Phase 3 加 2 类 → 14 类） |
| `gpusim.tma_desc` 伪指令 | 替代真机 `tensormap.*` 操作链构造 descriptor |
| Lane → element layout | fictional 简化（教学用，不与真机 PTX 一一对应） |
| 新 examples | 4 个：tc_matmul_precisions（6 PTX）、mixed_accum（2 PTX）、wgmma_basic、wgmma_async_pipeline |
| 新 tutorials | 4 章（12–15） |
| Phase 1/2 兼容 | parity 不变；如 Phase 1 deferred bug（hex `e`、`0f` literals、IPDOM）阻塞 Tensor Core kernel 才修 |
| 顺便修 | `Instr.type` 改为 `Optional[PtxType]`（清 Phase 1 的 b32 sentinel 谎言） |

---

## 2. 架构总图与模块改动

### 2.1 数据流变化

**sync mma**（同步、warp-wide）：
```
warp issue mma.m16n8k16.f16 %d, %a, %b, %c
  ├─ functional：从 32 lane 寄存器读 a/b/c → numpy + ml_dtypes 算 d → 分发回 32 lane
  └─ timing：reserve TC FU（occupancy=1），scoreboard 标 %d ready_at = now + tc_mma_latency
```

**wgmma**（async、warp-group-wide）：
```
4 warp 中第 i 个到达 wgmma.mma_async.sync.aligned.m64n128k16.f16
  ├─ warp-group sync：所有 4 warp 都到达此 PC 才 issue
  ├─ functional（warp 0 代理）：解 A_desc/B_desc → smem tile → numpy + ml_dtypes 算 64×128 → 分发到 128 lane
  ├─ 入 WgmmaQueue：(commit_group_id, completion_at, dst_regs)
  ├─ scoreboard 标 %d uncommitted（ready_at = ∞，wait_group 时再下推）
  └─ TC FU reserve occupancy = 4 cycle（warp-group 同步开销）

后续 wgmma.commit_group：把当前 in-flight 划成新 group
后续 wgmma.wait_group N：阻塞至 in-flight group 数 ≤ N；触发时下推 dst_regs ready_at
```

**TMA-lite**：
```
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
  ├─ 解析 tensor descriptor (gmem_base, dim_x, dim_y, stride, elem_bytes)
  ├─ functional：复制 tile gmem → smem（zero-copy）
  ├─ 异步：mbarrier 注册 pending_tx (bytes, completion_at)
  └─ SM 每 cycle tick mbarrier；completion_at 到达 → arrived_count++
```

### 2.2 关键不变量

- **Functional 与 timing 分离**：mma/wgmma 数值用 numpy + ml_dtypes 立即算；TC FU + WgmmaQueue 只管 cycle
- **warp 0 代理 wgmma**：硬件上 4 warp 协同，模拟器上 "warp 0 集中算 + 结果分发到 128 lane"，语义等价
- **Trace 仍是防火墙**：mma/wgmma/tma/mbarrier 事件全经 Recorder
- **ml_dtypes 提供真实位宽 storage**：cache hit / HBM bytes 报表自动正确

### 2.3 模块拓扑

```
gpusim/core/
├── tensor_core/                 ← 新包
│   ├── __init__.py
│   ├── mma.py                   sync mma 执行（3 shape × 6 dtype）
│   ├── wgmma.py                 wgmma + WgmmaQueue
│   └── precision.py             dtype dispatch、ml_dtypes cast 帮手
├── tma.py                       ← TensorDescriptor + bulk copy 语义
├── mbarrier.py                  ← MbarrierPool 状态机
├── functional_units.py          + FUKind.TC
├── warp.py                      + warp_group_id 字段、wgmma_pending_pc
├── sub_core.py                  _issue 增 mma/wgmma/tma/mbarrier 路径
├── exec.py                      InstrExecutor 增对应指令
└── sm.py                        warp-group sync 协调 + mbarrier tick

gpusim/frontend/
├── ir.py                        + 8 个新 PtxType + RegGroup + TensorDescriptor + MbarrierHandle；Instr.type → Optional
├── lexer.py                     + COLONCOLON token；可选修 0f literal
└── parser.py                    + brace-list operands + mma/wgmma/cp.async.bulk.tensor.2d/mbarrier.* 解析

gpusim/config/
├── schema.py                    + TensorCoreConfig
└── default_hopper.yaml          + tensor_core 节

gpusim/trace/
├── events.py                    + MmaEvent / WgmmaEvent / TmaEvent / MbarrierEvent
├── recorder.py                  + 4 个新方法
└── writer.py                    + 4 个新 parquet 文件

gpusim/analysis/metrics.py       + 7 个新指标
gpusim/viz/                      + 4 个新 HTML 节 + Perfetto 新 track
gpusim/api.py                    + Result.tc_metrics + 4 个 events_df

pyproject.toml                   + ml_dtypes>=0.4 依赖
```

### 2.4 与 Phase 1+2 的 carry-over

仅当 Phase 3 example 真撞到才修：
- **`0f` PTX float literal**（Phase 1 deferred）：tc_matmul_precisions 用 numpy 构造数据，不在 PTX 写 float literal → 不阻塞
- **parser hex `e` digit**：tensor_core kernel 不太可能用 hex `e` 立即数 → 不阻塞
- **IPDOM 启发式**：wgmma_async_pipeline 有循环 + 多分支，**可能撞到**——预留 budget
- **INSTR_COMPLETE event**（Phase 1 deferred）：与 Phase 3 无关
- **Device class**（Phase 2 deferred）：与 Phase 3 无关，Phase 4 再做

### 2.5 边界原则

1. **Functional vs timing 分离** —— Phase 1+2 既有原则，继承
2. **Trace 是防火墙** —— Phase 1+2 既有原则，继承
3. **Tag-only / 数值 layer-bypass** —— Phase 2 既有原则；mma/wgmma 加新维度：register layout 是 fictional，但 functional 计算精确
4. **API 兼容** —— `gpusim.run(...)` 签名不变；Phase 1/2 example 参数不动

---

## 3. PTX 子集扩展 + IR 改动

### 3.1 新增指令（约 11 类 + 形状变种）

**Tensor Core 计算**：

| 指令 | 用途 |
|---|---|
| `mma.sync.aligned.m{M}n{N}k{K}.row.col.{dtypeD}.{dtypeA}.{dtypeB}.{dtypeC}` | sync 矩阵乘累加 |
| `wgmma.mma_async.sync.aligned.m64n128k{K}.{dtypeD}.{dtypeA}.{dtypeB}` | async warp-group MMA |
| `wgmma.fence.sync.aligned` | smem write → wgmma read 栅栏 |
| `wgmma.commit_group.sync.aligned` | 把 in-flight wgmma 划成新 group |
| `wgmma.wait_group.sync.aligned N` | 阻塞至 in-flight group 数 ≤ N |

**TMA + mbarrier**：

| 指令 | 用途 |
|---|---|
| `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [smem], [desc], [mbar]` | async 2D tensor copy gmem→smem |
| `mbarrier.init.shared::cta [mbar], expected_count` | 初始化 mbarrier |
| `mbarrier.arrive.shared::cta [mbar]` | 通知一次到达 |
| `mbarrier.try_wait.parity.shared::cta %p, [mbar], phase` | 检查 mbarrier 是否完成（不阻塞，返回 pred） |

**精度转换**（扩充 Phase 1 已有 `cvt`）：

| 指令 | 用途 |
|---|---|
| `cvt.{rnd}.{dst_ty}.{src_ty}` | 6 种新精度互转 |

**伪指令**（教学简化）：

| 指令 | 用途 |
|---|---|
| `gpusim.tma_desc %rd_handle, %rd_gmem_base, dim_x, dim_y, stride_y, elem_bytes` | 构造 TMA descriptor 句柄（替代真机 `tensormap.*` 链） |

> 注：Phase 3 **不**新增标量算术（`add.f16` 等）。新精度只在 mma 输入/输出 + cvt 边界出现。

### 3.2 新 PtxType 枚举值

```python
class PtxType(Enum):
    # Phase 1 既有
    s32 = "s32"; u32 = "u32"; s64 = "s64"; u64 = "u64"
    b32 = "b32"; b64 = "b64"; f32 = "f32"; pred = "pred"
    # Phase 3 新增
    f16   = "f16"
    bf16  = "bf16"
    e4m3  = "e4m3"
    e5m2  = "e5m2"
    tf32  = "tf32"
    s8    = "s8"
    u8    = "u8"
    s16   = "s16"
```

### 3.3 新 IR 节点

```python
@dataclass(frozen=True)
class TensorDescriptor:
    """Hopper TMA 2D descriptor (simplified — no swizzle, no multicast)."""
    gmem_base_reg: str
    dim_x: int
    dim_y: int
    stride_y: int
    elem_bytes: int


@dataclass(frozen=True)
class MbarrierHandle:
    """Pointer to mbarrier in shared memory."""
    smem_addr: int


@dataclass(frozen=True)
class RegGroup:
    """A `{reg0, reg1, ...}` operand group (e.g., mma matrix fragment)."""
    regs: tuple[Reg, ...]


# Operand union 扩展
Operand = Reg | Imm | RegGroup
```

### 3.4 Mma 解码器

`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 含 8 个 dotted modifier。Phase 1 的 `_type_from_op` 不够用，新加专门 decoder：

```python
@dataclass(frozen=True)
class MmaSpec:
    is_async: bool          # True for wgmma
    m: int; n: int; k: int
    layout_a: str           # "row" | "col"
    layout_b: str
    dtype_d: PtxType
    dtype_a: PtxType
    dtype_b: PtxType
    dtype_c: PtxType


def parse_mma_op(op: str) -> MmaSpec | None:
    """Decode a mma/wgmma opcode string. Returns None for non-mma."""
```

`InstrExecutor` 在分发 mma 指令时调 `parse_mma_op`；timing layer 也用它查 latency。

### 3.5 Lexer 改动

| 修改 | 原因 |
|---|---|
| 加 `COLONCOLON` token (`::`) | PTX namespace 分隔（`shared::cluster`、`mbarrier::complete_tx::bytes`） |
| 修 `0f<8 hex>` PTX float literal（Phase 1 deferred） | 实施期撞到再修 |

### 3.6 Parser 改动

1. **Brace-list operands** —— 在 mma/wgmma 操作数位置识别 `{reg, reg, ...}`，归到 `RegGroup`
2. **多 dotted-modifier opcode** —— 用 `parse_mma_op` 解码而非 Phase 1 的"取最后一段"启发式
3. **`gpusim.tma_desc` 伪指令** —— 新加专用 parser handler
4. **`COLONCOLON` 在 cp.async/mbarrier 路径** —— 把 `shared::cluster` 等当作 single namespace token sequence

### 3.7 IR 节点更新

`Instr.type` 由 `PtxType` 改为 `Optional[PtxType]`——清 Phase 1 的 b32 sentinel 谎言。mma 类指令 `Instr.type=None`，dtype 信息通过 `parse_mma_op(instr.op)` 取。

---

## 4. Tensor Core 详细设计

### 4.1 sync mma 执行语义

**Shape 总览**：

| Shape | M×N×K | 总输入元素 (A,B,C) | 输出元素 D | 每 lane 寄存器数 |
|---|---|---|---|---|
| m16n8k8 (TF32) | 16×8×8 | 128+64+128 | 128 | A:4, B:2, C:4, D:4 |
| m16n8k16 (FP16/BF16) | 16×8×16 | 256+128+128 | 128 | A:8, B:4, C:4, D:4 |
| m16n8k32 (FP8/INT8) | 16×8×32 | 512+256+128 | 128 | A:16, B:8, C:4, D:4 |

**Lane → element 映射（fictional 简化，spec §11 标注不与真机对应）**：

```
A[16][16] (FP16, m16n8k16):
  lane i (0..31), reg %aj (0..7): A[i / 2][(i % 2) * 8 + j]
B[16][8]:
  lane i, reg %bj (0..3): B[i / 2][(i % 2) * 4 + j]
D[16][8]:
  lane i, reg %dj (0..3): D[i / 2][(i % 2) * 4 + j]
```

**Functional 执行**（`gpusim/core/tensor_core/mma.py`）：

```python
def execute_mma(spec: MmaSpec, w: WarpFnState, dst, a, b, c):
    A = collect_matrix(w, a, spec.m, spec.k, spec.dtype_a)
    B = collect_matrix(w, b, spec.k, spec.n, spec.dtype_b)
    C = collect_matrix(w, c, spec.m, spec.n, spec.dtype_c)
    D = (A.astype(np.float32) @ B.astype(np.float32) + C.astype(np.float32))
    D = D.astype(_numpy_dtype_for(spec.dtype_d))   # ml_dtypes 整合
    distribute_matrix(w, dst, D)
```

**Timing**：mma 路由到 `FUKind.TC`，issue_occupancy=1，result latency = `tc_mma_latency`（默认 8 cycle）。scoreboard 标 dst 寄存器 ready_at = now + 8。

### 4.2 wgmma 执行语义

**Warp-group sync 协调**（在 `gpusim/core/sm.py`）：

类似 `bar.sync` 但作用域 = 同 `warp_group_id` 的 4 warp。`Warp` 加 `wgmma_pending_pc: int = -1` 字段。

```python
# 在 SM main loop barrier coordination 之后追加
by_wg: dict[int, list[Warp]] = {}
for w in active_warps:
    by_wg.setdefault(w.warp_group_id, []).append(w)
for wg_id, ws in by_wg.items():
    non_done = [w for w in ws if not w.finished]
    if non_done and all(w.wgmma_pending_pc >= 0 for w in non_done):
        executor_warp = non_done[0]
        execute_wgmma_for_group(executor_warp, ws, ...)
        for w in non_done:
            w.stack.update_top_pc(w.wgmma_pending_pc + 1)
            w.wgmma_pending_pc = -1
```

**Functional 执行**（`gpusim/core/tensor_core/wgmma.py`）：

`execute_wgmma_for_group(executor_warp, all_4_warps, A_desc, B_desc, C_regs, D_regs)`：

1. 解析 A_desc → smem region；用 `SharedMemory.load_*` 读 64×K 矩阵
2. 同样读 B（K×128）
3. 收集 C：从 4 warps × 32 lanes × N regs 重建 64×128 矩阵
4. 用 numpy + ml_dtypes 算 D = A·B + C
5. 分发 D 到 4 warps × 32 lanes × N regs

**4 warp × N reg 的 layout（fictional）**：

```
warp w (0..3) lane i (0..31) reg %dj (0..63):
    D[w * 16 + i / 2][(i % 2) * 64 + j]
```

**Timing + WgmmaQueue**：

- issue_occupancy = `tc_wgmma_occupancy`（默认 4 cycle）
- result latency = `tc_wgmma_latency`（默认 32 cycle）
- 但 wgmma async：issue 后**不直接标 scoreboard ready**
- 每条 wgmma 入 `WgmmaQueue` per warp-group：

```python
@dataclass
class InflightWgmma:
    issued_at: int
    completion_at: int
    dst_regs: tuple[tuple[str, ...], ...]  # 4 warps × N regs
    commit_group_id: int

class WgmmaQueue:
    in_flight: list[InflightWgmma]
    committed_groups: list[int]    # group id 列表
    next_group_id: int
    capacity: int = 16             # max in-flight；满则 WGMMA_QUEUE_FULL
```

### 4.3 commit_group 与 wait_group N

**commit_group**（warp 0 代理）：
- `next_group_id += 1`
- 当前所有未 committed 的 in_flight wgmma 标 commit_group_id = `next_group_id`
- push 到 `committed_groups`
- issue_occupancy = 1，无 scoreboard 影响

**wait_group N**：
- 阻塞 warp 直到 `len(committed_groups) ≤ N`
- 等待中 warp 状态 = `WGMMA_WAIT`（新 stall token）
- 每 cycle 检查：最早 committed group 的所有 wgmma 都过 completion_at → drain group → pop
- Drain 同时：把那些 wgmma 的 dst_regs 在 scoreboard 上 ready_at 设为当前 cycle

### 4.4 新功能单元 + 配置

```python
class FUKind(Enum):
    FP32 = "fp32"; INT = "int"; LSU = "lsu"; BRU = "bru"; SYNC = "sync"
    TC = "tc"      # NEW
```

`FUSet.classify(op)`：
- `mma.*` / `wgmma.*` → `TC`
- `cp.async.bulk.tensor.*` → `LSU`（复用，类似 ld.global，但 bypass cache）
- `mbarrier.*` → `SYNC`

`TensorCoreConfig`：

```python
@dataclass
class TensorCoreConfig:
    tc_mma_latency: int = 8
    tc_mma_occupancy: int = 1
    tc_wgmma_latency: int = 32
    tc_wgmma_occupancy: int = 4
    wgmma_queue_capacity: int = 16
```

`SMConfig.tensor_core: TensorCoreConfig = field(default_factory=TensorCoreConfig)`。

### 4.5 新 stall tokens

| Token | 触发 |
|---|---|
| `WGMMA_QUEUE_FULL` | issue wgmma 时队列满 |
| `WGMMA_WAIT` | wait_group N 等待中 |

总 stall 类数：12（Phase 2 含 MSHR_FULL）→ **13**（含 WGMMA_QUEUE_FULL）→ **14**（含 WGMMA_WAIT）。

---

## 5. TMA-lite + Mbarrier 设计

### 5.1 `gpusim.tma_desc` 伪指令

替代真机 `tensormap.*` 操作链。用法：

```
gpusim.tma_desc %rd_handle, %rd_gmem_base, dim_x, dim_y, stride_y, elem_bytes;
```

`InstrExecutor` 在执行此指令时：
1. 在 per-SM 的 `TensorDescriptorPool` 分配新 entry
2. 填字段
3. 写 entry index 到 `%rd_handle`

后续 `cp.async.bulk.tensor.2d` 通过 handle 拿 descriptor。Spec §11 明确标注：真机 PTX 没有此伪指令。

### 5.2 cp.async.bulk.tensor.2d 执行语义

**Functional**：
1. 解 descriptor 字段
2. 总字节 = `dim_x * dim_y * elem_bytes`
3. 用 `GlobalMemory` + `SharedMemory` 直接拷贝 gmem → smem
4. 立即完成（数值正确）

**Timing**（决定 mbarrier 何时 arrive）：
1. cache_lines = ceil(tile_bytes / 128)
2. **Bypass L1/L2**——TMA 真机走 dedicated copy engine，不经 cache。模拟器复用 HBM channel queue
3. 每 line 调 `hbm.request(line_addr, now)`；max(...) 作为 completion_at
4. completion_at 注册到 mbarrier 的 pending_tx

**默认参数**：
- `tma_issue_occupancy = 4 cycle`
- 完成时间 = max(per-line HBM serve cycle)

### 5.3 Mbarrier 状态机

```python
@dataclass
class Mbarrier:
    expected_count: int
    arrived_count: int = 0
    phase: int = 0
    pending_tx: list[tuple[int, int]] = field(default_factory=list)


class MbarrierPool:
    """Per-CTA pool. SM holds one per active CTA."""
    def init(self, smem_addr: int, expected: int) -> None: ...
    def arrive(self, smem_addr: int) -> None: ...
    def arrive_tx(self, smem_addr: int, tx_bytes: int, completion_at: int) -> None: ...
    def tick(self, now: int) -> None: ...     # SM 每 cycle 调，drain pending
    def try_wait(self, smem_addr: int, expected_phase: int) -> bool: ...
```

`tick(now)` drain pending_tx with completion_at <= now，每个 drain 等价于一次 `mbarrier.arrive`。

### 5.4 `mbarrier.try_wait` 不引入新 stall

Try_wait 配 `@!%p bra LOOP` 自旋等待。**spin-loop 的 cycle 浪费自然出现在 trace（warp 状态 = ISSUED 但只在 try_wait + bra）**。教学正确——真机也是 spin。

### 5.5 SM main loop 集成

```python
# 在 sub_cores.step 之后、barrier coordination 之前
for cta_id in active_ctas:
    self._mbarrier_pools[cta_id].tick(cycle)
```

---

## 6. Trace + 分析 + 可视化

### 6.1 完整事件清单（Phase 1+2+3）

| 类别 | 事件 | 频率 |
|---|---|---|
| Phase 1（8） | WARP_STATE, INSTR_ISSUE, SMEM_ACCESS, GMEM_ACCESS, DIV_PUSH/POP, BAR_REACH/RELEASE, CTA_LAUNCH/RETIRE | 高（RLE 压缩） |
| Phase 2（3） | L1_ACCESS, L2_ACCESS, HBM_ACCESS | 中-低 |
| **Phase 3（4，新增）** | **MmaEvent, WgmmaEvent, TmaEvent, MbarrierEvent** | 中-低 |

### 6.2 Phase 3 新事件 schema

```python
@dataclass(frozen=True)
class MmaEvent:
    cycle: int
    warp_id: int
    pc: int
    precision: str
    shape_m: int
    shape_n: int
    shape_k: int
    accum_dtype: str
    flops_count: int


@dataclass(frozen=True)
class WgmmaEvent:
    kind: str            # "ISSUE" | "COMMIT_GROUP" | "WAIT_GROUP" | "DRAIN"
    cycle: int
    warp_group_id: int
    pc: int
    precision: str = ""
    shape_m: int = 0; shape_n: int = 0; shape_k: int = 0
    accum_dtype: str = ""
    commit_group_id: int = -1
    wait_n: int = -1
    completion_at: int = -1


@dataclass(frozen=True)
class TmaEvent:
    cycle: int
    completion_at: int
    smem_dst: int
    gmem_base: int
    dim_x: int
    dim_y: int
    bytes_total: int
    n_cache_lines: int
    mbarrier_addr: int


@dataclass(frozen=True)
class MbarrierEvent:
    kind: str            # "INIT" | "ARRIVE" | "ARRIVE_TX" | "FLIP" | "TRY_WAIT"
    cycle: int
    cta_id: int
    smem_addr: int
    expected: int = 0
    arrived: int = 0
    phase: int = 0
    pred_result: bool = False
```

### 6.3 新增分析指标

| 函数 | 输出 | 教学用途 |
|---|---|---|
| `tc_utilization(mma_df, wgmma_df, total_cycles, n_sub_cores=4)` | DataFrame[n_sub_cores] | TC busy % per sub-core |
| `precision_distribution(mma_df, wgmma_df)` | DataFrame：precision → count + flops | FP8 vs FP16 比例 |
| `effective_tflops(mma_df, wgmma_df, total_cycles, freq_ghz=1.0)` | dict per precision | 与真机数据卡片桥梁 |
| `async_overlap_ratio(wgmma_df, warp_state_df)` | scalar 0..1 | wgmma 在飞时 warp 是否在做事 |
| `mbarrier_wait_distribution(wgmma_df, mbarrier_df)` | pd.Series：cycle 直方图 | wait_group 等待分布 |
| `wgmma_queue_pressure(wgmma_df, total_cycles)` | pd.Series：每 cycle in-flight 数 | 队列何时满 |
| `tma_bandwidth_utilization(tma_df, total_cycles, total_hbm_bw)` | scalar | TMA 是否撑满 HBM |

### 6.4 HTML 报告新增节

在 Phase 2 的 §6–§10 后追加 4 个新节：

| 节 | 内容 |
|---|---|
| **§11 Tensor Core utilization** | Per-sub-core bar chart + peak/theoretical TFLOPS |
| **§12 Precision distribution** | Stacked bar by precision + FLOPS contribution table |
| **§13 wgmma async pipeline timeline** | Plotly Gantt：每个 warp-group 的 wgmma issue/in-flight/wait_group/commit_group |
| **§14 Mbarrier flips & TMA arrivals** | 时序：每 mbarrier 一行，TMA arrive_tx + thread arrives + flip 标记 |

### 6.5 Result API 扩展

```python
@dataclass
class Result:
    # Phase 1+2 fields...

    @property
    def mma_events_df(self) -> pd.DataFrame: ...
    @property
    def wgmma_events_df(self) -> pd.DataFrame: ...
    @property
    def tma_events_df(self) -> pd.DataFrame: ...
    @property
    def mbarrier_events_df(self) -> pd.DataFrame: ...

    @property
    def tc_metrics(self) -> dict: ...
    def tc_summary(self) -> str: ...
```

`Result.summary()` 升级再加一行带 TC 信息（与 Phase 2 cache_summary 同等）。

### 6.6 Perfetto 集成

| 事件 | Perfetto track | 视觉 |
|---|---|---|
| `MmaEvent` | per-warp "TC" track | 红色 instant |
| `WgmmaEvent("ISSUE")` | per-warp-group "TC" track | 大红色 instant |
| `WgmmaEvent("WAIT_GROUP")` | per-warp-group "TC" track | 灰色 instant |
| `TmaEvent` | per-CTA "TMA" track | 蓝色 instant + duration |
| `MbarrierEvent("FLIP")` | per-CTA "Barrier" track | 绿色 instant |

新 track 类型："TC" per warp-group、"TMA" per CTA、"Barrier" per CTA。

### 6.7 Parquet 落盘

新增 4 个 parquet 文件：`mma.parquet`、`wgmma.parquet`、`tma.parquet`、`mbarrier.parquet`。

---

## 7. 测试策略

延续 Phase 1+2 三层金字塔。

### 7.1 单元测试（pytest）

| 模块 | 关键测试 |
|---|---|
| `core/tensor_core/precision` | ml_dtypes cast round-trip、fp16/bf16/fp8 storage size、cvt 精度 |
| `core/tensor_core/mma` | 3 shape × 6 dtype 的 functional 正确性（与 numpy reference 对比） |
| `core/tensor_core/wgmma` | wgmma_basic functional + WgmmaQueue allocate/commit/wait/drain |
| `core/tma` | TensorDescriptor pool、cp.async.bulk.tensor.2d functional |
| `core/mbarrier` | init/arrive/arrive_tx/tick/try_wait state machine |
| `frontend/parser` | mma 多 dtype opcode 解码、brace-list operands、`gpusim.tma_desc` 伪指令、COLONCOLON |
| `analysis/metrics` | 7 个新指标的 fixture |
| `viz/html_report` | 4 个新节都正确插入 |

### 7.2 Functional Parity（numpy）

4 个新 example 各有 numpy 参考实现：
- `tc_matmul_precisions`：每变体单独对比，容忍度按精度（FP32=0、FP16/BF16=1e-2、FP8=2e-1、TF32=1e-3、INT8=精确）
- `mixed_accum`：FP16 accum 容忍度 3e-1（精度退化是教学点），FP32 accum 2e-3
- `wgmma_basic`：64×128 matmul，FP32 ref，rtol=1e-2
- `wgmma_async_pipeline`：256×128 matmul，FP32 ref，rtol=1e-2

Phase 1/2 example 全部继续通过。

### 7.3 Reference Fixture（真机对照）

`tests/reference/data/<name>.ref.json` schema 已存在；4 个新 kernel 加 schema：
- `effective_tflops` ±20% 容忍
- `tc_utilization` ±15%
- `precision_distribution` 严格相等

`gen_reference.py` SUPPORTED_KERNELS 加 4 项。

### 7.4 微基准（教科书事实）

新增 `tests/microbench/test_phase3_facts.py`：

```
- FP8 m16n8k32 单 mma cycles ≤ 1.1× FP16 m16n8k16 单 mma cycles
  （同 latency 8 cycle，但 FP8 单条覆盖 2× K → 2× FLOPS/cycle）
- FP16 accum 与 FP32 accum 在 64 次累加后误差 ratio ≥ 100
  （FP16 accum 误差 ≈ 3e-1，FP32 accum 误差 ≈ 2e-3）
- wgmma m64n128k16 单条 cycles ≪ 64 × sync mma m16n8k16 cycles
  （等价覆盖 64×128 输出需 64 条 sync mma；wgmma 单条 ≤ 1/4 此路径，async pipeline + 大粒度收益）
- async_overlap_ratio in wgmma_async_pipeline ≥ 0.5（pipeline 重叠生效）
- mbarrier flip 数 ≈ K-tile 数 × 2（双 buffer，上限近似）
```

---

## 8. 项目结构改动

### 8.1 目录新增

```
gpusim/core/tensor_core/
├── __init__.py
├── mma.py
├── wgmma.py
└── precision.py
gpusim/core/tma.py
gpusim/core/mbarrier.py

tests/unit/cache/                (沿用，无改)
tests/unit/tensor_core/
├── __init__.py
├── test_precision.py
├── test_mma.py
└── test_wgmma.py
tests/unit/core/test_tma.py
tests/unit/core/test_mbarrier.py
tests/microbench/test_phase3_facts.py
```

### 8.2 配置文件迁移

`default_hopper.yaml` 加 `tensor_core` 节：

```yaml
tensor_core:
  tc_mma_latency: 8
  tc_mma_occupancy: 1
  tc_wgmma_latency: 32
  tc_wgmma_occupancy: 4
  wgmma_queue_capacity: 16
```

### 8.3 依赖

`pyproject.toml`：

```toml
dependencies = [
    # Phase 1+2 既有...
    "ml_dtypes>=0.4",   # NEW for FP16/BF16/FP8 native dtypes
]
```

---

## 9. 教学示例与讲义

### 9.1 4 个新 example kernels

| # | Example | 教学意图 | PTX 数量 |
|---|---|---|---|
| 1 | **tc_matmul_precisions** | sync mma + 6 精度对比 | 6 PTX 变体（FP32 baseline + FP16 + BF16 + E4M3 + TF32 + INT8） |
| 2 | **mixed_accum** | accumulator 精度的必要性 | 2 PTX 变体（FP16 in/out vs FP16 in + FP32 accum） |
| 3 | **wgmma_basic** | 单 wgmma 看 Hopper shape | 1 PTX：64×128 matmul 单条 wgmma |
| 4 | **wgmma_async_pipeline** | TMA + wgmma 真实生产模式 | 1 PTX：256×128 matmul + ping-pong + TMA 重叠 wgmma |

每个目录：`{kernel*.ptx} + reference.py + run.py + README.md`。

### 9.2 4 章新讲义

| # | 标题 | 关联 example |
|---|---|---|
| 12 | Tensor Core 入门：sync mma 与 register layout | tc_matmul_precisions（FP16 部分） |
| 13 | 精度面板：FP16/BF16/FP8/TF32/INT8 trade-off | tc_matmul_precisions |
| 14 | 混合精度：FP32 accumulator 为何不可省略 | mixed_accum |
| 15 | wgmma + TMA：Hopper 真实生产模式 | wgmma_basic + wgmma_async_pipeline |

每章结尾固定栏目：**看模拟器** / **改一改** / **真机对照**。

---

## 10. 与 Phase 1+2 兼容性

### 10.1 不会破坏的部分

| 维度 | 状态 |
|---|---|
| `gpusim.run(...)` 函数签名 | 不变 |
| Phase 1/2 example PTX | 不动 |
| Phase 1/2 parity 测试 | 全部继续通过 |
| `gpusim.cli` 命令集 | 不变 |
| Result 旧字段（含 Phase 2 cache_metrics 等） | 不变 |
| 配置 yaml 旧 sections | 不变 |
| Stall token 既有 12 类 | 不变 |
| HTML 报告既有 §1–§10 | 位置 + 内容不变 |
| Perfetto trace 既有 track | 不变 |

### 10.2 会变的部分

| 维度 | 变化 |
|---|---|
| **新依赖** | `ml_dtypes>=0.4` |
| **Cycle 数（Phase 1/2 examples）** | 几乎不变——既有 kernel 不发 mma/wgmma |
| **Stall 直方图** | 多 2 类 `WGMMA_QUEUE_FULL`/`WGMMA_WAIT` |
| **HTML 报告** | 多 4 节（§11–§14） |
| **Perfetto** | 多 3 类 track（TC / TMA / Barrier） |
| **Trace parquet** | 多 4 文件 |
| **Result** | 多 5 个属性 + `tc_summary()` |
| **`Instr.type`** | `PtxType` → `Optional[PtxType]` |
| **`Operand` union** | + `RegGroup` |
| **`PtxType`** | 加 8 项 |

---

## 11. 显式不在范围内（Phase 3）

记录以避免误解：

- **Atomics**（`atom.*`、`red.*`）
- **Memory fence / coherency**（除 `wgmma.fence`）
- **L1 / L2 prefetching**
- **Multi-SM L2 共享**（Phase 4）
- **Volta+ ITS**（Phase 1+2 既有继承）
- **多 GPU、NVLink、NCCL**
- **Tensor Core 全 PTX 形状集**（仅每精度一个 canonical shape）
- **真机 lane → element layout 精确映射**（fictional layout 简化）
- **真机 wgmma async semantics 完整 fence 模型**（仅简化版）
- **TMA descriptor 的真机字节级 encoding**（用 `gpusim.tma_desc` 伪指令简化）
- **TMA swizzle modes / multicast**
- **`tensormap.*` 系列 PTX 指令**
- **TMA store**（`cp.async.bulk.tensor.2d` 反方向 smem→gmem）—— Phase 3 仅 load，store 可 Phase 4+ 加
- **Cluster 级 distributed shared memory**
- **Async shared memory store**（`cp.async.shared`）
- **DRAM command-level 时序**（与 Phase 2 一致）
- **FP64 mma**（Hopper 真机有，教学价值低）

---

## 12. 已知近似与简化

- **Lane → element 是 fictional layout**：教学方便，与真机 PTX register-to-element mapping 不一致；学生不能据此读 cuBLAS 代码
- **`gpusim.tma_desc` 伪指令**：模拟器约定，非真机 PTX；等价于 host 端 `cuTensorMapEncode` 后传入 kernel
- **TMA bypass cache**：直达 HBM channel queue，与真机 dedicated copy engine 行为一致
- **wgmma 数值精确**（用 numpy + ml_dtypes），但 register layout 是简化版
- **TC FU 单条 issue**：每 sub-core 1 个 TC，不建模真机的多 lane 并行
- **mbarrier 简化**：仅 phase + arrived + pending_tx，不建模真机的"transaction count" 完整字段
- **`wgmma.fence` 简化**：在模拟器里是 1-cycle no-op（功能上 mma_async 已经隐含 fence；spec 标注真机的精确语义）
- **No FP64**：Phase 3 不实现 FP64 Tensor Core，教学价值低
- **No `cp.async`**（Ampere-style）：仅 Hopper TMA-lite

---

## 13. Phase 3 实施里程碑（高层）

| 里程碑 | 交付 |
|---|---|
| **M1** | Frontend 扩展：lexer (`COLONCOLON`) + IR (8 个新 PtxType + RegGroup + TensorDescriptor + MbarrierHandle) + parser (brace-list + mma/wgmma/cp.async.bulk.tensor.2d/mbarrier.* 的多 dotted-modifier + `gpusim.tma_desc` 伪指令) + 单元测试。无运行时行为变化 |
| **M2** | sync mma：`FUKind.TC` + `tensor_core/mma.py` + `tensor_core/precision.py` (ml_dtypes) + SubCore TC 路由 + 2 examples (tc_matmul_precisions 6 变体 + mixed_accum 2 变体) + parity tests + 微基准 |
| **M3** | wgmma 核心：`warp_group_id` field + warp-group sync coordination in SM + `WgmmaQueue` + `tensor_core/wgmma.py` + commit/wait_group + scoreboard async 整合 + 2 个新 stall tokens + wgmma_basic example |
| **M4** | TMA + mbarrier：`tma.py` (TensorDescriptor + cp.async.bulk.tensor.2d) + `mbarrier.py` (state machine) + SM mbarrier tick + wgmma_async_pipeline example + 集成测试 |
| **M5** | Trace + 分析 + viz + 收尾：4 trace 事件 + 7 分析指标 + 4 HTML 节 + Perfetto + Result API + 4 章新讲义 + reference fixture 扩展 + Phase 3 微基准 + README v3 + tag `phase3-complete` |

预估总任务数 30–35（与 Phase 2 的 27 + 3 fix 相近）。

每个 milestone 之间打 git tag (`M{1..5}-phase3-complete`) 作为 review checkpoint。

---

## 14. 设计协作记录

本文档由用户与 Claude（Opus 4.7, 1M context）通过 `superpowers:brainstorming` 流程逐节确认产出。所有关键决策均经用户显式确认（B/C/A 选择或"确认"回复）。

下一步：交由 `superpowers:writing-plans` 产出可执行的实施计划。
