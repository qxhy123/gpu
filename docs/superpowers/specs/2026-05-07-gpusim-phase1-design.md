# gpusim Phase 1 — 核心 SM 模拟器设计文档

**日期**：2026-05-07
**状态**：设计阶段（待实现）
**作者**：与 Claude 协同 brainstorm
**范围**：仅 Phase 1。Phase 2–5 仅作为愿景列出。

---

## 1. 愿景与分阶段

### 1.1 项目愿景

`gpusim` 是一个**教学优先**的、用于学习 NVIDIA GPU 微架构与 AI Infra 的全功能模拟器。学习者通过编写小型 kernel、跑模拟器、看可视化报告与配套讲义，理解 SIMT 执行、warp 调度、内存层次、Tensor Core、多 GPU 通信等关键议题。

### 1.2 分阶段路线图

整个项目分 5 个 Phase，每个 Phase 独立成 spec 与实现周期：

| Phase | 范围 | 状态 |
|---|---|---|
| **Phase 1** | 单 SM、cycle-approximate、PTX 子集、shared/global memory（无 cache）、可视化、教学示例与讲义 | **本文档** |
| Phase 2 | L1 / L2 cache、HBM 带宽建模 | 后续 |
| Phase 3 | Tensor Core、FP16/BF16/FP8、wgmma | 后续 |
| Phase 4 | 多 SM、CTA→SM 调度、L2 共享 | 后续 |
| Phase 5 | 多 GPU、NVLink、NCCL collective | 后续 |

每个新 Phase 在前一阶段基础上扩展，不破坏已有 API。

### 1.3 Phase 1 目标一句话

> 在 Python 里实现一个单 SM、Hopper-shaped、cycle-approximate 的 GPU 模拟器，能跑 PTX 子集、能精确建模 SIMT 分歧、shared memory bank conflict 与 global memory coalescing，输出 HTML 报告 + Perfetto 时间轴 + Notebook 友好 API，并附 6 个教学示例和 8 篇讲义。

### 1.4 已锁定决策

| 维度 | 决策 |
|---|---|
| 实现语言 | Python（runtime 全 Python，不引入 C++ 扩展） |
| 输入 | PTX 子集（~30 条核心指令） |
| 模拟精度 | Cycle-approximate；可切换 functional 快速模式 |
| 参考架构 | Hopper（H100, sm_90），普通 warp 路径；TMA/cluster/wgmma 留给 Phase 3 |
| 主循环 | Cycle-stepped 模拟 |
| 调度策略 | LRR + GTO，可配置 |
| SIMT 模型 | PDOM stack（不实现 Volta+ 的 ITS） |
| 可视化出口 | HTML 静态报告 + Perfetto trace + Notebook DataFrame |
| 测试 | 单元测试 + numpy 参考 + 真机参考结果文件接口 |
| 交付 | 库 + CLI + 6 个示例 + 8 篇讲义 |

---

## 2. 架构与模块划分

### 2.1 总图

```
                  ┌─────────────────────────────────────┐
   kernel.ptx ──▶ │  1. frontend (PTX parser + IR)      │
                  └──────────────────┬──────────────────┘
                                     │  IR (typed instr stream + kernel meta)
                                     ▼
                  ┌─────────────────────────────────────┐
   config.yaml ─▶ │  2. config (SM params, mode toggle) │
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
   launch args ─▶ │  3. core (SM model: warps, sched,   │
                  │     SIMT stack, regfile, pipeline,  │
                  │     functional units, smem, gmem)   │
                  └──────────────────┬──────────────────┘
                                     │  per-cycle events
                                     ▼
                  ┌─────────────────────────────────────┐
                  │  4. trace (event recorder, RLE,     │
                  │     parquet writer)                 │
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │  5. analysis (stall classification, │
                  │     occupancy, bank conflict count, │
                  │     coalescing stats)               │
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │  6. viz (HTML report, Perfetto JSON,│
                  │     pandas DataFrames for notebook) │
                  └─────────────────────────────────────┘

                  ┌─────────────────────────────────────┐
   CLI ──────────▶│  gpusim.cli (run / show / explain)  │──▶ uses 1-6
                  └─────────────────────────────────────┘
```

### 2.2 模块职责

1. **frontend**：词法+语法解析 PTX 子集 → 内部 IR（每条指令是 typed dataclass）。纯转换，无副作用。也负责 IPDOM（reconvergence point）的静态计算。
2. **config**：加载 `SMConfig`（warp 数、sub-core 数、scheduler 策略、regfile 大小、smem 大小/bank 数、各功能单元延迟与吞吐）。dataclass + YAML loader。
3. **core**：模拟器心脏。`SM` 对象持有 warps、sub-cores、schedulers、SIMT stack、register file、shared memory、functional units（INT/FP/MEM/BRU/SYNC）。提供 `step()` 推进一个 cycle，`run()` 跑到完成。所有微架构行为都在这里。
4. **trace**：core 通过事件回调写入；事件流是结构化记录（cycle, warp_id, event_type, payload）。设计成可被多个消费者订阅，避免 core 直接耦合到 viz。
5. **analysis**：消费 trace，产出聚合指标。无状态、可独立测试。
6. **viz**：把 analysis 结果渲染成 HTML 报告 / Perfetto trace JSON / pandas DataFrame。无业务逻辑。

### 2.3 边界原则

- core 不知道 viz/CLI 的存在，只发事件
- frontend 输出的 IR 是 core 的唯一输入语言
- trace 是 core 与下游分析层之间的"防火墙"——下游全靠 trace 驱动，便于回放与离线分析
- 每个模块对外暴露窄接口，内部可独立替换

---

## 3. PTX 子集与 IR

### 3.1 起步指令集（约 30 条）

| 类别 | 指令 | 说明 |
|---|---|---|
| 数据移动 | `mov`, `ld.global`, `st.global`, `ld.shared`, `st.shared`, `ld.param` | 三种内存空间 |
| 整数算术 | `add.s32/u32`, `sub.s32`, `mul.lo.s32`, `mad.lo.s32`, `shl.b32`, `shr.s32` | |
| 浮点算术 | `add.f32`, `sub.f32`, `mul.f32`, `mad.f32`, `fma.f32` | FP32 通路 |
| 比较与分支 | `setp.eq/ne/lt/le/gt/ge`, `@p bra`, `bra` | 谓词驱动 → SIMT stack |
| 同步 | `bar.sync`, `membar.cta` | CTA 内同步 |
| 特殊寄存器 | `mov %tid.{x,y,z}`, `%ntid.x`, `%ctaid.x`, `%nctaid.x` | 通过 mov 读取 |
| 转换 | `cvt.s32.f32`, `cvt.f32.s32` | 最常用两条 |

### 3.2 显式不在 Phase 1 中

FP64、FP16/BF16、Tensor Core（`mma`/`wgmma`）、`cp.async`、TMA、原子操作、纹理、warp shuffle（`shfl`）。后续 Phase 加入。

> **关于 warp shuffle 的说明**：教学价值高但 Phase 1 不引入。理由：reduction 等示例可用 shared memory 版本完成；shuffle 实现需小心处理 lane 间数据通路，会膨胀 Phase 1 复杂度。

### 3.3 IR 数据结构

```python
@dataclass(frozen=True)
class Instr:
    op: str                    # "add.f32", "ld.global", ...
    dst: tuple[Operand, ...]
    src: tuple[Operand, ...]
    pred: Predicate | None     # @p / @!p / None
    space: MemSpace | None     # global / shared / param / None
    type: PtxType              # s32, u32, f32, b32, ...
    pc: int                    # 指令地址
    src_loc: SrcLoc            # 文件:行，便于错误信息与归因

@dataclass(frozen=True)
class Kernel:
    name: str
    params: list[Param]        # 形参列表
    regs: RegDecl              # 寄存器声明（数量、类型）
    instrs: list[Instr]
    labels: dict[str, int]     # label → instr index
    ipdom: dict[int, int]      # 每条 bra 指令的 IPDOM PC
```

性质：
- 不可变；core 在执行时不修改 IR，状态全部在 SM 对象里
- 不丢源信息：`src_loc` 让 trace 与报告能反指源码

### 3.4 解析器实现策略

手写递归下降解析器，~300 行 Python，零外部依赖。不使用 lark/ANTLR。

---

## 4. 核心模拟策略与 SM 蓝本

### 4.1 主循环：Cycle-stepped

主循环 `for cycle in count(): sm.step()`，每个组件每 cycle 都被调用一次推进自身状态。

理由：
- Phase 1 跑教学级 kernel（~10⁴–10⁶ 指令），Python cycle-stepped 性能完全够用
- 每 cycle 都可暂停、可观察，教学价值核心
- functional 快速模式作为 cycle-stepped 的特例（关闭 timing 组件），无需独立架构

### 4.2 Hopper-shaped SM 默认参数

```
SM 总览
  sub_cores per SM          : 4
  max warps per SM          : 64    （每 sub-core 16）
  max threads per SM        : 2048
  max CTAs per SM           : 32
  registers per SM          : 65536 × 32-bit
  shared memory per SM      : 228 KB（用户上限；Phase 1 默认 48KB）
  shared memory banks       : 32（4-byte stride）

Sub-core
  warp scheduler            : 1（每 cycle 最多发射 1 条 warp 指令）
  registers                 : 16384 × 32-bit，4 banks
  FP32 throughput           : 1 warp-instr / cycle
  INT32 throughput          : 1 warp-instr / cycle
  LSU throughput            : 1 warp memory-instr / cycle
  SFU throughput            : 1 warp / 4 cycle（Phase 1 暂不用）
  branch unit               : SIMT stack 操作，1 cycle

延迟与占用（cycles，教学近似值）
  FP32/INT32 ALU
    operand-ready latency       : 4
    LSU/ALU issue occupancy     : 1
  FMA                           : 同上
  shared memory load/store
    LSU issue occupancy         : N（N = bank conflict degree，详见 6.1；无冲突 N=1）
    operand-ready latency       : 20 + (N − 1)
  global memory load
    LSU issue occupancy         : 1（mem-instr 进入 outstanding queue 后即释放 issue 槽）
    operand-ready latency       : 400（Phase 1 无 cache，固定）
  global memory store           : 异步（不阻塞，进 store buffer）
  bar.sync                      : 等所有 warp 到达
```

### 4.3 关于参数来源的声明

H100 没有完全公开的 cycle-level 时序表。上述数值综合自 NVIDIA whitepaper、micro-benchmark 论文（Jia et al. 系列）以及 Volta/Ampere 已知值的合理外推。所有数值均通过 `config.yaml` 暴露给用户，HTML 报告中标注当前使用的参数集。

### 4.4 Pipeline 阶段（每 sub-core）

```
  Fetch → Decode → Issue/Sched → Operand Collector → Execute → Writeback
```

每个 warp 在 sub-core 中持有：1 个 instruction buffer（fetch 后存放）、1 个 scoreboard（追踪未完成的写）。

---

## 5. SIMT 栈、调度器、功能单元

### 5.1 SIMT 栈（PDOM）

每个 warp 维护一个栈，栈顶是当前活跃 mask。

```python
@dataclass
class SIMTEntry:
    pc: int           # 该路径下一条指令
    active_mask: int  # 32-bit，1 表示该 lane 活跃
    rpc: int          # reconvergence PC（IPDOM）
```

算法（Fung 2007 PDOM stack）：
1. frontend 静态分析每条 `@p bra L`，预先标好 IPDOM 作为 RPC
2. 执行分歧分支：
   - 谓词对所有活跃 lane 同向 → 不分歧，仅更新栈顶 PC
   - 否则 push 两帧：(taken_pc, taken_mask, rpc) 与 (fallthrough_pc, ~taken_mask & cur_mask, rpc)
3. 当栈顶 PC == 栈顶 RPC → pop，回到上层路径
4. 每 cycle 只执行栈顶帧（核心约束 → 分歧成本可见）

**显式简化**：不实现 Volta+ 引入的 ITS（Independent Thread Scheduling）。讲义中说明真实 Hopper 用 ITS。

### 5.2 Warp 调度器

两种内置策略，每个 sub-core 一个独立调度器，拥有该 sub-core 的 16 个 warp slot：

- **LRR（Loose Round Robin）**：每 cycle 轮转，跳过非 ready warp
- **GTO（Greedy-Then-Oldest）**：当前 warp 能发射就持续发射；stall 后切到 SM 上"最早启动且 ready"的 warp

通过 `config.scheduler.policy = "lrr" | "gto"` 切换。

#### Ready 判定（每 cycle 检查）

1. instruction buffer 非空
2. scoreboard 无 RAW 冲突
3. 所需功能单元本 cycle 有空（结构 hazard）
4. 谓词不全为假（否则空发射，记 `PRED_OFF`）
5. 不在 `bar.sync` 等待中

每次 ready 失败记 stall reason 进 trace。

### 5.3 功能单元（每 sub-core）

| 单元 | 处理的指令 | 吞吐 | 延迟 | 流水化 |
|---|---|---|---|---|
| FP32 ALU | `add/sub/mul/mad/fma.f32` | 1/cycle | 4 | 是 |
| INT32 ALU | `add/sub/mul/mad/shl/shr.s32`, `setp` | 1/cycle | 4 | 是 |
| BRU | `bra`, `@p bra` | 1/cycle | 1 | 是 |
| LSU | `ld.*`, `st.*`, `mov` | 1/cycle | 见内存模型 | 是（含 outstanding queue） |
| SYNC | `bar.sync`, `membar` | 1/cycle | 见同步 | 否（阻塞） |

流水化：连续两条同类指令可背靠背 issue，结果在 4 cycle 后依次出来。

结构 hazard：同 sub-core 同 cycle 只能 issue 1 条指令；不建模 dual-issue。跨 sub-core 独立 issue，所以一个 SM 每 cycle 最多 4 条 warp-instr 发射。

---

## 6. 内存模型、寄存器文件、多 CTA、Stall 分类

### 6.1 Shared Memory（精确 bank conflict 模型）

- 32 banks，4-byte stride：`bank(addr) = (addr >> 2) & 31`
- 每条 warp shared 指令按线程算地址 → 按 bank 分组，得到**冲突度** N：
  - 全部不同 bank → N = 1
  - 同 bank 同地址 → broadcast，N = 1
  - K 个线程访问同 bank 的 K 个不同地址 → N = K（K-way conflict）
- N 决定该指令的 **LSU issue occupancy**（占用 LSU 的 cycle 数）和额外延迟（见 4.2）
- 报告输出每条 shared 指令的冲突度直方图（tiled matmul / reduction 教学核心证据）

### 6.2 Global Memory（Phase 1：无 cache，仅 coalescing 分析）

- 32 个线程的地址按 128-byte sector 聚合（H100 cache line = 128B）
- `n_transactions = 不同 sector 数`
- 每笔 transaction 固定 `global_latency` cycle（默认 400，可配）
- LSU 维护 outstanding 队列（默认 16 entry/sub-core），队满则新 mem-instr issue 时 STRUCTURAL stall
- store 进 store buffer，不阻塞后续 issue（除非显式 `membar`）
- 关键指标：`coalescing_efficiency = active_threads / (n_transactions × 32)`
- Phase 2 加 cache 后，`global_latency` 变成 cache 模型的输出；Phase 1 先把 transaction 数算对

### 6.3 寄存器文件（per sub-core）

- 4 banks，`bank(reg) = reg_id & 3`
- Operand collector（教学版）：每条指令读 ≤3 个 src，issue 那 cycle 计算需读的 bank 集合
  - 全部 bank 不同 → 0 额外延迟
  - 有 K 个 bank 冲突 → +K-1 cycle，记 `OPERAND` stall
- 不实现真实 collector 的 reservation station 队列；保留扩展点

### 6.4 多 CTA 与 Occupancy（教学重点）

启动时计算理论 occupancy：

```
warps_per_cta     = ceil(threads_per_cta / 32)
max_ctas_by_warps = 64  // warps_per_cta
max_ctas_by_regs  = 65536 // (regs_per_thread * threads_per_cta)
max_ctas_by_smem  = smem_per_sm // smem_per_cta
active_ctas       = min(上述三者, max_ctas_per_sm=32)
```

报告清晰展示**哪一项是瓶颈**（warps / regs / smem）。

调度策略：
- 启动时一次性把 `active_ctas` 个 CTA 装入 SM 的 warp slot
- 某 CTA 全部 warp 退出后立即装入下一个待调度 CTA
- 同 SM 内不同 CTA 之间不共享 shared memory（每 CTA 独立切片）
- `bar.sync` 仅同 CTA 内同步

### 6.5 Stall 分类（10 类 token）

每 cycle 每 warp 记一个 token：

| Token | 含义 |
|---|---|
| `ISSUED` | 本 cycle 成功发射 |
| `IDLE` | warp 已结束 / 该 slot 空 |
| `FETCH_EMPTY` | 指令未到位（接口预留） |
| `SCOREBOARD` | 等待未完成写（RAW） |
| `STRUCTURAL` | 功能单元 / issue 槽 / LSU 队列繁忙 |
| `OPERAND` | regfile bank 冲突 |
| `MEM_DEP` | 等待 outstanding load |
| `BARRIER` | `bar.sync` 等其它 warp |
| `PRED_OFF` | 全谓词为假（空发射） |
| `DIVERGENCE_SERIAL` | 当前 cycle 在执行非全活跃 mask 的栈帧（分歧成本） |

`analysis` 模块把这些聚合成：每 warp 时序条形图、按原因 stall 直方图、SM 总 IPC、warp-level occupancy 时序。

---

## 7. Trace、分析与可视化

### 7.1 事件类型

| Event | 何时发 | Payload |
|---|---|---|
| `CTA_LAUNCH` / `CTA_RETIRE` | CTA 调度上 SM / 退出 | `cta_id`, `warps`, `regs`, `smem_bytes` |
| `WARP_STATE` | 每 cycle 每 warp 一条 | `warp_id`, `state_token`, `pc` |
| `INSTR_ISSUE` | 指令发射 | `warp_id`, `pc`, `op`, `src_loc`, `active_mask` |
| `INSTR_COMPLETE` | 写回 | `warp_id`, `pc` |
| `SMEM_ACCESS` | shared 访存完成 | `bank_conflict_degree`, `addresses[32]` |
| `GMEM_ACCESS` | global 访存发起 | `n_transactions`, `coalesce_eff`, `addresses[32]` |
| `DIV_PUSH` / `DIV_POP` | SIMT 栈操作 | `pc`, `rpc`, `taken_mask` |
| `BAR_REACH` / `BAR_RELEASE` | bar.sync | `cta_id`, `barrier_id` |

### 7.2 容量控制

`WARP_STATE` 高频（cycles × warps）。在内存中用 per-warp run-length encoding：连续相同 state 合并为 `(start_cycle, end_cycle, state)` 段；落盘时直接写 RLE 段。一个跑 100 万 cycle、64 warp 的 kernel，事件量 ~MB 级。

### 7.3 落盘格式

- 默认 `parquet`（结构化、列式、pandas/polars 直读）
- 可同时导出 `perfetto.json`（仅 ISSUE / WARP_STATE / BAR / DIV 转换为 Perfetto 切片）

### 7.4 Analysis 指标

无状态函数，输入 parquet，输出指标 dict + DataFrames：

| 指标 | 说明 |
|---|---|
| `ipc_timeline` | SM 每 cycle 的指令发射数时序（4 sub-core 求和；时间轴上的 IPC 曲线） |
| `stall_breakdown` | 按 token 聚合的 cycle 占比 |
| `stall_by_source_line` | 教学杀手锏：每行 PTX 各 stall 原因占总执行时间多少 |
| `bank_conflict_hist` | 每条 shared 指令的冲突度直方图 |
| `coalescing_per_instr` | 每条 global 指令的 transaction 数与 efficiency |
| `occupancy_timeline` | 同时活跃 warp/CTA 的时序曲线 |
| `divergence_cost` | 因 SIMT 分歧序列化付出的 cycle 总数 |
| `bottleneck_classification` | 启动期 occupancy 三项瓶颈中谁触顶（warps/regs/smem） |

### 7.5 可视化（三条出口）

#### HTML 报告（默认）

单文件、自包含，Jinja2 + Plotly 内嵌：
- 顶部：kernel meta（grid/block、occupancy 计算、瓶颈项）
- 中部：四张图——IPC 时序 / Stall 饼图 / 每行 PTX 热度条 / Bank conflict 直方图
- 底部：源码视图（PTX 全文，每行右侧标注被发射次数与 stall 占比），可点击跳到对应时间窗

#### Perfetto trace 导出

`gpusim run --perfetto out.json`，每个 warp 一条 track，instruction 作为 slice，div/bar 作为 instant 事件，跨 warp 的 BAR 事件用 flow line 连接。用户拖拽进 https://ui.perfetto.dev。

#### Notebook API

```python
result = gpusim.run("kernel.ptx", grid=(8,), block=(128,), config="hopper.yaml")
result.summary()
result.stall_df
result.events
result.timeline(warp=0)
```

API 是 `gpusim.run` 的薄封装；CLI 内部也调它。

---

## 8. 测试与验证策略

三层金字塔：

### 8.1 Layer 1 — 单元测试（pytest）

| 模块 | 关键测试 |
|---|---|
| `frontend` | 每条 PTX 指令 round-trip；语法错误定位；IPDOM 计算正确性 |
| `core/simt_stack` | 无分歧不 push；分歧后 mask 正确分裂；reconverge 时 pop；嵌套分支 |
| `core/scheduler` | LRR 轮转跳过非 ready；GTO 黏性 + 切换语义；两策略产出固定 trace |
| `core/regfile` | bank 冲突计数；scoreboard RAW/WAW 检测 |
| `core/smem` | 32 种典型 stride 模式的 bank conflict 度（含 broadcast、permute） |
| `core/gmem` | coalescing efficiency 在 stride=1/2/4/random 下的期望值 |
| `core/occupancy` | 三种瓶颈各自触顶的样例 |
| `analysis` | 给定固定 trace，stall 分类与归因输出固定值（金标 fixture） |

### 8.2 Layer 2 — Functional Parity（numpy 参考）

每个 example kernel 同时有一份 numpy 参考实现，模拟器 outputs 与之逐元素对比（rtol=1e-5）。无需 GPU。

### 8.3 Layer 3 — 真机参考接口（用户提供数据）

#### 目录结构

```
tests/reference/
├── README.md                       # 如何在真机上生成
├── gen_reference.py                # 用户在真机上运行的脚本
└── data/
    ├── vector_add.ref.json         # 一个 kernel 一个文件
    ├── reduction.ref.json
    └── ...
```

#### `*.ref.json` schema

```json
{
  "kernel": "vector_add",
  "ptx_path": "kernels/vector_add.ptx",
  "launch": {"grid": [8], "block": [128]},
  "device": {"name": "H100 SXM5", "sm_count": 132},
  "inputs_shape": {"a": [1024], "b": [1024], "n": []},
  "inputs_seed": 42,
  "outputs": {"c": "<base64 npy>"},
  "metrics": {
    "active_warps_per_sm": 64,
    "achieved_occupancy": 1.0,
    "smem_bank_conflicts": 0,
    "gld_efficiency": 1.0
  }
}
```

#### 两层对比

pytest 标记 `@pytest.mark.reference`，仅当 ref 文件存在时跑：

1. **数值层**：`outputs` numpy 数组逐元素对比（rtol=1e-5）
2. **指标层**：模拟器 `metrics` 与真机在合理误差内（occupancy ±5%、bank conflicts 严格相等、gld_efficiency ±10%）
3. **timing 层**：故意不做 cycle 数对齐（cycle-approximate 不要求精确）

#### 真机数据采集脚本

`gen_reference.py`：
- 用 `nvcc` 编译 kernel（PTX inline 或 `.cu` 包装）
- 跑 kernel 拿 outputs
- 用 `nsys` / `ncu` / CUPTI 抓 metrics
- 写出 `.ref.json`

用户在真机跑一次 → 产物 commit 进仓库 → CI 与本地无 GPU 也能跑。

### 8.4 微基准期望（教科书事实断言）

固化为 pytest，fail 即模拟器有原则性问题：
- 32-stride shared 访问 → 32-way bank conflict
- stride-2 global 访问 → coalescing efficiency = 50%
- 完全分歧的 if/else → 两路径串行（执行 cycle ≈ 两路径之和）
- 单 warp kernel → IPC ≤ 1

---

## 9. 项目结构、CLI/API、依赖

### 9.1 目录结构

```
gpusim/
├── pyproject.toml
├── README.md
├── docs/
│   ├── tutorial/                # 配套讲义
│   └── superpowers/specs/       # 设计文档
├── gpusim/                      # 主包
│   ├── api.py                   # gpusim.run() 主入口
│   ├── cli.py                   # CLI
│   ├── frontend/                # PTX 解析、IR、IPDOM
│   ├── config/                  # SMConfig + default_hopper.yaml
│   ├── core/                    # sm / sub_core / warp / simt_stack /
│   │                            # scheduler / regfile / scoreboard /
│   │                            # functional_units / smem / gmem /
│   │                            # occupancy / exec
│   ├── trace/                   # events / recorder / writer
│   ├── analysis/                # stall / attribution / metrics
│   └── viz/                     # html_report / perfetto / notebook
├── examples/                    # 6 个教学 kernel
│   └── <name>/{kernel.ptx, kernel.cu, reference.py, run.py, README.md}
├── tests/
│   ├── unit/
│   ├── parity/
│   ├── reference/
│   └── microbench/
└── scripts/
    └── ptx_from_cuda.py         # .cu → .ptx 封装
```

### 9.2 CLI（typer）

```
gpusim run KERNEL.ptx
   --grid X[,Y[,Z]]  --block X[,Y[,Z]]
   --inputs name:path.npy[,name:path.npy...]
   --config CONFIG.yaml          # 默认 default_hopper.yaml
   --output report.html
   --perfetto trace.json
   --trace events.parquet
   --mode {timing,functional}    # 默认 timing
   --seed N

gpusim show KERNEL.ptx           # 打印 IR + IPDOM 标注
gpusim explain report.html       # 终端摘要
gpusim doctor                    # 检查依赖、config 合法性
```

### 9.3 Python API

```python
import gpusim

result = gpusim.run(
    "kernel.ptx",
    grid=(8,), block=(128,),
    params={"a": a_arr, "b": b_arr, "n": 1024},
    config="hopper.yaml",         # 或 SMConfig 实例
    mode="timing",
    seed=42,
)

result.outputs           # dict[str, np.ndarray]
result.summary()         # 一行总结
result.metrics           # 见 7.4
result.events_df         # pandas, 原始事件
result.stall_df          # pandas, stall 归因
result.timeline(warp=0)  # plotly figure
result.html_report("out.html")
result.perfetto("trace.json")
```

`result` 是 frozen dataclass + lazy properties。

### 9.4 依赖

**Runtime**：`numpy`, `pyyaml`, `pyarrow`, `pandas`, `plotly`, `jinja2`, `typer`

**Dev**：`pytest`, `pytest-cov`, `ruff`, `mypy`

**显式拒绝**：lark/ANTLR、Flask/FastAPI、Cython/pybind11。

### 9.5 性能预期

教学 kernel 规模（grid≤256, block≤256, 指令数 ≤ 几百条）：
- 模拟 1k–10k cycles，墙钟约 1–10 秒
- Python 解释器开销主导；可接受
- 优化路径（仅在测得需要时启用）：cProfile 找前 3 名热点，可能用 `numba.njit` 或 numpy 向量化局部优化。Phase 1 不引入 C++ 扩展。

---

## 10. 教学示例与讲义

### 10.1 6 个 example kernel

| # | Kernel | 教学意图 | 关键现象 |
|---|---|---|---|
| 1 | **vector_add** | 起步、最小可运行 | 完美 coalesce、occupancy 100%、IPC 上限 |
| 2 | **reduction_smem** | shared memory + `bar.sync` | 同步开销；不同 stride 的 bank 行为 |
| 3 | **tiled_matmul**（FP32, 16×16） | 数据复用、tile 加载到 smem | smem 命中、coalesced gmem load、`bar.sync` 占比 |
| 4 | **divergence_demo** | SIMT 分歧成本 | `DIV_PUSH/POP` 密集、`DIVERGENCE_SERIAL` 占比上升 |
| 5 | **bank_conflict_demo**（stride=1/32/broadcast） | 32-bank shared 行为 | 1-way / 32-way / broadcast 三档冲突直方图 |
| 6 | **coalescing_demo**（stride=1/2/4/random） | global transaction 数 | efficiency 100% / 50% / 25% / 低值 |

每个目录：
```
examples/<name>/
├── kernel.ptx
├── kernel.cu          # 教学交叉对照
├── reference.py       # numpy 参考
├── run.py             # 一键 run + 报告
└── README.md
```

每个 README.md 结构：
1. 本示例要展示什么
2. 关键代码点（带行号）
3. 运行命令
4. 预期观察（具体到指标值）
5. 延伸思考题

### 10.2 配套讲义（`docs/tutorial/`）

8 篇 markdown，每篇 ~1500–2500 字，配模拟器输出截图：

| # | 标题 | 关联示例 |
|---|---|---|
| 00 | 为什么需要 GPU 模拟器：能学到什么、不能学到什么 | — |
| 01 | SIMT 执行模型：warp、lane、active mask | vector_add |
| 02 | warp scheduler 与延迟掩藏：LRR vs GTO | vector_add 改 block size |
| 03 | global memory coalescing：transaction 是怎么算的 | coalescing_demo |
| 04 | shared memory 与 32 banks：冲突的本质 | bank_conflict_demo |
| 05 | 分支分歧：SIMT stack 怎么序列化 if-else | divergence_demo |
| 06 | occupancy：三个限制因素的相互作用 | reduction_smem 改 regs/smem |
| 07 | 端到端走一遍 tiled matmul：把所有概念串起来 | tiled_matmul |

每篇结尾固定栏目：
- **看模拟器**：精确 CLI 命令 + 在 HTML 报告里看哪一项
- **改一改**：让读者改一个参数后再跑
- **真机对照**（如该 kernel 有 reference fixture）：模拟器 vs 真机数据

讲义专注 Phase 1 内的概念。Tensor Core / cache / 多 GPU 等待对应 Phase 完成后再补章节。

---

## 11. 显式不在范围内（Phase 1）

记录在此以避免未来误解：

- FP64、FP16、BF16、FP8 数据通路
- Tensor Core（`mma`、`wgmma`）
- L1 / L2 cache 建模（Phase 1 全局内存为固定延迟无 cache）
- HBM 带宽与 bank 建模
- TMA（Tensor Memory Accelerator）
- Thread Block Cluster、分布式 shared memory
- `cp.async`、`cp.async.bulk`
- 原子操作（`atom.*`）
- 纹理与表面内存
- Warp shuffle（`shfl.*`）
- Volta+ ITS（Independent Thread Scheduling）
- 多 SM 并发（Phase 1 仅单 SM）
- 多 GPU、NVLink、NCCL
- 交互式 Web UI

---

## 12. 已知近似与开放说明

- **时序参数非官方**：所有 cycle 延迟与吞吐为综合公开材料的合理近似值，不等同于 H100 真机。报告中标注当前参数集。
- **PDOM 而非 ITS**：分歧模型是经典 PDOM stack，简化了 Volta+ 的 ITS 行为。
- **无 cache 的 global memory**：Phase 1 仅算 transaction 数与 coalescing 效率，固定延迟。Phase 2 替换为 cache 模型。
- **operand collector 简化**：仅按"读取 bank 集合"加 stall 周期，未实现真实 collector 的 reservation station 队列。

这些近似都是教学权衡——精度够展示对应现象，复杂度可控。

---

## 13. Phase 1 实施里程碑（高层）

详细任务由 writing-plans 阶段展开。

| 里程碑 | 交付 |
|---|---|
| M1 | frontend + functional executor → vector_add 数值正确 |
| M2 | cycle-stepped pipeline + scheduler + scoreboard → 第一份 trace |
| M3 | smem banks + gmem coalescing + regfile banks → bank/coalescing 示例 |
| M4 | 多 CTA + occupancy 计算 |
| M5 | trace recorder + analysis + HTML 报告 + perfetto 导出 |
| M6 | 6 个 examples + 讲义初稿 + reference fixture 接口 |

---

## 14. 设计协作记录

本文档由用户与 Claude（Opus 4.7, 1M context）通过 `superpowers:brainstorming` 流程逐节确认产出。所有关键决策均经用户显式确认（A/B/C 选择或 OK 回复）。

下一步：交由 `superpowers:writing-plans` 产出可执行的实施计划。
