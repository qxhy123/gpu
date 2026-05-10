# CUDA / Hopper 中文深度教程 — Design Spec

> **Status:** Brainstormed 2026-05-10. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- 一套独立的中文 CUDA / Hopper SM90 深度教程,目标读者是有 C/C++ 基础、想深入理解 NVIDIA GPU 硬件 + CUDA 软件栈的工程师 / 研究者。
- 完全独立于 gpusim,不引用 gpusim 任何模块、API 或代码。
- 22 章主题 + 00 全景索引 = 23 个 markdown 文件。
- 每章遵循统一八节结构(见 §3),~1500-2500 中文字。
- 内容深度对齐 NVIDIA 官方文档(CUDA C++ Programming Guide、PTX ISA、Hopper Architecture Whitepaper)。
- 代码示例使用真实 CUDA C++ / PTX 写法,可对照官方 sample 验证。

### Non-goals
- 不写英文版(独立项目,不平行翻译)。
- 不模拟 / 不引用 gpusim。
- 不写 Pascal / Volta / Turing / Ampere 历史细节(只在 Tensor Core 演进等必要处提一句)。
- 不覆盖 OpenCL、SYCL、HIP 等其他 GPU 编程模型。
- 不覆盖 Multi-Instance GPU (MIG) — 留给 Phase 2 教程。
- 不覆盖 CUDA Graph 12.4+ 的设备侧条件节点细节(只标注存在)。

---

## 2. 文档路径与组织

```
docs/cuda-zh/
├── 00-index.md
├── 01-simt-execution.md
├── 02-sm-internals.md
├── 03-smem-and-l1.md
├── 04-l2-cache-and-setaside.md
├── 05-hbm3-and-gmem.md
├── 06-atomics.md
├── 07-tensor-core.md
├── 08-wgmma-async-matmul.md
├── 09-tma.md
├── 10-mbarrier.md
├── 11-thread-block-cluster.md
├── 12-cta-scheduling-gigathread.md
├── 13-streams-and-events.md
├── 14-nvlink-nvswitch.md
├── 15-nccl-collectives.md
├── 16-cuda-graphs.md
├── 17-persistent-and-dynamic-parallelism.md
├── 18-stream-ordered-allocator.md
├── 19-unified-memory.md
├── 20-cuda-driver-api.md
├── 21-profiling-toolchain.md
└── 22-ptx-to-sass.md
```

总计 23 个文件,纯 Markdown,无外部依赖。

---

## 3. 每章统一结构

每章必须包含以下 8 节,顺序固定:

```markdown
# NN · <中文标题>

> **一句话总结。**

## 1. 是什么 / 为什么有它
读者背景重建。一段话(80-150 字)解释这个组件存在的根本原因。

## 2. 硬件视角(微架构细节)
- Hopper SM90 上具体怎么实现
- 关键数字(端口数、bank 数、cache size、cycle 延迟)
- **涉及架构布局或硬件块关系时,使用 Mermaid `flowchart` / `classDiagram`(优先于 ASCII art)**

## 3. CUDA 编程接口
- C++ API(`cuda::`、`__device__`、PTX intrinsic)
- PTX 指令名(`mma.sync`、`cp.async.bulk`、`fence.sc`)
- 头文件路径

## 4. 关键性能指标
- 数值带宽 / 延迟 / 占用阈值
- 性能模型公式(如 occupancy = min(...))

## 5. 代码示例
- 1-2 段可读 CUDA C++ 或 PTX,展示典型用法
- 注释解释每行做什么
- **涉及多步流程 / 时序 / 状态变化时,用 Mermaid `sequenceDiagram` / `stateDiagram` 配合代码**

## 6. 实测手段
- NSight Systems / NSight Compute 用哪个 metric
- `nvprof` / CUPTI 事件名(若适用)
- `nvidia-smi` 命令(若适用)

## 7. 常见反模式
- 容易踩的 3-5 个坑
- 每个坑给一句"为什么错"

## 8. 延伸阅读
- CUDA C++ Programming Guide 章节号 + 标题
- PTX ISA 文档章节(若适用)
- Hopper Whitepaper 页码(若适用)
- 官方 sample 路径(若适用)
- 必要时一篇 NVIDIA 博客 URL(只引用 docs.nvidia.com 或 developer.nvidia.com)
```

**字数指引:** 每节 100-400 字;整章 1500-2500 字。代码块不计入字数。

---

### Mermaid 图表强制要求

**所有涉及"架构布局"、"数据流"、"控制流"、"状态机"、"通信时序"的章节必须用 Mermaid,不允许用 ASCII art 替代。**

| 内容类型 | 推荐 Mermaid 图类型 |
|---|---|
| 硬件块关系(SM 内部、缓存层级、SM↔L2↔HBM) | `flowchart TD` 或 `flowchart LR` |
| 类 / API 层级(Driver vs Runtime) | `classDiagram` |
| 时序(kernel launch 流程、NCCL collective 步骤、TMA 完成通知) | `sequenceDiagram` |
| 状态机(warp state、mbarrier phase、CTA lifecycle) | `stateDiagram-v2` |
| 拓扑(NVLink topology、Cluster CGA、ring/tree) | `graph LR` 节点带 label |
| 数据通路(算子流水线) | `flowchart LR` 横向 |

每章至少 **1 个** Mermaid 图(00 索引章 ≥ 2 个);上限不限。

Mermaid 渲染:GitHub / VS Code / 主流 markdown 浏览器原生支持,无需额外工具。

---

## 4. 各章主题与覆盖范围

### 00 · 全景索引 + Hopper SM90 架构图
- 整套教程的导航;给出"按硬件层级"和"按软件抽象层"两个阅读路径。
- Hopper SM90 全图 ASCII art:HBM3 → L2 → SM × 132 → 4 sub-partition × { warp scheduler / 32 FP32 ALU / 16 FP64 ALU / TC × 4 / TMA / mbarrier / regfile / SMEM }
- CUDA 软件栈分层图:CUDA C++ Runtime API ↔ Driver API ↔ ptxas ↔ SASS ↔ ucode

### 01 · SIMT 执行模型
- warp = 32 lane,谓词执行,Independent Thread Scheduling (Volta+),SIMT stack
- divergent branch 行为,convergence point,`__syncwarp`

### 02 · SM 内部结构
- 4 sub-partition,每个 sub 一个 warp scheduler,各自的 64 KiB regfile
- functional units:32 FP32 ALU、16 FP64、INT32、SFU、TC、TMA、Load/Store
- regfile 切片(64K registers/SM,256/thread cap)
- scoreboard 与依赖追踪

### 03 · 共享内存 + L1
- Hopper:228 KiB unified L1+SMEM,可配为 (0, 8, 16, 32, 64, 100, 132, 164, 196, 228) KiB SMEM 划分
- 32 banks × 4 B/word × 1 cycle,bank conflict 公式
- 双缓冲(SMEM ping-pong)

### 04 · L2 缓存 + set-aside
- 60 MiB L2(Hopper),16 路组相联,128 B line
- LRU 替换 + persistence-attribute(`cudaCtxResetPersistingL2Cache`)
- L2 set-aside(stream priority 影响)

### 05 · HBM3 + 全局内存
- 5 TB/s 带宽,5 stack × 1024-bit bus
- channel / bank / row,row-buffer 命中模型
- coalescing 规则:128 B 段,1 transaction
- gmem 访问模式与 sector 利用率

### 06 · 原子操作
- global atomic 走 L2 ALU(无需 cache line 拉到 SM)
- shared atomic 在 SMEM 控制器
- `red.async`(只写不返回值)的优化路径
- atomic 性能反模式(争用)

### 07 · Tensor Core
- 演进:Volta V100 → Ampere A100 → Hopper H100
- 数据类型:FP16/BF16/TF32/FP8 (E4M3, E5M2)/INT8/INT4
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`(Ampere 风格)
- Hopper 仍支持 mma.sync,但推荐 wgmma(下一章)

### 08 · wgmma 异步矩阵乘
- warp-group(128 thread)级 mma
- `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16`
- `wgmma.commit_group.sync.aligned`、`wgmma.wait_group.sync.aligned N`
- 与 TMA + mbarrier 配合的 pipeline 模式

### 09 · TMA(Tensor Memory Accelerator)
- `cp.async.bulk.tensor.[1-5]d.global.shared::cluster`
- `CUtensorMap` 描述符,5D box,swizzle 模式
- `cp.async.bulk.commit_group` / `wait_group`
- 用 mbarrier 通知完成

### 10 · mbarrier 异步屏障
- 64-bit SMEM 对象,内含 phase / arrived count / expected count
- `mbarrier.init.shared`、`mbarrier.arrive.shared`、`mbarrier.try_wait.shared`
- `mbarrier.expect_tx`(用于 TMA 完成)
- phase 翻转模型

### 11 · Thread Block Cluster (CGA)
- SM90 新增,1-16 CTA 组成 cluster,跨 CTA SMEM 互访(DSMEM)
- `__cluster_dims__`、`cooperative_groups::cluster_group`
- cluster barrier(`barrier.cluster.arrive` / `barrier.cluster.wait`)
- cluster TMA store

### 12 · CTA 调度 + GigaThread
- GigaThread 工作分发引擎
- 一个 SM 可同时跑多少 CTA(occupancy)= min(8 CTA, regfile 限制, smem 限制, warp 限制)
- `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSm)` 调优
- 优先级流的 dispatch 影响

### 13 · CUDA Streams + Events
- 默认流 vs 显式流;`cudaStreamCreate` / `cudaStreamCreateWithPriority`
- `cudaEventRecord` / `cudaStreamWaitEvent` / `cudaEventElapsedTime`
- `cudaStreamWaitAll` 模式
- L2 set-aside per stream

### 14 · NVLink + NVSwitch
- NVLink 4(Hopper):900 GB/s 总带宽 / GPU,18 link
- NVSwitch 3:fully-connected 8-way 或 256-GPU systems(NVL36/NVL72)
- P2P 显存访问(`cudaDeviceEnablePeerAccess`)
- SHARP(in-network reduction)

### 15 · NCCL 集合通信
- AllReduce(ring / tree / SHARP variants)
- ReduceScatter、AllGather、Broadcast、SendRecv
- 异步执行模型(`ncclGroupStart` / `ncclGroupEnd`)
- Bandwidth 模型 `2(N-1)/N · M / B`

### 16 · CUDA Graphs
- 显式构造(`cudaGraphCreate` / `cudaGraphAddKernelNode`)
- Stream Capture(`cudaStreamBeginCapture` / `cudaStreamEndCapture`)
- `cudaGraphInstantiate` / `cudaGraphLaunch`
- Conditional graph node(CUDA 12.4+,设备侧 / 主机侧两种)
- Graph update(`cudaGraphExecUpdate`)

### 17 · Persistent + Dynamic Parallelism
- Persistent kernel 模式(grid-stride loop + 工作队列)
- Dynamic Parallelism 2.0(`cudaLaunchKernelEx`,有限制版本)
- 与 cudaGraph child node 的关系

### 18 · Stream-ordered Allocator
- `cudaMallocAsync` / `cudaFreeAsync`
- `cudaMemPoolCreate` / `cudaMemPoolTrimTo` / 属性 `cudaMemPoolAttrReleaseThreshold`
- PyTorch caching allocator 如何在其上构建

### 19 · Unified Memory
- `cudaMallocManaged` 内存模型
- 按需页面迁移(50 µs 量级开销)
- `cudaMemPrefetchAsync` / `cudaMemAdvise`(SetReadMostly / PreferredLocation / AccessedBy)
- Page-fault 路径
- HMM(Heterogeneous Memory Management)与 ATS(GH200/MI300)

### 20 · CUDA Driver API
- libcuda.so 接口(vs libcudart.so 高层)
- Context 模型(primary context vs explicit context),`cuCtxPushCurrent`
- Module / Function 加载(`cuModuleLoad` / `cuModuleGetFunction`)
- `cuLaunchKernel` 与 `cudaLaunchKernel` 关系
- 何时需要直接调 driver API(动态加载 cubin、JIT、嵌入式)

### 21 · Profiling 工具栈
- NSight Systems(系统级时间线)
- NSight Compute(kernel-level metrics:`smsp__inst_executed`、`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_sector_*`)
- CUPTI(callback / activity API)
- NVTX(`nvtxRangePush` / `nvtxMarkA`,被 nsys 自动捕获)
- 推荐工作流(profile → 找瓶颈 → 验证)

### 22 · PTX → SASS 编译链
- nvcc 整体 pipeline:`nvcc → cudafe → cicc → ptxas → fatbin`
- `ptxas -arch=sm_90a` 与 `sm_90` 区别(后者无 wgmma/TMA)
- SASS 反汇编:`cuobjdump --dump-sass`
- ptxas 关键 flag:`-O3` / `--maxrregcount` / `--ptxas-options=-v`
- JIT 编译路径(driver 收到 PTX 时)

---

## 5. 内容质量准则

- **每章独立可读** — 读者从任意章入门都能理解。前置概念用一句话回顾,不假设读者已读前面所有章。
- **真实代码** — `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {…}, {…}, {…}, {…};` 而不是伪代码。
- **数字必须有来源** — Hopper 的具体数字(60 MiB L2、228 KiB SMEM、5 TB/s 带宽)必须可对照 NVIDIA 官方文档,引用时注明来源(Hopper Whitepaper p.X / Programming Guide §X.X)。
- **不做对比** — 不写 "AMD MI300 怎么做"、"Intel Xe 怎么做"。
- **避免营销语言** — 不用"革命性"、"业界领先"、"最强大"。
- **代码块加 ```cpp 或 ```ptx 或 ```bash 等正确 fence。**

---

## 6. 文件输出规范

- **编码:** UTF-8 无 BOM
- **换行:** LF
- **行长:** 中文不强制;代码块不超过 100 字符
- **标题层级:** 一级 `#` 仅章名(`# NN · 标题`);二级 `##` 用于八节
- **图片:** Mermaid 优先(架构 / 流程 / 状态 / 时序 / 拓扑都用 Mermaid);ASCII art 只用于无法 Mermaid 表达的场景(如内存 bank 物理布局矩阵);不引用外部 PNG/JPG。
- **超链接:** 仅链 docs.nvidia.com / developer.nvidia.com / github.com/NVIDIA

---

## 7. 实施策略

### 7.1 章节分组(便于并发实现)

22 章按主题相关性分 5 组,每组可由一个 subagent 写完:

| 组 | 章节 | 主题 |
|---|---|---|
| **G1** | 00, 01, 02 | 全景 + SIMT 基础 + SM 结构 |
| **G2** | 03, 04, 05, 06 | 内存层级 + 原子 |
| **G3** | 07, 08, 09, 10, 11 | 计算单元 + Hopper 异步特性 + Cluster |
| **G4** | 12, 13, 14, 15, 16, 17 | 调度 + Stream + 多 GPU + Graph + 持久化 |
| **G5** | 18, 19, 20, 21, 22 | 内存管理 + Driver API + 工具链 + 编译 |

5 组可并发或顺序实施。

### 7.2 测试策略
- 无单元测试(纯文档)。
- **结构验证:** 提供一个简单的 markdown 校验脚本(可选)— 检查每个文件是否有完整的 8 节标题 + 字数在 1500-2500 范围。
- **链接检查:** 不强制(避免 docs.nvidia.com 短期 404 噪音)。

### 7.3 里程碑

| Milestone | Scope | Tag |
|---|---|---|
| **M1** G1 完成 | 00-02 | `cuda-zh-G1-complete` |
| **M2** G2 完成 | 03-06 | `cuda-zh-G2-complete` |
| **M3** G3 完成 | 07-11 | `cuda-zh-G3-complete` |
| **M4** G4 完成 | 12-17 | `cuda-zh-G4-complete` |
| **M5** G5 完成 + 索引交叉链接 + 总结提交 | 18-22 + cross-link | `cuda-zh-complete` |

总计约 ~23 个写作任务(每章 1 个)+ 1 个交叉链接整理任务。

---

## 8. 文件清单

### 新增文件
```
docs/cuda-zh/00-index.md
docs/cuda-zh/01-simt-execution.md
docs/cuda-zh/02-sm-internals.md
docs/cuda-zh/03-smem-and-l1.md
docs/cuda-zh/04-l2-cache-and-setaside.md
docs/cuda-zh/05-hbm3-and-gmem.md
docs/cuda-zh/06-atomics.md
docs/cuda-zh/07-tensor-core.md
docs/cuda-zh/08-wgmma-async-matmul.md
docs/cuda-zh/09-tma.md
docs/cuda-zh/10-mbarrier.md
docs/cuda-zh/11-thread-block-cluster.md
docs/cuda-zh/12-cta-scheduling-gigathread.md
docs/cuda-zh/13-streams-and-events.md
docs/cuda-zh/14-nvlink-nvswitch.md
docs/cuda-zh/15-nccl-collectives.md
docs/cuda-zh/16-cuda-graphs.md
docs/cuda-zh/17-persistent-and-dynamic-parallelism.md
docs/cuda-zh/18-stream-ordered-allocator.md
docs/cuda-zh/19-unified-memory.md
docs/cuda-zh/20-cuda-driver-api.md
docs/cuda-zh/21-profiling-toolchain.md
docs/cuda-zh/22-ptx-to-sass.md
```

23 个 markdown 文件。

### 修改文件
无(此教程独立,不改 README,不改现有 `docs/tutorial/`)。

---

## 9. 验收准则

教程完成的标准:

- [ ] 23 个 markdown 文件全部存在
- [ ] 每个文件包含完整的 8 节(章名 + § 1-8)
- [ ] 每章字数在 1500-2500 范围
- [ ] 全部章节零 gpusim 引用(grep `gpusim` 应无命中)
- [ ] 5 个里程碑 tag 全到位
- [ ] 00-index.md 提供两条阅读路径并链接到所有 22 章
- [ ] 每章至少 1 个 Mermaid 图,00 索引至少 2 个(grep ` ```mermaid` 累计应 ≥ 24 命中)
