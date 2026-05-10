# CUDA / Hopper 中文深度教程 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 写完 23 个独立中文 markdown 章节,覆盖 NVIDIA Hopper SM90 + CUDA 12 全栈。每章 1500-2500 字、统一 8 节结构、至少 1 个 Mermaid、零 gpusim 引用。

**Architecture:** 23 章按 5 组分组(G1=00-02, G2=03-06, G3=07-11, G4=12-17, G5=18-22)。每组一个 subagent 一次 dispatch 写完。最后一个 milestone 做交叉链接 + 索引补全。

**Tech Stack:** Markdown,Mermaid。无代码,无测试 — 验证靠 grep 结构 + 字数。

---

## 全局规则(适用于所有任务)

### 文件目录
所有章节写入 `docs/cuda-zh/`(若不存在,在第一个任务里 `mkdir -p docs/cuda-zh`)。

### 每章标准结构(强制)

每个 `NN-<slug>.md` 必须严格按下面顺序输出:

```markdown
# NN · <中文标题>

> **一句话总结。**

## 1. 是什么 / 为什么有它

## 2. 硬件视角(微架构细节)

## 3. CUDA 编程接口

## 4. 关键性能指标

## 5. 代码示例

## 6. 实测手段

## 7. 常见反模式

## 8. 延伸阅读
```

### Mermaid 要求(强制)
- 每章 **至少 1 个** Mermaid 代码块(00 索引章 ≥ 2 个)
- 内容 ↔ 类型对照:
  - 硬件块关系 → `flowchart TD/LR` 或 `classDiagram`
  - 时序(launch、collective 步骤、TMA 完成) → `sequenceDiagram`
  - 状态机(warp state、mbarrier phase、CTA 生命周期) → `stateDiagram-v2`
  - 拓扑(NVLink、Cluster CGA、ring/tree) → `flowchart LR/TB` 节点带 label
  - 数据通路(算子流水线) → `flowchart LR` 横向

### 字数指引
- 每节 100-400 中文字;整章 1500-2500 字(代码块 + Mermaid 不计入)
- 末尾(§8 后)不加任何额外内容(无 footer、无 license、无版权)

### 内容质量约束
- **零 gpusim 引用** — 写完后必须 `grep -i "gpusim"` 空命中
- **真实数字必须有出处** — Hopper 关键数字标注 "Hopper Whitepaper p.X" 或 "CUDA C++ Programming Guide §X.X"
- **真实代码** — `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 …` 而非伪代码
- **不写营销语** — 不出现"革命性"、"业界领先"、"最强大"
- **不做友商对比** — 不提 AMD MI、Intel Xe

### 输出格式
- UTF-8 无 BOM,LF 换行
- 一级标题 `#` 仅用于章名(`# NN · 标题`)
- 二级标题 `##` 仅用于八节
- 代码块 fence 使用正确语言:`cpp`、`ptx`、`bash`、`mermaid`

### 写完每章后的本地验证(每个任务步骤都要做)

```bash
# 1. 字数(代码块外的中文字符)
.venv/bin/python -c "
import sys, re, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding='utf-8')
# 移除代码块
text = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', text, flags=re.DOTALL)
# 中文字符
zh = re.findall(r'[一-鿿]', text)
print(f'{p.name}: {len(zh)} 中文字')
" docs/cuda-zh/NN-xxx.md

# 2. 8 节标题齐全
grep -c "^## [1-8]\. " docs/cuda-zh/NN-xxx.md     # 应等于 8

# 3. Mermaid 数量
grep -c '^\`\`\`mermaid' docs/cuda-zh/NN-xxx.md   # 应 ≥ 1(00 章 ≥ 2)

# 4. 无 gpusim
! grep -i 'gpusim' docs/cuda-zh/NN-xxx.md         # 应无命中
```

---

# 里程碑 G1 — 全景索引 + 基础

## Task 1: `00-index.md` — 全景索引 + Hopper SM90 架构图

**Files:**
- Create: `docs/cuda-zh/00-index.md`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p docs/cuda-zh
```

- [ ] **Step 2: 写章节内容**

写 `docs/cuda-zh/00-index.md`,严格按 §1-§8 八节结构。要点:

**§1 是什么 / 为什么:** 解释这套教程的目标读者(有 C/C++ 基础、想深入理解 NVIDIA GPU + CUDA 12 的工程师),给出"按硬件层级"和"按软件抽象层"两条阅读路径。

**§2 硬件视角:** 用 Mermaid `flowchart TB` 画 Hopper SM90 全景图,从顶到底展示 GPU → SM × 132 → 4 sub-partition → { warp scheduler / 32 FP32 ALU / TC × 4 / TMA / mbarrier / regfile / SMEM };侧边连接 L2 → HBM3。提及关键数字:132 SM(80 GB SXM5)、60 MiB L2、80 GB HBM3、5 TB/s 带宽。

**§3 CUDA 编程接口:** 用 Mermaid `classDiagram` 或 `flowchart LR` 画 CUDA 软件栈分层:用户 C++ → CUDA C++ Runtime API → Driver API → ptxas/JIT → SASS → ucode。

**§4 关键性能指标:** 列出 H100 SXM5 的关键峰值数字表格(FP32、FP16/BF16 TC、FP8 TC、HBM3 带宽、NVLink4 带宽)。

**§5 代码示例:** 给一个最小的"hello world"风格 CUDA C++ kernel 启动示例,展示 host + device 两侧。

**§6 实测手段:** 介绍 NSight Systems 安装 + `nsys profile -t cuda,nvtx ./app` 命令、`nvidia-smi` 几个常用查询。

**§7 常见反模式:** "跳过 profiling 直接调优"、"假设默认 occupancy 就是最优"、"忽略 warp divergence" 等。

**§8 延伸阅读:** 22 章索引列表,每个条目 `[NN · 标题](NN-xxx.md) — 一句话简介`。再加官方文档:CUDA C++ Programming Guide、PTX ISA、Hopper Whitepaper、NSight 文档。

- [ ] **Step 3: 本地验证**

按"写完每章后的本地验证"四步走:字数 1500-2500 中文字、`## [1-8]\.` = 8、`\`\`\`mermaid` ≥ 2、无 gpusim 命中。

- [ ] **Step 4: Commit**

```bash
git add docs/cuda-zh/00-index.md
git commit -m "docs(cuda-zh): 00 全景索引 + Hopper SM90 架构图"
```

---

## Task 2: `01-simt-execution.md` — SIMT 执行模型

**Files:**
- Create: `docs/cuda-zh/01-simt-execution.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** 解释 warp 是 NVIDIA GPU 调度的最小单位(32 lane);为何不是 thread;SIMT 与传统 SIMD 的区别(独立分支)。
- **§2:** Volta+ 引入的 Independent Thread Scheduling — 每个 lane 有自己的 PC + call stack;divergent branch 的硬件行为;`SSY/SYNC` SASS 指令。**Mermaid `stateDiagram-v2`** 画 warp 状态:Active → Stalled (mem) → Stalled (sync) → Eligible → Issued。
- **§3:** `__syncwarp(mask)`、`__activemask()`、`__ballot_sync(mask, predicate)`、warp shuffle (`__shfl_sync`、`__shfl_down_sync`)、collectives in `cooperative_groups::coalesced_threads`。
- **§4:** warp issue 频率(1 warp/cycle/scheduler × 4 scheduler/SM = 4 warp/cycle/SM);分支对吞吐的影响(divergence cost = lanes_executed_total / lanes_active);warp occupancy。
- **§5:** 一个 PTX 片段展示 `setp.gt.s32 %p0, %r0, 0; @%p0 ...; @!%p0 ...;` 谓词分支;一个 CUDA C++ 片段展示 `__shfl_xor_sync` warp reduce。
- **§6:** NSight Compute metric `smsp__warps_active.avg.pct_of_peak_sustained_active`、`smsp__thread_inst_executed_per_inst_executed`(SIMT efficiency)。
- **§7:** 三大反模式:warp 内 if-else 数据相关分支、误用 `__syncthreads` 替代 `__syncwarp`、忘记 mask 全 0xFFFFFFFF 的隐式假设。
- **§8:** CUDA C++ Programming Guide §5.4.4 (SIMT Architecture)、PTX ISA §8 (SIMT Stack)、Volta 白皮书 ITS 章节。

- [ ] **Step 2: 验证**(同 Task 1 Step 3)

- [ ] **Step 3: Commit**

```bash
git add docs/cuda-zh/01-simt-execution.md
git commit -m "docs(cuda-zh): 01 SIMT 执行模型"
```

---

## Task 3: `02-sm-internals.md` — SM 内部结构

**Files:**
- Create: `docs/cuda-zh/02-sm-internals.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** SM 是 GPU 的"核";Hopper 一片有 132 个 SM(80 GB SXM5)/ 114 个(PCIe);单 SM 内部能并发 64 个 warp。
- **§2:** **Mermaid `flowchart TB`** 画 SM 内部:SM → 4 sub-partition,每个 sub 含 { 1 warp scheduler / 16K registers / 32 FP32 ALU / 16 INT32 / 16 FP64 / 4 SFU / 1 TC / 1 LD/ST / shared mbarrier hardware };共享 228 KiB unified L1+SMEM、TMA engine、cluster barrier 硬件。
- **§3:** `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)`、`cudaOccupancyMaxActiveBlocksPerMultiprocessor`、`__nv_bfloat16` 等 type 选择影响数据通路。
- **§4:** 寄存器堆:65536 regs/SM,每 thread 上限 255 (默认) / 256 (`-maxrregcount=256`);occupancy 公式 = min(8 CTA, regs_per_SM / regs_per_CTA, smem_per_SM / smem_per_CTA, max_warps / warps_per_CTA);scheduler 4 个 sub 各自独立调度,不能跨 sub 切 warp。
- **§5:** CUDA C++ 片段展示 `__launch_bounds__` 与 occupancy API;PTX 片段展示 `.maxntid` 指示。
- **§6:** NSight Compute metrics `sm__warps_active.avg.pct_of_peak_sustained_active`、`launch__registers_per_thread`、`smsp__inst_issued.sum`。
- **§7:** "无脑加 thread/CTA"(register pressure 反而降低 occupancy);忽略 sub-partition 调度独立性导致 warp 不平衡;误以为 4 个 scheduler 共享 issue。
- **§8:** Hopper Whitepaper §SM Architecture、CUDA C++ Programming Guide §K.7 (Compute Capability 9.x)、Best Practices Guide §10.

- [ ] **Step 2: 验证**

- [ ] **Step 3: Commit**

```bash
git add docs/cuda-zh/02-sm-internals.md
git commit -m "docs(cuda-zh): 02 SM 内部结构"
```

---

## Task 4: 验证 G1 + tag

- [ ] **Step 1: 验证 G1 三章质量**

```bash
for f in docs/cuda-zh/00-index.md docs/cuda-zh/01-simt-execution.md docs/cuda-zh/02-sm-internals.md; do
    echo "=== $f ==="
    grep -c "^## [1-8]\. " "$f"
    grep -c '^```mermaid' "$f"
    ! grep -i 'gpusim' "$f" && echo "no gpusim ref OK"
done
```
Expected: 每个文件 8 节齐全,mermaid ≥ 1(00 章 ≥ 2),无 gpusim 命中。

- [ ] **Step 2: Tag**

```bash
git tag cuda-zh-G1-complete
```

---

# 里程碑 G2 — 内存层级 + 原子

## Task 5: `03-smem-and-l1.md` — 共享内存 + L1

**Files:**
- Create: `docs/cuda-zh/03-smem-and-l1.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** SMEM 是 SM 内部低延迟可编程暂存(20 cycle 量级);L1 是 SMEM 的剩余物理空间用作硬件 cache;Hopper 上两者共享 228 KiB,可配比例。
- **§2:** **Mermaid `flowchart LR`** 画 SM → unified L1+SMEM 物理 SRAM(228 KiB)→ 用户配置 carveout(0/8/16/32/64/100/132/164/196/228 KiB SMEM,余下当 L1)。32 banks × 4 B/word × 1 cycle 访问;bank conflict 公式 `accessed_banks_per_warp / unique_banks`。
- **§3:** `extern __shared__ float s[];`、`__shared__ float s[64];`、`cudaFuncSetAttribute(...cudaFuncAttributeMaxDynamicSharedMemorySize...)`、`cudaFuncSetAttribute(...cudaFuncAttributePreferredSharedMemoryCarveout...)`。
- **§4:** SMEM 单 bank 单 cycle 1 word 读写;无冲突理论吞吐 32 word/cycle = 128 B/cycle/SM;L1 read 100-150 cycle 命中(L2 miss 时 200+ cycle 走 HBM)。
- **§5:** 用 SMEM 做 tile 矩阵转置,用 padding `+1` 消除 bank conflict 的经典模式。
- **§6:** NSight Compute `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`、`smsp__sass_average_data_bytes_per_wavefront_mem_shared`。
- **§7:** stride-32 访问触发全 bank conflict;忘记 padding 使矩阵转置慢 32 倍;在 SMEM 上做大跨度散落写。
- **§8:** Programming Guide §B.2.3 (Shared)、§K.7.4 (Hopper SMEM carveout)、Best Practices §9.2.3。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/03-smem-and-l1.md
git commit -m "docs(cuda-zh): 03 共享内存 + L1"
```

---

## Task 6: `04-l2-cache-and-setaside.md` — L2 缓存 + set-aside

**Files:**
- Create: `docs/cuda-zh/04-l2-cache-and-setaside.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** L2 是 GPU 全局共享的下层 cache;Hopper 60 MiB(SXM5);所有 GMEM 访问都经过 L2;面对 HBM 高延迟(400+ cycle)L2 命中是保性能关键。
- **§2:** **Mermaid `flowchart TB`** 画 SM × 132 → L2(60 MiB,16-way,128 B line)→ HBM3 stack × 5。L2 替换策略 LRU + persistence attribute;ECC 开启占 ~6.25%。
- **§3:** `cudaCtxResetPersistingL2Cache`、L2 access window: `cudaStreamSetAttribute(...cudaStreamAttributeAccessPolicyWindow, &attr...)` + `attr.hitRatio` + `attr.hitProp = cudaAccessPropertyPersisting`。
- **§4:** L2 命中延迟 ~100-150 cycle;L2 总带宽 ~5 TB/s 可饱和 HBM;persistence cap = 30 MiB(可调,默认 ¼)。
- **§5:** CUDA C++ 配置一个 stream 的 L2 set-aside window 让 hot lookup table 常驻 L2 的代码片段。
- **§6:** NSight Compute `lts__t_sectors_pipe_lsu_mem_global_op_ld.sum`、`lts__t_sector_hit_rate.pct`。
- **§7:** "把全部 GMEM 都设 persisting"(逐 stream cap 限制反而打架)、忘记 reset 导致下一个 kernel 命中失效、误以为 L2 set-aside 是物理切分(实际是替换策略偏好)。
- **§8:** Programming Guide §3.2.3.6 (L2 Access Management)、§K.7 Hopper L2 size、Best Practices §9.2.3.4。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/04-l2-cache-and-setaside.md
git commit -m "docs(cuda-zh): 04 L2 缓存 + set-aside"
```

---

## Task 7: `05-hbm3-and-gmem.md` — HBM3 + 全局内存

**Files:**
- Create: `docs/cuda-zh/05-hbm3-and-gmem.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** HBM 是 GPU 显存物理 backing(DRAM 堆叠);Hopper 用 HBM3,SXM5 上 5 stack × 16 GB = 80 GB,带宽 5 TB/s。
- **§2:** **Mermaid `flowchart LR`** 画 channel → bank group → bank → row → row buffer 层级;每 stack 1024-bit bus、16 channel × 64-bit;row open/close cost。说明 coalescing:warp 内 32 lane 若访问同一对齐 128 B 段,1 个 sector 即可服务。
- **§3:** `__ldg`(只读 cached load)、`__stcg`/`__stcs`/`__stwt`(write hint)、`cudaMemcpy` 的 cudaMemcpyHostToDevice / DeviceToHost(底层走 PCIe / NVLink)。
- **§4:** HBM 行命中 ~50 ns、行未命中 ~150 ns;sector size 32 B(4 sector / 128 B line);warp 一次访问最优是 32 lane → 128 B。
- **§5:** 给一个 strided vs coalesced 数组求和的 PTX/CUDA 对比示例。
- **§6:** NSight Compute `dram__bytes_read.sum`、`dram__sectors_read.sum`、`l1tex__t_sector_pipe_lsu_mem_global_op_ld.sum`(对比 sector 利用率)。
- **§7:** stride-N 访问(N>1)导致 sector 利用率 1/N;misalignment 让 1 个事务变 2 个;非合并的 struct of arrays。
- **§8:** Programming Guide §3.2.2.1 (Coalesced Access)、§K.7 Hopper HBM3、Best Practices §9.2.1。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/05-hbm3-and-gmem.md
git commit -m "docs(cuda-zh): 05 HBM3 + 全局内存"
```

---

## Task 8: `06-atomics.md` — 原子操作

**Files:**
- Create: `docs/cuda-zh/06-atomics.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** 原子是无锁同步的基本工具;NVIDIA GPU 的 atomic 在 L2(global)和 SMEM 控制器(shared)硬件实现;读-改-写一次完成,不抢 cache line 到 SM。
- **§2:** **Mermaid `flowchart LR`** 画 GMEM atomic 路径:SM → L2 ALU(在线 reduce)→ HBM;对比 SMEM atomic 在 SMEM 控制器内完成。`red.async`(只写不返回)的快路径。
- **§3:** `atomicAdd / atomicMin / atomicMax / atomicCAS / atomicExch`、`__half2 atomicAdd`(Hopper 原生)、PTX `atom.global.add.f32`、`red.global.add.f32`(只写)、`red.async.shared::cta`。
- **§4:** L2 atomic 单 line 串行化(争用线性下降);unique address atomic 可并发;`red.async` 比 `atom` 快(无 ack);Hopper 原生支持 BF16/FP8 atomic-add。
- **§5:** 给一个直方图统计:全局 `atomicAdd` 慢路径 vs 用 SMEM `atomicAdd` 在 CTA 内聚合再合并到全局的对比代码。
- **§6:** NSight Compute `lts__t_sectors_atom_red.sum`、`l1tex__t_sectors_pipe_lsu_mem_shared_op_atom.sum`。
- **§7:** 全 warp atomic 同址(争用爆炸)、用 atomic 替代 reduce(应该 shfl_sync 或 cooperative_groups)、忘记用 SMEM 缓冲就上 GMEM atomic。
- **§8:** Programming Guide §B.14 (Atomic Functions)、§K.7 Hopper atomics、PTX ISA §8.7.12。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/06-atomics.md
git commit -m "docs(cuda-zh): 06 原子操作"
```

---

## Task 9: 验证 G2 + tag

- [ ] **Step 1: 批量验证**

```bash
for f in docs/cuda-zh/0[3-6]-*.md; do
    echo "=== $f ==="
    .venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding='utf-8')
text = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', text, flags=re.DOTALL)
zh = re.findall(r'[一-鿿]', text)
print(f'  {len(zh)} 中文字')
" "$f"
    echo "  $(grep -c '^## [1-8]\. ' "$f") 节"
    echo "  $(grep -c '^```mermaid' "$f") mermaid"
    ! grep -i 'gpusim' "$f" >/dev/null && echo "  no gpusim ref"
done
```

- [ ] **Step 2: Tag**

```bash
git tag cuda-zh-G2-complete
```

---

# 里程碑 G3 — 计算单元 + Hopper 异步特性 + Cluster

## Task 10: `07-tensor-core.md` — Tensor Core

**Files:**
- Create: `docs/cuda-zh/07-tensor-core.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** TC 是 GPU 上的矩阵乘单元(D = A × B + C);从 Volta V100 引入,Hopper SM90 上每 sub-partition 1 个,共 132 SM × 4 = 528 个 TC;支持 FP16/BF16/TF32/FP8/INT8。
- **§2:** **Mermaid `flowchart LR`** 画一个 sub-partition 的 TC 数据通路:regfile → A/B operand collector → TC array → accumulator → regfile;说明 m16n8k16 fragment 形状。Hopper TC FP16 峰值 989 TFLOPS、FP8 1979 TFLOPS(SXM5)。
- **§3:** `nvcuda::wmma::fragment<...>`(C++ 高层 API)、`mma_sync(d, a, b, c)`、PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0,...}, {%h0,%h1}, {%h2}, {%f4,...};`。FP8 用 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`。
- **§4:** TC 利用率 = mma_inst × ops_per_inst / total_cycles / peak;FP16 单 m16n8k16 = 4096 FMA / cycle peak;矩阵尺寸不对齐时退化(必须按 m/n/k 对齐)。
- **§5:** PTX 片段展示 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ...` 一行;CUDA wmma 片段展示从 SMEM load fragment + mma + store fragment。
- **§6:** NSight Compute `sm__inst_executed_pipe_tensor_op_hmma.sum`、`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`。
- **§7:** 把矩阵 padding 到 16 对齐忘了 zero-init padding 区(产生 NaN)、累加器 dtype 用 FP16 而非 FP32(精度不够)、忘记 `mma.sync` 是 warp-collective(必须 32 thread 一起执行)。
- **§8:** Programming Guide §C.1 (WMMA)、PTX ISA §9.7.13 (mma.sync)、Hopper Whitepaper §Tensor Core。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/07-tensor-core.md
git commit -m "docs(cuda-zh): 07 Tensor Core"
```

---

## Task 11: `08-wgmma-async-matmul.md` — wgmma 异步矩阵乘

**Files:**
- Create: `docs/cuda-zh/08-wgmma-async-matmul.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** Hopper 引入的 warp-group(128 thread = 4 warp)级异步矩阵乘;允许把 mma 与 register/SMEM IO 重叠,显著提升大矩阵 GEMM 吞吐。
- **§2:** **Mermaid `sequenceDiagram`** 画 wgmma pipeline:warpgroup issue `wgmma.mma_async` → TC array 后台执行 → warpgroup 继续做下一阶段 IO → `wgmma.commit_group` 标记 → `wgmma.wait_group N` 等待至 N 组未完成。
- **§3:** PTX `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f0,...}, desc-a, desc-b, p, 1, 1, 0, 0;` (其中 desc 是 64-bit smem matrix descriptor);`wgmma.fence.sync.aligned`;`wgmma.commit_group.sync.aligned`;`wgmma.wait_group.sync.aligned 0`。
- **§4:** wgmma 峰值同 mma.sync(因共享 TC 硬件),但 IO/compute overlap 提升真实利用率到 85%+;m64n128k16 是 H100 推荐 fragment;commit_group 的延迟 1-2 cycle。
- **§5:** 完整 wgmma main loop 片段:`wgmma.fence` + 多个 `wgmma.mma_async` + `wgmma.commit_group` + `wgmma.wait_group`。
- **§6:** NSight Compute `sm__inst_executed_pipe_tensor_op_hmma_qmma.sum`(wgmma 计入)、`sm__warps_eligible.avg.pct_of_peak_sustained_active`(warpgroup 等待时降低)。
- **§7:** 忘记 `wgmma.fence` 让 mma 提前看到旧 SMEM、误用 thread 级 `mma.sync` 替代 wgmma(吞吐降一半)、commit_group 不配合 wait_group 导致 group 累积。
- **§8:** PTX ISA §9.7.14 (wgmma.mma_async)、Hopper Whitepaper §Async Mma、CUTLASS 3.x WGMMA examples on github.com/NVIDIA/cutlass。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/08-wgmma-async-matmul.md
git commit -m "docs(cuda-zh): 08 wgmma 异步矩阵乘"
```

---

## Task 12: `09-tma.md` — TMA(Tensor Memory Accelerator)

**Files:**
- Create: `docs/cuda-zh/09-tma.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** TMA 是 Hopper 引入的硬件 DMA 引擎;一条 PTX 指令把多维 tensor 的 box 从 GMEM 异步搬到 SMEM,期间 warp 可继续执行别的;之前需要每 thread 算地址 + 多次 ld.global,现在一条搞定。
- **§2:** **Mermaid `sequenceDiagram`** 画 TMA load 时序:CPU 准备 `CUtensorMap` → kernel 内 `cp.async.bulk.tensor` 提交 → TMA engine 后台从 GMEM/L2 搬数据 → 写完 SMEM 后 `mbarrier.expect_tx` 减 → 用户线程 `mbarrier.try_wait` 取走。
- **§3:** Host 端 `cuTensorMapEncodeTiled(...)`(driver API)生成 `CUtensorMap`;kernel 内 `cp.async.bulk.tensor.5d.global.shared::cluster.tile.mbarrier::complete_tx::bytes [%dst], [%tensor_map, {coords}], [%mbar];`。Swizzle: 32B / 64B / 128B 三种,匹配 SMEM bank。
- **§4:** TMA 一次最大 5D box(任一维 ≤ 256);带宽与 SM 数无关(全局 TMA 引擎);典型 256x256 BF16 box ~32 KiB ~50 cycle dispatch + ~200 cycle 完成。
- **§5:** PTX 片段展示一个 2D 16x256 BF16 tile 的 `cp.async.bulk.tensor.2d ...` + `mbarrier.expect_tx 8192`。
- **§6:** NSight Compute `sm__pipe_tensor_load_async_cycles_active`、`sm__inst_executed_pipe_tensor_load.sum`。
- **§7:** 忘记设置 `mbarrier.expect_tx` 字节数(wait 永远卡住)、`CUtensorMap` swizzle 与 SMEM 访问不匹配(bank conflict)、kernel 重复设 tensor_map(应该 host 一次性)。
- **§8:** PTX ISA §9.7.16 (cp.async.bulk.tensor)、Driver API `cuTensorMapEncode*`、Hopper Whitepaper §TMA。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/09-tma.md
git commit -m "docs(cuda-zh): 09 TMA"
```

---

## Task 13: `10-mbarrier.md` — mbarrier 异步屏障

**Files:**
- Create: `docs/cuda-zh/10-mbarrier.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** mbarrier 是 SMEM 中的 64-bit 屏障对象,可被多个 thread 异步 arrive 并 phase 翻转 wait;TMA 等异步引擎用它通知完成;比传统 `__syncthreads()` 更细粒度。
- **§2:** **Mermaid `stateDiagram-v2`** 画 mbarrier 状态机:Init(expected=N, arrived=0, phase=0)→ Arrive(arrived++)→ 满足后 Phase Flip(arrived=0, phase++)→ Wait 检测到 phase 翻转返回。
- **§3:** PTX `mbarrier.init.shared.b64 [%mbar], 32;`(初始 expected=32)、`mbarrier.arrive.shared.b64 %p, [%mbar];`(返回当前 phase token)、`mbarrier.try_wait.shared.b64 %ok, [%mbar], %phase, %time;`、`mbarrier.expect_tx.shared.b64 [%mbar], 8192;`(用于 TMA)。
- **§4:** mbarrier 单条指令 ~5 cycle;phase 翻转后旧 wait token 失效;支持 expected count 动态修改(`mbarrier.complete_tx`)。
- **§5:** 一个 producer/consumer 双缓冲流水线 PTX 片段:producer warp 做 TMA → expect_tx → consumer warp `mbarrier.try_wait` 后消费。
- **§6:** NSight Compute `smsp__inst_executed_op_mbar.sum`、`smsp__warp_cycles_per_issue_active`。
- **§7:** 重用 mbarrier 但忘记 phase 翻转后 token 失效、`expect_tx` 字节数不匹配 TMA 实际搬运量(死锁)、跨 cluster 用 `mbarrier.shared` 而非 `mbarrier.shared::cluster`。
- **§8:** PTX ISA §9.7.12 (mbarrier)、Hopper Whitepaper §Async Pipeline Barrier、libcu++ `<cuda/barrier>` 高层封装。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/10-mbarrier.md
git commit -m "docs(cuda-zh): 10 mbarrier 异步屏障"
```

---

## Task 14: `11-thread-block-cluster.md` — Thread Block Cluster (CGA)

**Files:**
- Create: `docs/cuda-zh/11-thread-block-cluster.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** SM90 引入 Thread Block Cluster(也叫 Cooperative Grid Array,CGA);1-16 个 CTA 组成一个 cluster,被调度到同一 GPC 的相邻 SM 上,可互相访问对方的 SMEM (DSMEM)。突破单 SM SMEM 容量 + 单 SM 计算能力。
- **§2:** **Mermaid `flowchart TB`** 画 cluster 拓扑:GPC 容纳多 SM,1 个 cluster ≤ 16 CTA 落在同 GPC 的 SM 上;cluster 内任 CTA 通过 distributed shared memory(DSMEM)地址访问其他 CTA SMEM;cluster barrier 硬件协调跨 SM 同步。
- **§3:** `__cluster_dims__(cx, cy, cz)`、`cooperative_groups::cluster_group cg = this_cluster();`、`cg.sync()`、`cg.map_shared_rank(smem_ptr, target_rank)`(DSMEM 地址转换)。
- **§4:** cluster max size 16 CTA(当前编译 + driver 限制 8);DSMEM 访问延迟 ~25 cycle(同 GPC SM-to-SM);cluster barrier ~10 cycle。
- **§5:** PTX `mapa.shared::cluster.u32 %dst_addr, %src_smem, %target_cta;` 后接 `ld.shared::cluster.u32 %r0, [%dst_addr];`;`barrier.cluster.arrive`;`barrier.cluster.wait`。
- **§6:** NSight Compute `smsp__inst_executed_op_dsmem_ld.sum`、`smsp__inst_executed_op_dsmem_st.sum`、`smsp__inst_executed_op_cluster_barrier.sum`。
- **§7:** cluster size 超 8 但 driver 不支持(运行时报错)、DSMEM 访问跨 cluster(地址翻译失败)、cluster barrier 误以为是 grid-wide(只在 cluster 内)。
- **§8:** Programming Guide §5.2.7 (Thread Block Clusters)、§K.7.7 (Hopper Cluster)、Hopper Whitepaper §Cluster。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/11-thread-block-cluster.md
git commit -m "docs(cuda-zh): 11 Thread Block Cluster"
```

---

## Task 15: 验证 G3 + tag

- [ ] **Step 1: 批量验证**

```bash
for f in docs/cuda-zh/0[7-9]-*.md docs/cuda-zh/1[0-1]-*.md; do
    echo "=== $f ==="
    grep -c "^## [1-8]\. " "$f"
    grep -c '^```mermaid' "$f"
    ! grep -i 'gpusim' "$f" >/dev/null && echo "  no gpusim ref"
done
```

- [ ] **Step 2: Tag**

```bash
git tag cuda-zh-G3-complete
```

---

# 里程碑 G4 — 调度 + Stream + 多 GPU + Graph + 持久化

## Task 16: `12-cta-scheduling-gigathread.md` — CTA 调度 + GigaThread

**Files:**
- Create: `docs/cuda-zh/12-cta-scheduling-gigathread.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** GigaThread 是 GPU 顶层硬件调度器,把 grid 的 CTA 分发到 132 个 SM;影响 occupancy / fairness / priority dispatch;cluster 模式下要按 GPC 整组发。
- **§2:** **Mermaid `flowchart TB`** 画 dispatch 路径:Driver/Runtime 提交 grid → GigaThread Engine → per-SM CTA queue → SM 内 sub-partition 取 warp 执行。说明 occupancy 计算 = min(8 CTA, regfile, smem, max_warps)。Hopper 单 SM 64 warp 上限。
- **§3:** `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSm)`、`cudaOccupancyMaxActiveBlocksPerMultiprocessor`、`cudaOccupancyMaxPotentialBlockSize`、`cudaFuncSetAttribute(...cudaFuncAttributePreferredClusterDimension...)`。
- **§4:** GigaThread 1 cycle/dispatch(单 SM 接收最多 1 CTA/cycle);cluster grid 必须按 GPC 同时分发(全部就位才能开始);priority stream 影响发顺序但不影响 SM 内调度。
- **§5:** CUDA C++ 片段:`int blockSize, minGridSize; cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelFn, 0, 0); kernelFn<<<minGridSize, blockSize>>>(...);`
- **§6:** NSight Compute `gpc__cycles_active.sum`、`smsp__warps_active.avg.pct_of_peak_sustained_active`、`launch__waves_per_multiprocessor`。
- **§7:** "tail effect"(grid 不被 SM 数整除导致最后一波只占少数 SM)、忽略 cluster grid 必须 GPC 整除、过 small CTA(浪费 occupancy 槽)。
- **§8:** Programming Guide §5.2.5 (Compute Capability)、§B.20 (Launch Bounds)、Best Practices §10。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/12-cta-scheduling-gigathread.md
git commit -m "docs(cuda-zh): 12 CTA 调度 + GigaThread"
```

---

## Task 17: `13-streams-and-events.md` — CUDA Streams + Events

**Files:**
- Create: `docs/cuda-zh/13-streams-and-events.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** CUDA Stream 是设备端命令的有序队列;不同 stream 间无序可并发;Event 是 stream 上的时间戳/同步标记;通过 record/wait 实现跨 stream 依赖。
- **§2:** **Mermaid `sequenceDiagram`** 画 2 stream + 1 event 的协调:streamA launch K1 → record(ev) → ; streamB wait(ev) → launch K2(等 K1)。Hopper 同时活跃 stream 上限受 hyper-q 限制(128)。
- **§3:** `cudaStreamCreate / cudaStreamCreateWithPriority`、`cudaStreamAttachMemAsync`、`cudaEventCreate / cudaEventRecord / cudaStreamWaitEvent`、`cudaEventElapsedTime`、`cudaStreamSynchronize`、`cudaStreamWaitAll`(via 多 wait_event)。
- **§4:** 默认 NULL stream 阻塞所有别的 stream(legacy);per-thread default stream(`--default-stream per-thread`)解锁;event record/wait ~1 µs 主机开销 + ~50 cycle 设备开销。
- **§5:** 一段双 stream 重叠 H2D copy / kernel / D2H copy 的标准 ping-pong 代码。
- **§6:** NSight Systems 时间线观察 stream lane;Compute 不直接报 stream metric。
- **§7:** 用默认 stream 期望并发(被串行化)、忘记 `cudaStreamSynchronize` 导致 host code 提前继续、Event 在不同 device 上 record/wait(未启用 P2P 时报错)。
- **§8:** Programming Guide §3.2.6 (Concurrent Execution)、§3.2.6.5 (Streams)、§3.2.6.6 (Events)、Best Practices §9.1.2。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/13-streams-and-events.md
git commit -m "docs(cuda-zh): 13 CUDA Streams + Events"
```

---

## Task 18: `14-nvlink-nvswitch.md` — NVLink + NVSwitch

**Files:**
- Create: `docs/cuda-zh/14-nvlink-nvswitch.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** NVLink 是 GPU 间 / GPU-CPU 间高带宽点对点链路;NVSwitch 是非阻塞 crossbar 把多 GPU 互连成全连接;Hopper 用 NVLink 4.0,SXM5 单 GPU 总带宽 900 GB/s,DGX H100 8 卡通过 NVSwitch 全连接。
- **§2:** **Mermaid `flowchart LR`** 画 8-GPU DGX 拓扑:8 个 GPU,每个 18 link 各连到 4 个 NVSwitch;NVSwitch 跨 chassis 扩到 256 GPU 系统(NVL36/NVL72)。
- **§3:** `cudaDeviceEnablePeerAccess`、`cudaMemcpyPeerAsync`、`cudaMemAdvise(...cudaMemAdviseSetAccessedBy, deviceId)`(unified memory 优化)、NCCL 自动用 NVLink。
- **§4:** NVLink 4 单 link 25 GB/s 双向 = 50 GB/s,SXM5 18 link → 900 GB/s;NVSwitch 3 引入 SHARP(in-network reduction)把 allreduce 带宽再乘 2。
- **§5:** 一段两 GPU 之间 P2P `cudaMemcpyPeerAsync` 转账的代码 + 启用 P2P 的 setup。
- **§6:** `nvidia-smi nvlink --status`、`nvidia-smi nvlink -gt c -i 0`(吞吐计数)、NSight Systems 自动展示 NVLink 流量。
- **§7:** 忘记 `cudaDeviceEnablePeerAccess`(DMA 走 PCIe 退化 100×)、跨 NUMA host 拷贝走慢路径、misalign address 让 NVLink 退到 32 B 事务。
- **§8:** NVLink Architecture Whitepaper、NVSwitch Whitepaper、Programming Guide §3.2.5 (Peer-to-Peer Memory Access)。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/14-nvlink-nvswitch.md
git commit -m "docs(cuda-zh): 14 NVLink + NVSwitch"
```

---

## Task 19: `15-nccl-collectives.md` — NCCL 集合通信

**Files:**
- Create: `docs/cuda-zh/15-nccl-collectives.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** NCCL(NVIDIA Collective Communications Library)是多 GPU / 多节点的高性能 collective 实现;PyTorch DDP/FSDP、JAX pjit 全靠它;支持 ring/tree/SHARP 多种算法。
- **§2:** **Mermaid `flowchart LR`** 画 ring allreduce 8 GPU 的步骤:N-1 round 的 reduce-scatter(每 round 1/N 块流转)+ N-1 round allgather;说明带宽 = `2(N-1)/N · M / B`,N→∞ 趋近 2M/B。
- **§3:** `ncclCommInitRank / ncclCommInitAll`、`ncclAllReduce(sbuf, rbuf, count, dtype, op, comm, stream)`、`ncclReduceScatter / ncclAllGather / ncclSendRecv`、`ncclGroupStart / ncclGroupEnd`(批量异步)。
- **§4:** ring allreduce bus bandwidth ≈ 2 NVLink BW;tree allreduce 在小消息上更优(log N latency);SHARP allreduce 把 reduce 卸到 NVSwitch,带宽再乘 2。
- **§5:** 一段标准的 NCCL allreduce 模板代码,包含 init、单次 op、destroy。
- **§6:** `NCCL_DEBUG=INFO` 环境变量、NCCL 自带的 `ncclProfiler` 钩子、NSight Systems 可见 NCCL 内核。
- **§7:** 不同 rank 调 NCCL 顺序不一致(死锁)、用错 comm 实例(跨 group 串数据)、忘 `ncclGroupStart/End` 让多 op 串行化。
- **§8:** github.com/NVIDIA/nccl(README)、NCCL User Guide(docs.nvidia.com/deeplearning/nccl)、SHARP Whitepaper。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/15-nccl-collectives.md
git commit -m "docs(cuda-zh): 15 NCCL 集合通信"
```

---

## Task 20: `16-cuda-graphs.md` — CUDA Graphs

**Files:**
- Create: `docs/cuda-zh/16-cuda-graphs.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** CUDA Graph 把一系列 stream 操作捕获为 DAG,实例化后能 launch 复用,显著降低 host launch 开销(尤其训练 step 内 launch 上百个小 kernel 时)。
- **§2:** **Mermaid `flowchart LR`** 画一个典型 capture 出来的 DAG:K1 → K2 → memcpy → K3 + K4 并行 → K5;说明 instantiate 后 launch 一条命令完成。
- **§3:** 显式 `cudaGraphCreate / cudaGraphAddKernelNode / cudaGraphAddMemcpyNode / cudaGraphAddDependencies`;捕获 `cudaStreamBeginCapture(stream, mode)` / `cudaStreamEndCapture(stream, &graph)`(mode: global/thread/relaxed);`cudaGraphInstantiate(&exec, graph, ...)`、`cudaGraphLaunch(exec, stream)`、`cudaGraphExecUpdate(exec, newGraph, ...)`、12.4+ conditional node `cudaGraphAddNode(...cudaGraphNodeTypeConditional...)`。
- **§4:** launch 开销从 ~5 µs 降到 ~1 µs(一个 instantiate 后的 launch);适合每 iter ≥ 50 launch 的训练循环;capture mode 决定多线程多 stream 行为。
- **§5:** 标准 stream capture + replay 模板:`cudaStreamBeginCapture` → 一段 stream API 调用 → `cudaStreamEndCapture` → `cudaGraphInstantiate` → `for(...) cudaGraphLaunch`。
- **§6:** NSight Systems 自动展示 graph 节点 + replay 时间线;`cudaGraphDebugDotPrint` 导 .dot 文件可视化。
- **§7:** capture 期间调非 capture-safe API(如 cudaMalloc)、conditional node 在 < 12.4 上用、忘记 `cudaGraphInstantiate` 直接 launch graph。
- **§8:** Programming Guide §3.2.8 (CUDA Graphs)、Driver API `cuGraph*`、Best Practices §11。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/16-cuda-graphs.md
git commit -m "docs(cuda-zh): 16 CUDA Graphs"
```

---

## Task 21: `17-persistent-and-dynamic-parallelism.md` — Persistent + Dynamic Parallelism

**Files:**
- Create: `docs/cuda-zh/17-persistent-and-dynamic-parallelism.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** Persistent kernel 是长生命周期 grid,grid-stride loop 从工作队列取任务,避免反复 launch;Dynamic Parallelism 让 device 端在 kernel 内部 launch child kernel,支持递归 / 数据相关并行。
- **§2:** **Mermaid `sequenceDiagram`** 画 persistent kernel 工作流:host launch persistent grid(占满 SM)→ host 不断 push work item → persistent CTA atomic-pop work → 处理 → 循环 → host 发停止信号。Dynamic parallelism: parent kernel `cudaLaunchKernelEx(...)` → child grid 在另 SM 跑 → parent 可选 wait。
- **§3:** Persistent: 用户自己写 grid-stride + work queue(无专门 API);DP: `cudaLaunchKernelEx(&attribs, fn, ...)`(CUDA 12+ 推荐)、historic `<<<>>>` from device(deprecated);`cudaStreamFireAndForget`。
- **§4:** Persistent 适合不规则 / 长尾 workload;launch 摊销 0 开销;DP child launch ~10 µs(比 host launch 高);cluster 内 DP 受限。
- **§5:** Persistent kernel 模板:`__global__ void server(WorkQueue* q) { while(true) { item = atomicPopWork(q); if (no_more) return; process(item); } }`;DP 模板:`__global__ void parent(...) { ...; cudaLaunchKernelEx(&a, child, n, ...); }`。
- **§6:** persistent: NSight Systems 看到 grid 持续运行;DP: `cudaDeviceGetLimit(cudaLimitDevRuntimePendingLaunchCount, ...)` 监控队列深度。
- **§7:** persistent 占满 SM 后其他 stream 完全饿死(必须留 SM 资源);DP 深度递归触发限制(默认 24 层);忘记 `__syncthreads` 让 child 看到不一致状态。
- **§8:** Programming Guide §6.5 (CUDA Dynamic Parallelism)、§3.2.8.7.10 (Capturing Stream API)、CUDA Sample `cdpSimpleQuicksort`。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/17-persistent-and-dynamic-parallelism.md
git commit -m "docs(cuda-zh): 17 Persistent + Dynamic Parallelism"
```

---

## Task 22: 验证 G4 + tag

- [ ] **Step 1: 批量验证**

```bash
for f in docs/cuda-zh/1[2-7]-*.md; do
    echo "=== $f ==="
    grep -c "^## [1-8]\. " "$f"
    grep -c '^```mermaid' "$f"
    ! grep -i 'gpusim' "$f" >/dev/null && echo "  no gpusim ref"
done
```

- [ ] **Step 2: Tag**

```bash
git tag cuda-zh-G4-complete
```

---

# 里程碑 G5 — 内存管理 + Driver API + 工具链 + 编译

## Task 23: `18-stream-ordered-allocator.md` — Stream-ordered Allocator

**Files:**
- Create: `docs/cuda-zh/18-stream-ordered-allocator.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** `cudaMallocAsync` 让 alloc/free 与 stream 排队,免去同步 + 内核 alloc 的开销;PyTorch / JAX 的 caching allocator 在它之上;Hopper 推荐路径。
- **§2:** **Mermaid `stateDiagram-v2`** 画 block 状态:Allocated → FreeOnStreamA(可被 streamA 重用)→ AfterEvent(可被任意 stream 重用)→ Released(还给 OS)。说明 release_threshold 控制何时还 OS。
- **§3:** `cudaMallocAsync(&p, n, stream)`、`cudaFreeAsync(p, stream)`、`cudaMemPoolCreate(&pool, props)`、`cudaMemPoolTrimTo(pool, minBytesToKeep)`、`cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &v)`、`cudaDeviceSetMemPool`、`cudaMallocFromPoolAsync`。
- **§4:** 同 stream alloc-free-alloc 立即重用(0 µs);跨 stream 需要 event sync;首次 alloc 仍走慢路径(~50 µs);trim 显著降低 reserved 内存。
- **§5:** 一段 PyTorch-like 训练 step 模板:每 iter `cudaMallocAsync` 拿 activation,完成后 `cudaFreeAsync`,iter 间内存被透明复用。
- **§6:** `cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &v)`(当前 in-flight)、`cudaMemPoolAttrReservedMemHigh`(峰值 reserved)、NSight Systems 看 pool 事件。
- **§7:** alloc 在 streamA / free 在 streamB 但没 event 同步(race)、release_threshold 设为 0(频繁 trim 抖动)、用旧的 cudaMalloc 混用(失去 pool 优势)。
- **§8:** Programming Guide §3.2.5.5 (Stream Ordered Memory Allocator)、Driver API `cuMemAlloc*`、Best Practices §15.3。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/18-stream-ordered-allocator.md
git commit -m "docs(cuda-zh): 18 Stream-ordered Allocator"
```

---

## Task 24: `19-unified-memory.md` — Unified Memory

**Files:**
- Create: `docs/cuda-zh/19-unified-memory.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** Unified Memory 让 CPU/GPU 共享一个虚拟地址空间;`cudaMallocManaged` 分配的内存按需迁移(page fault);省去手工 cudaMemcpy,但隐式拷贝可能变慢点。
- **§2:** **Mermaid `sequenceDiagram`** 画 page fault 流程:CPU 写 page(本地)→ GPU launch 触发 page fault → GPU MMU 报 fault → driver 把 page 从 host 迁到 device → GPU 继续。Hopper / Grace 上 ATS(Address Translation Service)让 GPU 直接走 CPU 页表。
- **§3:** `cudaMallocManaged(&p, n, cudaMemAttachGlobal)`、`cudaMemPrefetchAsync(p, n, deviceId, stream)`、`cudaMemAdvise(p, n, cudaMemAdviseSetReadMostly, deviceId)` / `cudaMemAdviseSetPreferredLocation` / `cudaMemAdviseSetAccessedBy`、`cudaStreamAttachMemAsync`(per-stream 可见性)。
- **§4:** Page fault 开销 ~50 µs / page(4 KB);prefetch 提前迁移消除 fault;ReadMostly 启用复制(读多 GPU 都有副本);GH200 因 ATS 直接走 CPU 页表,page fault 显著降低。
- **§5:** 标准模板:`cudaMallocManaged(&p, n); init_on_cpu(p); cudaMemPrefetchAsync(p, n, gpu, stream); kernel<<<...>>>(p);`。
- **§6:** NSight Systems 时间线显示 UM 迁移事件 + 字节;`cudaMemRangeGetAttribute(...cudaMemRangeAttributeReadMostly...)` 查 advice 状态。
- **§7:** 频繁 CPU/GPU 交替写同 page(乒乓迁移)、忘 prefetch 让首次 launch 慢百倍、用 `cudaMemAdviseSetReadMostly` 然后写(强制取消 advice)。
- **§8:** Programming Guide §3.2.4 (Unified Memory)、§K.2 (Unified Memory Programming)、Best Practices §9.2.2.4 (UM Performance)。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/19-unified-memory.md
git commit -m "docs(cuda-zh): 19 Unified Memory"
```

---

## Task 25: `20-cuda-driver-api.md` — CUDA Driver API

**Files:**
- Create: `docs/cuda-zh/20-cuda-driver-api.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** Driver API(`libcuda.so`,`cu*` 前缀)是 CUDA 最底层接口;Runtime API(`libcudart.so`,`cuda*` 前缀)在它之上加 syntactic sugar。直接用 driver API 适合动态加载 cubin / 自带 PTX 的 JIT 框架(如 TensorRT、PyTorch eager-mode 部分)。
- **§2:** **Mermaid `classDiagram`** 画两层 API 的关系:用户代码 → CUDA Runtime API → Driver API → kernel-mode driver → GPU。说明 primary context vs explicit context 的区别。
- **§3:** `cuInit(0)`、`cuDeviceGet`、`cuCtxCreate / cuCtxDestroy`、`cuDevicePrimaryCtxRetain`、`cuModuleLoad / cuModuleLoadData / cuModuleGetFunction`、`cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, smem, stream, kparams, extra)`、`cuMemAlloc / cuMemFree / cuMemcpy*`。
- **§4:** Driver API 调用比 Runtime API 多 1-2 µs(per-call dispatch);primary context 是 thread 间共享的默认上下文(避免显式 push/pop);多 device 切换要 push/pop。
- **§5:** 一段最小 driver API kernel launch 代码:`cuInit / cuDeviceGet / cuDevicePrimaryCtxRetain / cuModuleLoad / cuModuleGetFunction / cuLaunchKernel`。
- **§6:** `nvprof --print-gpu-trace`(显示 driver-level launch);`CUDA_LAUNCH_BLOCKING=1` 环境变量串行化;`cuGetErrorString` 错误诊断。
- **§7:** 忘 `cuInit(0)`(后续全部失败)、Runtime + Driver 混用相同 context 但忘记 retain(销毁后 cuda runtime 还在用)、误用 `cuLaunchKernel` 的 `extra` 参数(应只传 NULL 或固定 metadata)。
- **§8:** CUDA Driver API Reference(docs.nvidia.com/cuda/cuda-driver-api)、Programming Guide §3.4 (Interoperability between Runtime and Driver APIs)、CUDA Sample `vectorAddDrv`。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/20-cuda-driver-api.md
git commit -m "docs(cuda-zh): 20 CUDA Driver API"
```

---

## Task 26: `21-profiling-toolchain.md` — Profiling 工具栈

**Files:**
- Create: `docs/cuda-zh/21-profiling-toolchain.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** NSight 套件是 NVIDIA 官方 profile 工具;NSight Systems 看时间线 + 系统级开销,NSight Compute 看单 kernel 指标;NVTX 让用户标记的 range 出现在时间线;CUPTI 是 callback/activity API,所有第三方 profiler 基于它。
- **§2:** **Mermaid `flowchart TD`** 画 profile 工具栈:用户代码 + NVTX → CUDA Runtime → Driver → GPU;CUPTI 钩子从 Driver 收 activity;NSight Systems / Compute / 第三方 profiler 都基于 CUPTI。
- **§3:** `nvtxRangePushA("name") / nvtxRangePop`、`nvtxMarkA`、`#include <nvtx3/nvToolsExt.h>`;CUPTI `cuptiActivityRegisterCallbacks`(高级,一般不直接用)。
- **§4:** NSight Systems overhead < 5%;Compute 是 kernel replay(同一 kernel 跑多次取不同 metric),overhead 大但不走真路径;NVTX range push/pop ~50 ns/call。
- **§5:** 命令行模板:`nsys profile -t cuda,nvtx -o myrun python train.py`;`ncu --target-processes all --set full -o report ./app`;C++ 中 `nvtxRangePushA("forward"); model.forward(); nvtxRangePop();`。
- **§6:** `nsys stats myrun.nsys-rep`(命令行汇总)、`ncu --print-summary --import report.ncu-rep`、Compute 内置的 source code attribution。
- **§7:** 在生产环境留着 Compute(replay 改语义)、用 nsys 分析单 kernel(应该用 Compute)、忘记 `cudaProfilerStart/Stop` 控制 profile 范围(全程 profile 数据爆炸)。
- **§8:** NSight Systems User Guide(docs.nvidia.com/nsight-systems)、NSight Compute User Guide、NVTX 3 (github.com/NVIDIA/NVTX)、CUPTI Reference。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/21-profiling-toolchain.md
git commit -m "docs(cuda-zh): 21 Profiling 工具栈"
```

---

## Task 27: `22-ptx-to-sass.md` — PTX → SASS 编译链

**Files:**
- Create: `docs/cuda-zh/22-ptx-to-sass.md`

- [ ] **Step 1: 写章节**

要点:
- **§1:** PTX(Parallel Thread eXecution)是 NVIDIA 的虚拟 ISA,跨架构稳定;SASS 是真机指令(每代 SM 不同);ptxas 把 PTX 编译成 SASS,fatbin 打包多 arch SASS。
- **§2:** **Mermaid `flowchart LR`** 画 nvcc pipeline:`.cu → cudafe (split host/device) → cicc (PTX gen) → ptxas (SASS gen) → fatbin (multi-arch)`;运行时 driver 选 fatbin 中匹配的 SASS,缺时 JIT PTX。
- **§3:** nvcc flags `-arch=sm_90 / sm_90a`(`a` 启用 wgmma/TMA 等 feature)、`-gencode arch=compute_90,code=sm_90`、`-Xptxas -O3 / --maxrregcount=N / --ptxas-options=-v`;`cuobjdump --dump-sass binary.cubin`、`nvdisasm binary.cubin`。
- **§4:** ptxas `-O3` 跟 `-O0` SASS 差距巨大(寄存器分配、指令调度);`-arch=sm_90a` 比 `sm_90` 多 wgmma/TMA;JIT 编译开销 ~100 ms / kernel(可缓存)。
- **§5:** 编译 + 反汇编命令模板:`nvcc -arch=sm_90a -ptx kernel.cu -o k.ptx && ptxas -arch=sm_90a -O3 k.ptx -o k.cubin && cuobjdump --dump-sass k.cubin`。
- **§6:** `--ptxas-options=-v` 显示 register / smem 用量;Compute 显示 SASS-level source attribution;`nvdisasm -hex binary.cubin` 看 hex bytes。
- **§7:** `-arch=sm_90` 想跑 wgmma(应 `sm_90a`)、关掉 `-O3` 后基准比真实生产慢 5×、忘记 cuda_compute_capability 与运行时 device 匹配。
- **§8:** PTX ISA Reference(docs.nvidia.com/cuda/parallel-thread-execution)、CUDA Compiler Driver NVCC、Inline PTX in CUDA C++。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/22-ptx-to-sass.md
git commit -m "docs(cuda-zh): 22 PTX → SASS 编译链"
```

---

## Task 28: 索引补全 + 全集验证 + 最终 tag

**Files:**
- Modify: `docs/cuda-zh/00-index.md`(补全章节链接表)

- [ ] **Step 1: 补全 00-index.md 的章节列表**

打开 `docs/cuda-zh/00-index.md` §8 节,确认章节链接表 22 项全在并指向正确文件。如果 G1 任务里还没全列上(因写的时候后续章节还不存在),现在补:

```markdown
## 8. 延伸阅读

### 本教程章节索引

| # | 标题 | 主题 |
|---|---|---|
| [01](01-simt-execution.md) | SIMT 执行模型 | warp / divergence / SIMT stack |
| [02](02-sm-internals.md) | SM 内部结构 | sub-partition / regfile / functional units |
| [03](03-smem-and-l1.md) | 共享内存 + L1 | unified L1+SMEM / 32 banks |
| [04](04-l2-cache-and-setaside.md) | L2 缓存 + set-aside | 60 MiB L2 / persistence attribute |
| [05](05-hbm3-and-gmem.md) | HBM3 + 全局内存 | channel/bank/row / coalescing |
| [06](06-atomics.md) | 原子操作 | global/shared atomic / red.async |
| [07](07-tensor-core.md) | Tensor Core | mma.sync / FP8/BF16/TF32 |
| [08](08-wgmma-async-matmul.md) | wgmma 异步矩阵乘 | warpgroup-level async mma |
| [09](09-tma.md) | TMA | cp.async.bulk.tensor / 5D box |
| [10](10-mbarrier.md) | mbarrier 异步屏障 | phase 翻转 / expect_tx |
| [11](11-thread-block-cluster.md) | Thread Block Cluster | DSMEM / cluster barrier |
| [12](12-cta-scheduling-gigathread.md) | CTA 调度 + GigaThread | occupancy / launch bounds |
| [13](13-streams-and-events.md) | CUDA Streams + Events | 并发 / 优先级 / event 同步 |
| [14](14-nvlink-nvswitch.md) | NVLink + NVSwitch | P2P / 900 GB/s / SHARP |
| [15](15-nccl-collectives.md) | NCCL 集合通信 | ring/tree allreduce |
| [16](16-cuda-graphs.md) | CUDA Graphs | capture / instantiate / replay |
| [17](17-persistent-and-dynamic-parallelism.md) | Persistent + Dynamic Parallelism | grid-stride / cudaLaunchKernelEx |
| [18](18-stream-ordered-allocator.md) | Stream-ordered Allocator | cudaMallocAsync / mempool |
| [19](19-unified-memory.md) | Unified Memory | cudaMallocManaged / page migration |
| [20](20-cuda-driver-api.md) | CUDA Driver API | libcuda / context / cuLaunchKernel |
| [21](21-profiling-toolchain.md) | Profiling 工具栈 | NSight / CUPTI / NVTX |
| [22](22-ptx-to-sass.md) | PTX → SASS 编译链 | ptxas / sm_90a / cuobjdump |

### 官方参考

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [NSight Systems / Compute User Guides](https://docs.nvidia.com/nsight-systems/)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/)
```

- [ ] **Step 2: 全集验证脚本**

```bash
cd docs/cuda-zh
echo "=== 文件清单 ==="
ls -1 *.md | wc -l    # 应输出 23

echo ""
echo "=== 每章统计 ==="
for f in $(ls -1 *.md | sort); do
    sections=$(grep -c "^## [1-8]\. " "$f")
    mermaid=$(grep -c '^```mermaid' "$f")
    zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding='utf-8')
text = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', text, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', text)))
" "$f")
    if grep -qi 'gpusim' "$f"; then
        gpusim="GPUSIM_FOUND"
    else
        gpusim="ok"
    fi
    printf "%-50s sections=%d mermaid=%d zh_chars=%d %s\n" "$f" "$sections" "$mermaid" "$zh" "$gpusim"
done

echo ""
echo "=== 总 mermaid 计数(应 ≥ 24)==="
grep -c '^```mermaid' *.md | awk -F: '{s+=$2} END {print s}'
```

- [ ] **Step 3: 验证通过后 commit + tag**

如果脚本输出显示:
- 23 个 md 文件
- 每个文件 8 sections, mermaid ≥ 1, 1500-2500 中文字
- 00-index.md mermaid ≥ 2
- 总 mermaid ≥ 24
- 无 gpusim 命中

那么:

```bash
git add docs/cuda-zh/00-index.md
git commit -m "docs(cuda-zh): 索引补全 22 章交叉链接 + 官方文档参考"
git tag cuda-zh-G5-complete
git tag cuda-zh-complete
```

否则修复未达标章节,重做验证。

---

## 验收准则

教程完成的标准:

- [ ] 23 个 markdown 文件全部存在(`ls docs/cuda-zh/*.md | wc -l` = 23)
- [ ] 每章包含 8 节(`grep -c "^## [1-8]\. "` = 8)
- [ ] 每章 1500-2500 中文字
- [ ] 每章至少 1 个 Mermaid;00 索引 ≥ 2 个;总 mermaid ≥ 24
- [ ] 全部章节零 gpusim 引用(`! grep -i gpusim docs/cuda-zh/*.md`)
- [ ] 5 个里程碑 tag 全到位:`cuda-zh-G1-complete` ... `cuda-zh-G5-complete` + `cuda-zh-complete`
- [ ] 00-index.md §8 提供 22 章交叉链接表
