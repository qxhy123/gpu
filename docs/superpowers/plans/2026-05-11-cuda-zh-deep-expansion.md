# cuda-zh Deep Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有 25 章 cuda-zh 教程扩深到每章 4000-5000 字 + ≥ 2 Mermaid + senior AI Infra 视角(微架构机制 + 真实生产数字 + 失败模式 + 实现导读 + 设计权衡)。

**Architecture:** 6 个里程碑(DG1-DG6),每个 milestone 一个 subagent 一次 dispatch 内部串行扩 2-6 章。subagent 必须先 Read 现有章节,在已有内容基础上加深(不是重写),保持 8 节结构 + Mermaid 强制 + 零 gpusim 等既定规范。最终 push 到 GitHub origin。

**Tech Stack:** Markdown,Mermaid。无代码,无测试 — 验证靠 grep + 字数。

---

## 全局规则(适用于所有任务)

### 字数 + Mermaid 配额
- 每章扩深后 **4000-5000 中文字**(代码块 + Mermaid 不计)
- 每章 **Mermaid ≥ 2**(00 索引章 ≥ 3)
- 全集 Mermaid 总数 **≥ 51**
- 零 gpusim 引用(`grep -i gpusim` 空命中)

### 五类必加内容(每章扩深时分散到 §2 / §4 / §5 / §7)
1. **微架构机制级细节** — 寄存器 bank、WGMMA descriptor bit field、TMA box 编码、mbarrier 64-bit 内部布局、L2 set-aside way-bias、HBM3 row-buffer 调度
2. **真实生产数字 + 案例** — H100 SXM5 实测 / 论文实测,而不是 spec 上限
3. **失败模式 + 调试手段** — production race / deadlock / overflow / ECC,以及诊断方法
4. **实现导读 / 当前前沿** — CUTLASS 3.x / vLLM / TensorRT-LLM / FlashAttention-3 真实源码位置 + 论文 ablation
5. **替代方案 / 设计权衡** — NVIDIA 选 A 不选 B 的具体理由

### 编辑原则
- subagent 必须先 `Read` 现有章节文件,**在原文基础上加深**,不是重写
- 保持 8 节结构 + 既定 §7 标题(常规章为"常见反模式";23/24 capstone 章为"优化方法体系")
- 新增 Mermaid 优先放 §2(微架构图)+ §3 / §5 / §7 任选(实现 / 调用流程 / 优化分类)
- 真实数字必须可追溯:Hopper Whitepaper / CUDA Programming Guide / 公开论文,引用注明
- UTF-8 LF;`#` 仅章名;`##` 仅 §1-§8

### 验证脚本(每章扩完跑)
```bash
F=docs/cuda-zh/NN-xxx.md
echo "8 sections: $(grep -c '^## [1-8]\. ' $F) (expect 8)"
echo "mermaid: $(grep -c '^```mermaid' $F) (expect ≥ 2; 00 expect ≥ 3)"
.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding='utf-8')
text = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', text, flags=re.DOTALL)
zh = re.findall(r'[一-鿿]', text)
print(f'  {len(zh)} 中文字 (expect 4000-5000)')
" $F
! grep -i 'gpusim' $F && echo "no gpusim ref OK"
```

字数若不在 4000-5000:扩展 §2/§5/§7 直到达标(优先扩微架构细节 + 实现导读两段),或裁剪冗余直到不超 5000。

---

# DG1 — 基础(00 索引、01 SIMT、02 SM 内部)

## Task 1: 扩深 00 索引

**Files:**
- Modify: `docs/cuda-zh/00-index.md`

- [ ] **Step 1: 读现有内容**

Run: `cat docs/cuda-zh/00-index.md` 通读;识别现有 2 个 Mermaid + 当前章节链接表 + 阅读路径布局。

- [ ] **Step 2: 加深内容**

按下面要点扩,目标 4000-5000 中文字 + Mermaid ≥ 3:

**§1 增加内容:**
- 加第 4 条阅读路径「按 senior gap 阅读路径」:列出 senior 工程师常缺漏的 6-8 个深入主题(WGMMA descriptor、TMA encoding、mbarrier 内部、L2 set-aside、PagedAttention internals、FP8 training overflow、CUDA Graph capture mode、NCCL stragglers),指引读者跳读对应章节深度段落。
- 加 H100 SXM5 vs PCIe / H200 / GH200 关键差异对照表(SM 数、L2、HBM、NVLink BW、TDP)。

**§2 增加内容(新增 1 个 Mermaid):**
- 在原 Hopper SM90 全景 `flowchart TB` 后,加一个 GPC × 9 / SM × 132 拓扑 Mermaid `flowchart LR`(展示 GPC0..GPC8 横向、每个 GPC 内 SM 数、cluster 边界);
- 解释 cluster 必须 GPC-local 的硬件原因。

**§3 增加内容:**
- 软件栈分层图后追加文字:cuBLASLt / cuDNN v9 / Transformer Engine / Triton / CUTLASS 在该栈中的位置 + 何时直接调它们 vs 调 PyTorch。

**§4 增加表格:**
- H100 SXM5 vs A100 SXM4 vs V100 SXM2 关键峰值对比(FP32 / TC FP16 / TC BF16 / TC FP8 / HBM BW / NVLink BW / SM 数 / L2),每行注明出处。

**§5 增加示例:**
- 在 hello-world kernel 后,增加一个"profile-driven optimization workflow"代码片段 — 包含 nsys profile + ncu metric 提取 + 决策三步走。

**§7 增加内容:**
- 列出 5 个 senior 易踩的"反入门"陷阱:盲信 occupancy 数字、忽略 prefetch / stream 重叠、混用 driver + runtime API 上下文、用 cudaMalloc 不用 mempool、kernel benchmark 代替 end-to-end profile。

**§8 章节索引表:**
- 已有 22 行 + 23/24 行;额外加一段「按主题分组速查」:把 25 章按 5 大主题(基础 / 内存 / 计算 / 调度 / 工具)分组列出,每组 3-6 章。

- [ ] **Step 3: 验证**

```bash
F=docs/cuda-zh/00-index.md
grep -c '^## [1-8]\. ' $F          # expect 8
grep -c '^```mermaid' $F            # expect ≥ 3
.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding='utf-8')
text = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', text, flags=re.DOTALL)
zh = re.findall(r'[一-鿿]', text)
print(f'  {len(zh)} 中文字 (expect 4000-5000)')
" $F
! grep -i 'gpusim' $F && echo "no gpusim ref OK"
```

- [ ] **Step 4: Commit**

```bash
git add docs/cuda-zh/00-index.md
git commit -m "docs(cuda-zh): 00 索引深度扩展 — senior gap 路径 + GPC 拓扑 + 跨代对比"
```

---

## Task 2: 扩深 01 SIMT 执行模型

**Files:**
- Modify: `docs/cuda-zh/01-simt-execution.md`

- [ ] **Step 1: Read 现有章节**

Run: `cat docs/cuda-zh/01-simt-execution.md` 通读。

- [ ] **Step 2: 加深内容(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: 加 ITS 在 Volta+ 的硬件实现 — 每 lane 独立 PC + return PC + 收敛栈;Hopper warp scheduler 的 4-issue 调度 + scoreboard 标志位结构(每 warp 8 bit 资源标志);warp 在 Stalled_Mem / Stalled_Sync / Stalled_Dep 各态的硬件停留计数器。新增 Mermaid `flowchart TD` 画 warp issue → scoreboard 检查 → 资源争用判定路径。
- **§3 接口深化**: `__activemask()` 在 ITS 下的语义陷阱(reconverge 后的 mask 不可信);`cooperative_groups::coalesced_threads()` 的硬件支持;`__syncwarp(mask)` mask 必须严格匹配实际活跃 lane(否则 UB)。
- **§4 性能数字**: warp issue 4/cycle/SM × 132 SM = 528 warp/cycle 峰值;divergent loop 最坏 32× 慢;`__shfl_sync` 实测 5-cycle 延迟;`__ballot_sync` 1-cycle。
- **§5 代码深化**: PTX 谓词分支 + Volta+ 的 SSY/SYNC SASS 反汇编片段对比;一个 warp-reduce 的标准 implementation 与 cooperative_groups 简化版对比。
- **§7 反模式深化**: `__activemask` 缓存后被 reconverge 失效;无 mask 的 `__shfl` 在编译器 reorder 后乱序;misaligned divergent control flow 导致 nested SSY 栈溢出。
- **§8 增加引用**: Volta ITS Whitepaper 具体页;CUDA Programming Guide §5.4.4;PTX ISA §8。

- [ ] **Step 3: 验证 + Commit**

```bash
git add docs/cuda-zh/01-simt-execution.md
git commit -m "docs(cuda-zh): 01 SIMT 深度扩展 — ITS 硬件实现 + scoreboard + warp scheduler"
```

---

## Task 3: 扩深 02 SM 内部结构

**Files:**
- Modify: `docs/cuda-zh/02-sm-internals.md`

- [ ] **Step 1: Read 现有章节**

Run: `cat docs/cuda-zh/02-sm-internals.md`

- [ ] **Step 2: 加深内容(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: 4 sub-partition 的 64K regs / 16K per scheduler 实际分配规则;LD/ST 单元 16-wide 处理 32-wide warp 的 2-cycle 拆分;TC 与 FP32 ALU issue port 共享情况(同 cycle 不能同 issue);`.maxnreg=255` 上限的硬件原因(8-bit register field)。新增 Mermaid `flowchart TB` 画 sub-partition 内部:warp scheduler → instruction buffer → operand collector → execute pipe(FP32 / TC / LD/ST / SFU)→ writeback。
- **§3 接口深化**: `__launch_bounds__(maxThreads, minBlocksPerSM)` 第二参数告诉 ptxas 控制 register cap 以达到 occupancy 目标;`cudaFuncGetAttributes` 查实际 register / smem 用量;`cudaFuncSetAttribute` 调 `cudaFuncAttributePreferredClusterDimension`。
- **§4 性能数字**: 真实 register pressure 案例 — GPT 70B forward 大 GEMM kernel 用 232 regs/thread,occupancy 限到 25%;DSMEM 跨 SM 访问 ~25 cycle;regfile bank conflict 让 mma issue 慢 2 cycle。
- **§5 代码深化**: PTX `.maxntid 256` + `.maxnreg 96` 控制 ptxas 的实例;CUDA C++ 用 `cudaOccupancyMaxPotentialBlockSize` 自动选 block size 的实战。
- **§7 反模式深化**: `--maxrregcount=32` 强压寄存器数导致大量 spill 到 local memory(L1 cache);忽略 sub-partition 调度独立性导致 warp 不平衡;混用 explicit kernel + cluster grid 但没设 cluster dim。
- **§8**: CUDA Programming Guide §K.7.1-§K.7.5 (Hopper SM 详细);Best Practices §10。

- [ ] **Step 3: 验证 + Commit**

```bash
git add docs/cuda-zh/02-sm-internals.md
git commit -m "docs(cuda-zh): 02 SM 内部深度扩展 — sub-partition 资源分配 + register pressure 案例"
```

---

## Task 4: DG1 验证 + tag

- [ ] **Step 1: 批量验证**

```bash
for f in docs/cuda-zh/0[0-2]-*.md; do
    echo "=== $f ==="
    sec=$(grep -c '^## [1-8]\. ' "$f")
    mer=$(grep -c '^```mermaid' "$f")
    zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding='utf-8')
text = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', text, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', text)))
" "$f")
    grep -qi 'gpusim' "$f" && g="GPUSIM_FOUND" || g="ok"
    printf "  sections=%d mermaid=%d zh=%d %s\n" "$sec" "$mer" "$zh" "$g"
done
```
Expected: 3 章全部 sections=8、mermaid≥2(00 章 ≥ 3)、zh ∈ [4000,5000]、gpusim=ok。

- [ ] **Step 2: Tag**

```bash
git tag cuda-zh-deep-DG1-complete
```

---

# DG2 — 内存层级(03 SMEM+L1, 04 L2, 05 HBM3, 06 atomics)

## Task 5: 扩深 03 共享内存 + L1

**Files:**
- Modify: `docs/cuda-zh/03-smem-and-l1.md`

- [ ] **Step 1: Read 现有章节**

Run: `cat docs/cuda-zh/03-smem-and-l1.md`

- [ ] **Step 2: 加深内容(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: SMEM swizzle 模式 32B / 64B / 128B 的 bank-to-byte 映射规则,与 wgmma fragment(MN-major / K-major)对齐时的等价性;unified L1+SMEM 在不同 carveout 下的 cycle latency 差异;`cp.async` 异步拷贝的 staging buffer 与 SMEM 写回路径。新增 Mermaid `flowchart LR`:`cp.async` 走 GMEM → L2 → SMEM 的旁路 LSU 通路。
- **§3 接口深化**: `cp.async.cg.shared.global` (cache global)vs `cp.async.ca.shared.global` (cache all,过 L1)的选择;`cp.async.commit_group` + `cp.async.wait_group N`;`cudaFuncSetAttribute(cudaFuncAttributePreferredSharedMemoryCarveout, 100)` 的 percentage 含义。
- **§4 性能数字**: SMEM 单 bank 1 cycle / 4B = 32 word/cycle 峰值;real-world double-buffer GEMM 实测吞吐 vs 单 buffer 1.6×;wgmma 用 128B-swizzle SMEM tile 比无 swizzle 快 30%。
- **§5 代码深化**: 一段 producer-consumer 双缓冲 pipeline PTX 片段(cp.async + mbarrier);`+1` padding 矩阵转置 vs swizzle 转置的对比示例。
- **§7 反模式深化**: 用 32B swizzle 但 wgmma 期望 128B(silent slow);忘记 `cp.async.wait_group` 让 SMEM 数据未到位被读;double buffer phase 翻转错误导致脏读。
- **§8**: PTX ISA §9.7.10 (cp.async)、Programming Guide §K.7.4 (Hopper SMEM carveout)、CUTLASS Swizzle code 引用。

- [ ] **Step 3: 验证 + Commit**

```bash
git add docs/cuda-zh/03-smem-and-l1.md
git commit -m "docs(cuda-zh): 03 SMEM+L1 深度 — swizzle + cp.async + double buffer 实测"
```

---

## Task 6: 扩深 04 L2 缓存 + set-aside

**Files:**
- Modify: `docs/cuda-zh/04-l2-cache-and-setaside.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: L2 set-aside 实际是 way-bias(在 16 路组相联中标记某些 way 为 persistent),不是物理切分;persistence attribute 在 LRU 替换时降低受害概率;ECC double-bit error 的 L2 反应路径。新增 Mermaid `flowchart TB` 画 SM access → L2 lookup → tag check → way 选择 → persistence weight → replacement decision。
- **§3 接口深化**: `cudaCtxResetPersistingL2Cache` 的具体语义(清掉所有 persistence flag,不清数据);`cudaStreamSetAttribute` access policy window 的 hitProp / missProp / hitRatio 三参数交互;`cudaDevAttrMaxPersistingL2CacheSize` 查上限。
- **§4 性能数字**: persistence cap 默认 ¼ L2 = 15 MiB(SXM5);hot embedding lookup table 配 persistence 后的 L2 hit rate 从 30% 升到 95%;persistence 与多 stream 争用时的 cap 共享行为。
- **§5 代码深化**: 完整的 embedding lookup 配 persistence 代码 + ResetPersistingL2Cache 时机选择;NSight Compute 用 `lts__t_sector_hit_rate.pct` 验证。
- **§7 反模式深化**: 忘记 reset persistence 导致下个 kernel 命中失效;hitRatio 设 1.0 但 window 超 cap 被截断;多 stream 都设 persistence 互相挤压。
- **§8**: Programming Guide §3.2.3.6 + Best Practices §9.2.3.4 + Hopper Whitepaper §L2。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/04-l2-cache-and-setaside.md
git commit -m "docs(cuda-zh): 04 L2 深度 — way-bias 机制 + persistence cap 实战"
```

---

## Task 7: 扩深 05 HBM3 + 全局内存

**Files:**
- Modify: `docs/cuda-zh/05-hbm3-and-gmem.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: HBM3 row buffer hit / partial / miss 三态延迟差(50 / 100 / 150 ns);bank-group rotation 调度策略让连续地址命中不同 bank-group 提升并行度;async global load 的 LSU sector tracker 容量(每 SM 大约 32 outstanding)。新增 Mermaid `sequenceDiagram` 画一次 GMEM read 全路径:warp issue → LSU 合并 → L1 lookup → L2 lookup → HBM channel arbiter → bank → row buffer → return。
- **§3 接口深化**: `__ldg`(只读 cache)在 Hopper 上等价 `ld.global.nc`;`__stcs`(streaming)绕过 cache 减少污染;`cuda::pipeline` 与 `cp.async` 的高级封装;`cudaMemcpyAsync` 在 P2P 时走 NVLink 而非 PCIe。
- **§4 性能数字**: SXM5 HBM3 5 stack × 1 TB/s = 5 TB/s 理论;实测 saturate 需要 64+ active warps + coalesced;sector 利用率 1.0 vs 0.25 的 4× 性能差;`dram__throughput.avg.pct_of_peak_sustained_elapsed` 良好水平 80%+。
- **§5 代码深化**: 一段 SoA vs AoS coalesced 对比 + sector 利用率分析;`cuda::memcpy_async` 模板的 producer-consumer 实战。
- **§6 实测深化**: 完整 NSight Compute metric set:`dram__bytes_read.sum`、`dram__sectors_read.sum`、`l1tex__t_sector_pipe_lsu_mem_global_op_ld.sum` 三项做 sector 利用率分析。
- **§7 反模式深化**: stride-N 访问让 sector 利用率 1/N;misalign 8B 让单事务变 2;struct of arrays 错排导致每 lane 走不同 sector。
- **§8**: Programming Guide §3.2.2.1 + §K.7 + Hopper Whitepaper § HBM3 + Best Practices §9.2.1。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/05-hbm3-and-gmem.md
git commit -m "docs(cuda-zh): 05 HBM3+GMEM 深度 — row buffer 三态 + LSU sector tracker + 实测 saturate"
```

---

## Task 8: 扩深 06 原子操作

**Files:**
- Modify: `docs/cuda-zh/06-atomics.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: L2 ALU 的 atomic 串行化粒度 — 同 line 内 atomic 串行,不同 line 并行(per-line per-cycle);`red.async` 是 fire-and-forget,L2 内排队 8 entry 深度;BF16 / FP8 atomic add Hopper 原生硬件支持矩阵(global yes,shared yes);CAS-based atomic for unsupported dtype(如 INT4 atomic 用 CAS+packing)。新增 Mermaid `flowchart LR` 画 SM atomic issue → L2 ALU 队列 → cache line lock → ALU operation → unlock → ack(或 red.async 无 ack)。
- **§3 接口深化**: `atomicAdd` 多种重载(half、bfloat16、half2、bfloat162);`atomicCAS_block`(SMEM CAS);`red.async.shared::cta` PTX vs `red.global.add.f32` 选择;`__threadfence_system` 跨 GPU + CPU 顺序保证。
- **§4 性能数字**: 单 line atomic 吞吐 ~1 op/cycle/L2 slice;争用 N 个 thread 同址时退化到 N cycle 串行;`red.async` 比 `atom` 快 2-3×(无 ack);SMEM atomic 比 GMEM atomic 快 10×。
- **§5 代码深化**: histogram 三级归约模板:warp shuffle reduce → SMEM atomic → GMEM atomic merge,完整代码 + 性能分析。
- **§6 实测深化**: NSight Compute `lts__t_sectors_atom_red.sum`、`l1tex__t_sectors_pipe_lsu_mem_shared_op_atom.sum`、争用率推断(`lts__t_atom_red_cycles_active / lts__t_atom_red_count`)。
- **§7 反模式深化**: 全 warp atomic 同址(争用爆炸)、用 atomic 替代 reduce(应该 shuffle)、误用 `red.async` 然后立即读结果(没 ack 不可见)。
- **§8**: Programming Guide §B.14 + PTX ISA §8.7.12 + Hopper Whitepaper § Atomics。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/06-atomics.md
git commit -m "docs(cuda-zh): 06 原子操作深度 — L2 ALU 串行化 + red.async 队列 + 三级 histogram 实战"
```

---

## Task 9: DG2 验证 + tag

- [ ] **Step 1: 批量验证 + tag**

```bash
for f in docs/cuda-zh/0[3-6]-*.md; do
    sec=$(grep -c '^## [1-8]\. ' "$f"); mer=$(grep -c '^```mermaid' "$f")
    zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text('utf-8')
t = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', t, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', t)))
" "$f")
    grep -qi 'gpusim' "$f" && g="FOUND" || g="ok"
    printf "%s: sec=%d mer=%d zh=%d %s\n" "$f" "$sec" "$mer" "$zh" "$g"
done

git tag cuda-zh-deep-DG2-complete
```

---

# DG3 — TC + Hopper 异步 + Cluster(07-11)

## Task 10: 扩深 07 Tensor Core

**Files:**
- Modify: `docs/cuda-zh/07-tensor-core.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: mma.sync m16n8k16 fragment 在 32 lane 内的 elem 分布(每 lane 持 4 个 a-elem、2 个 b-elem、4 个 c-elem);TC 数据通路 sub-partition 内 1 个 TC × 4 sub-partition × 132 SM = 528 TC;sparsity 2:4 在 Ampere+ 的 INT8/FP16 数据通路(metadata + values)。新增 Mermaid `flowchart LR` 画 mma 一拍数据流:regfile lane × 32 → operand collector → TC array → accumulator → regfile writeback。
- **§3 接口深化**: PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0..%f3}, {%h0,%h1}, {%h2}, {%f0..%f3};` 各操作数对应的 lane → elem 映射表;wmma `<m,n,k,a,b,c>` template 的 fragment 类型;FP8 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` k=32 的原因(2 byte/elem vs 1 byte)。
- **§4 性能数字**: H100 SXM5 mma.sync FP16 989 TFLOPS、BF16 989 TFLOPS、FP8 1979 TFLOPS、FP8 + sparsity 3958 TFLOPS;真实 GEMM 利用率 70-85%(CUTLASS 3.x 实现);非对齐 m/n/k 退化(padding 或慢路径)。
- **§5 代码深化**: 完整 wmma fragment 加载 + mma + store 模板;PTX 直接调 mma.sync 的 32-thread coordinated launch 代码。
- **§7 反模式深化**: padding 后忘记 zero-init(NaN propagation);accumulator 用 FP16 而非 FP32(精度爆炸);`mma.sync` 不是 32-thread 一致执行(divergent warp UB)。
- **§8**: PTX ISA §9.7.13 (mma.sync) + CUTLASS gemm/collective 引用 + Hopper Whitepaper § TC。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/07-tensor-core.md
git commit -m "docs(cuda-zh): 07 TC 深度 — fragment lane 分布 + sparsity + 真实 GEMM 利用率"
```

---

## Task 11: 扩深 08 wgmma 异步矩阵乘

**Files:**
- Modify: `docs/cuda-zh/08-wgmma-async-matmul.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: WGMMA descriptor 64-bit bit-by-bit 拆解 — bits[13:0] base addr (14 bit, 16B-aligned)、bits[29:16] leading dim、bits[45:32] stride、bits[51:49] swizzle (None/32B/64B/128B)、bits[48] base offset;commit_group 队列深度 4(超过则 stall);warp-specialization 模式下 producer warp(TMA) + consumer warp(wgmma)的 mbarrier 协调。新增 Mermaid `sequenceDiagram` 画 producer-consumer pipeline:producer 4 warp → TMA load → mbarrier expect_tx → consumer 4 warp → wgmma.fence → wgmma.mma_async × N → commit_group → wait_group。
- **§3 接口深化**: PTX `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f0..%f63}, %desc-a, %desc-b, %p, 1, 1, 0, 0;` 8 个 modifier 含义(ScaleD/ScaleA/ScaleB/TransA/TransB);`wgmma.fence.sync.aligned` 与普通 fence 区别(强制刷新 SMEM 描述符);`wgmma.commit_group.sync.aligned` 之后必须 `wgmma.wait_group.sync.aligned N` 才能读 accumulator。
- **§4 性能数字**: m64n128k16 FP16 单条 wgmma = 64*128*16*2 = 262144 FMA;commit_group 1-2 cycle 入队;wait_group 0 强同步 ~10 cycle;CUTLASS 3.x 在 H100 实测 GEMM 87% peak。
- **§5 代码深化**: CUTLASS 3.x sm90_collective_mma 风格的 main loop 片段(wgmma.fence + 多 mma_async + commit + wait);warp specialization 的 `__warpgroup_roles` 模式。
- **§7 反模式深化**: forgot wgmma.fence 让 mma 看旧 SMEM;commit/wait 不配对累积超 4 group stall;在非 warp-group 边界(如 thread block != 128*N)调用 wgmma。
- **§8**: PTX ISA §9.7.14 + CUTLASS 3.x sm90 examples + Hopper Whitepaper § Async MMA + Greg Diamos blog on warp-specialization。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/08-wgmma-async-matmul.md
git commit -m "docs(cuda-zh): 08 wgmma 深度 — descriptor bit field + warp-specialization + CUTLASS 3.x 实战"
```

---

## Task 12: 扩深 09 TMA

**Files:**
- Modify: `docs/cuda-zh/09-tma.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: `CUtensorMap` 128-byte 描述符在 GMEM 解码 — 包含 globalAddress、tensorDataType、tensorRank、interleave、swizzle、l2Promotion、oobFill;TMA 跨 cluster (`shared::cluster`)与单 SM (`shared`)路径差;5D box 越界处理 — `cuTensorMapEncodeTiled` 的 `oobFill` 参数(zero-fill / OOB-NaN);TMA L2 cache promote/demote 行为(数据透传 L2)。新增 Mermaid `sequenceDiagram` 画 TMA 异步 load 全流程:host 设 tensor_map → kernel cp.async.bulk.tensor → TMA engine 解 descriptor → L2 lookup → HBM read → SMEM write → mbarrier expect_tx 减 → 用户 try_wait 取走。
- **§3 接口深化**: host `cuTensorMapEncodeTiled(map, dtype, rank, gAddr, gSize[5], gStride[4], boxSize[5], elementStride[5], interleave, swizzle, l2Promotion, oobFill)` 各参数;kernel PTX `cp.async.bulk.tensor.5d.global.shared::cluster.tile.mbarrier::complete_tx::bytes [%dst], [%map, {x,y,z,w,t}], [%mbar];`;cluster TMA store(`cp.async.bulk.tensor.x.shared::cluster`)。
- **§4 性能数字**: 描述符 dispatch ~50 cycle;5D 256x256 BF16 box(128 KiB)L2 命中时完成 ~500 cycle、HBM miss ~2000 cycle;跨 cluster TMA 比单 SM 多 25 cycle(经 GPC 网络)。
- **§5 代码深化**: host C++ 完整 `cuTensorMapEncodeTiled` 调用 + kernel PTX `cp.async.bulk.tensor.2d` + `mbarrier.expect_tx 8192` + `mbarrier.try_wait`;swizzle 选 128B 配合 wgmma 128B 的对齐示例。
- **§7 反模式深化**: forgot expect_tx 字节数(死锁);swizzle mismatch 让 wgmma SMEM 读触发 bank conflict;descriptor 在每 kernel 重新生成(应 host 一次性 reuse)。
- **§8**: PTX ISA §9.7.16 + CUDA Driver API `cuTensorMapEncode*` + Hopper Whitepaper § TMA + CUTLASS sm90 mainloop。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/09-tma.md
git commit -m "docs(cuda-zh): 09 TMA 深度 — 128B descriptor 拆解 + 跨 cluster 路径 + L2 promote"
```

---

## Task 13: 扩深 10 mbarrier 异步屏障

**Files:**
- Modify: `docs/cuda-zh/10-mbarrier.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: mbarrier 64-bit 内部布局 — bits[19:0] pending_tx_byte / 4(20 bit)、bits[39:20] arrived_count(20 bit)、bits[59:40] expected_count(20 bit)、bit[60] phase、bits[63:61] reserved;phase 翻转条件 `arrived == expected && pending_tx == 0`;`try_wait.parity` 检测 phase XOR token;`mbarrier.complete_tx` 用于 TMA 完成时 expect_tx 减;同 mbarrier 多 phase reuse 的硬件保证。新增 Mermaid `stateDiagram-v2` 画 mbarrier 状态机(扩深版):Init → Arriving (arrived++, pending_tx--) → 满足 → Phase Flip (arrived=0, pending_tx=0, phase^=1) → Wait check parity → 通过。
- **§3 接口深化**: PTX `mbarrier.init.shared.b64 [%mbar], 32;`(初始 expected = 32, arrived = 0, phase = 0);`mbarrier.arrive.shared.b64 %p, [%mbar];`(返回 phase token);`mbarrier.try_wait.parity.shared.b64 %ok, [%mbar], %prev_phase, %time;`(timeout 单位 cycle);`mbarrier.expect_tx.shared.b64 [%mbar], 8192;`(TMA 配套,字节数);libcu++ `<cuda/barrier>` 高层封装。
- **§4 性能数字**: arrive 1-2 cycle、try_wait 1 cycle hit / poll 直到通过;phase 翻转后旧 token 自动失效,无需手工 reset;同一 mbarrier 单 kernel 内可重 phase 数十次。
- **§5 代码深化**: 标准双缓冲 producer-consumer pipeline 完整 PTX 模板(producer warp TMA + expect_tx,consumer warp wgmma + arrive,buffer A/B 交替);CUTLASS 3.x 中 mbarrier 实战引用。
- **§7 反模式深化**: phase 翻转后用旧 token try_wait(永远 false);expect_tx 字节数算错(死锁或脏读);跨 cluster 用 `mbarrier.shared` 而非 `mbarrier.shared::cluster`。
- **§8**: PTX ISA §9.7.12 + libcu++ `<cuda/barrier>` + CUTLASS 3.x sm90 pipeline。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/10-mbarrier.md
git commit -m "docs(cuda-zh): 10 mbarrier 深度 — 64-bit bit field + phase 翻转 + 双缓冲实战"
```

---

## Task 14: 扩深 11 Thread Block Cluster

**Files:**
- Modify: `docs/cuda-zh/11-thread-block-cluster.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: GPC 内 SM 分配规则(Hopper SXM5 9 GPC × 18 SM 但实际 enable 132 SM);DSMEM 地址翻译 `mapa.shared::cluster` 把(rank, smem_offset)→ 物理 SM SMEM addr,翻译表在每 SM 的 cluster scheduler;cluster size > 8 的 driver gating(8.6+ 才允许 16);cluster TMA store 跨 SM coalescing 规则。新增 Mermaid `flowchart TB` 画 GPC 内部:1 GPC 含多 SM(SM0..SM17)、SM 之间高带宽 DSMEM 互连(~25 cycle);cluster 必须落在同 GPC,跨 GPC 不允许。
- **§3 接口深化**: `__cluster_dims__(cx, cy, cz)` kernel attribute 与 launch 时 attribute 的优先级;`cooperative_groups::cluster_group cg = this_cluster();`、`cg.sync()`、`cg.map_shared_rank(smem_ptr, rank)`;`cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);`(允许 size > 8);PTX `barrier.cluster.arrive` / `barrier.cluster.wait`;`mapa.shared::cluster.u32`。
- **§4 性能数字**: cluster max size 16 CTA(default 8);DSMEM 访问 ~25 cycle(同 GPC SM-to-SM)、SMEM 本地 ~20 cycle;cluster barrier ~10-15 cycle;cluster CTA 必须全 GPC 就位才能开始,导致比单 CTA grid dispatch 慢。
- **§5 代码深化**: cluster cooperative GEMM 完整代码 — 一个 cluster 的 N 个 CTA 共同处理大 tile,DSMEM 共享 K 维 reduce;`mapa.shared::cluster.u32 %dst, %src_smem, %target_cta;` 后接 `ld.shared::cluster.u32 %r, [%dst];`。
- **§7 反模式深化**: cluster size > 8 但 driver < 11.8(运行时报错);误把 DSMEM 当 unified address 跨 cluster 用(地址翻译失败);cluster grid 不被 GPC 数整除导致 tail effect 加剧。
- **§8**: Programming Guide §5.2.7 + §K.7.7 + Hopper Whitepaper § Cluster + CUTLASS sm90 cluster GEMM。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/11-thread-block-cluster.md
git commit -m "docs(cuda-zh): 11 Cluster 深度 — GPC 内 SM 分配 + DSMEM 翻译 + cluster GEMM 实战"
```

---

## Task 15: DG3 验证 + tag

```bash
for f in docs/cuda-zh/0[7-9]-*.md docs/cuda-zh/1[0-1]-*.md; do
    sec=$(grep -c '^## [1-8]\. ' "$f"); mer=$(grep -c '^```mermaid' "$f")
    zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text('utf-8')
t = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', t, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', t)))
" "$f")
    grep -qi 'gpusim' "$f" && g="FOUND" || g="ok"
    printf "%s: sec=%d mer=%d zh=%d %s\n" "$f" "$sec" "$mer" "$zh" "$g"
done

git tag cuda-zh-deep-DG3-complete
```

---

# DG4 — 调度 + Stream + 多 GPU + Graph + 持久化(12-17)

## Task 16: 扩深 12 CTA 调度 + GigaThread

**Files:**
- Modify: `docs/cuda-zh/12-cta-scheduling-gigathread.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: GigaThread Engine 的 cluster grid dispatch 同步要求(必须全 cluster CTA 同时分发);per-SM CTA queue 深度;preempt 机制(MIG instance 切换 + TimeSlice 上下文 swap)开销 ~微秒级;launch 队列在 driver 层的 inflight cap(`cudaLimitDevRuntimeSyncDepth`)。新增 Mermaid `sequenceDiagram` 画一次 grid launch:host cudaLaunchKernel → driver 入队 → GigaThread 取队 → 按 cluster (若有) → per-SM CTA queue → SM 接收 → sub-partition 取 warp。
- **§3 接口深化**: `__launch_bounds__` 在 ptxas 端的 register cap 推导;`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 的内部计算与 CUDA `cudaOccAvailableDynamicSMemPerBlock` 配套;`cudaFuncSetAttribute(...cudaFuncAttributePreferredClusterDimension...)` 实际在 GigaThread 的 hint 作用;`cudaFuncAttributeNonPortableClusterSizeAllowed` 允许 size > 8。
- **§4 性能数字**: GigaThread 1 CTA dispatch / cycle / SM(132 CTA peak/cycle);tail effect 量化 — grid_size=130, SM=132 → 最后 1 wave 只用 130/132 = 98% 利用;cluster grid 全 GPC 同步开销 ~10-50 cycle;register cap 32 实测 occupancy 翻倍但 spill 让性能反降 30%。
- **§5 代码深化**: `cudaOccupancyMaxPotentialBlockSize` + `cudaFuncGetAttributes` 联用自动选 block size 的实战;cluster grid + non-portable cluster size 16 的 launch 代码。
- **§7 反模式深化**: 默认 occupancy 数字盲信(忽略 DSMEM / cluster 资源);tail effect 在小 batch 推理上的放大;预留 SM 给优先级 stream(persistent kernel 占满)反而饿死所有别的。
- **§8**: Programming Guide §5.2.5 + §B.20 + Best Practices §10 + Hopper Whitepaper § GigaThread。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/12-cta-scheduling-gigathread.md
git commit -m "docs(cuda-zh): 12 CTA 调度深度 — GigaThread dispatch + cluster sync + tail effect 量化"
```

---

## Task 17: 扩深 13 CUDA Streams + Events

**Files:**
- Modify: `docs/cuda-zh/13-streams-and-events.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: hyper-Q 32 hardware queue 与用户 N stream 的 N:1 映射(driver 调度);default-per-thread vs legacy default 的隔离差(legacy 阻塞所有,per-thread 每线程独立);CUDA 12 引入的 `cudaStreamGetCaptureInfo` 用于 capture mode 死锁调试;event 的 GPU 端实现(L2 内的 timestamp 写入 + signal)。新增 Mermaid `flowchart TB` 画 driver stream 调度:用户 N stream → driver 调度器 → 32 hw queue → GigaThread。
- **§3 接口深化**: `cudaStreamCreateWithFlags(...cudaStreamNonBlocking...)`(不被 default stream 阻塞);`cudaStreamCreateWithPriority`(priority 范围由 `cudaDeviceGetStreamPriorityRange` 查);`cudaEventCreateWithFlags(...cudaEventDisableTiming...)`(只 sync 不计时,更快);`cudaStreamWaitEvent` flag `cudaEventWaitDefault` vs `cudaEventWaitExternal`(后者用于 graph capture);`cudaStreamSetAttribute(...cudaStreamAttributeAccessPolicyWindow...)` L2 set-aside per-stream。
- **§4 性能数字**: stream create / destroy ~微秒;event record/wait ~50 cycle device + ~1 µs host;hyper-Q 32 queue 满载后 driver 软调度 overhead 显现;legacy default stream 强阻塞代价(完全串行)。
- **§5 代码深化**: 双 stream H2D / kernel / D2H ping-pong 完整代码 + cudaStreamWaitEvent 跨 stream 协调;`cudaStreamAttachMemAsync` 给 unified memory 指定 stream 可见性的 multi-stream 加速例子。
- **§7 反模式深化**: 默认 stream 期望并发(legacy 模式被串行化);忘 `cudaEventDisableTiming` 让纯 sync event 慢 5×;P2P 跨 device event wait 没启用 enablePeerAccess 直接报错。
- **§8**: Programming Guide §3.2.6 + Best Practices §9.1.2 + CUDA Runtime API stream/event 完整列表。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/13-streams-and-events.md
git commit -m "docs(cuda-zh): 13 Stream 深度 — hyper-Q 映射 + per-thread default + capture debug"
```

---

## Task 18: 扩深 14 NVLink + NVSwitch

**Files:**
- Modify: `docs/cuda-zh/14-nvlink-nvswitch.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: NVLink 4 链路编码(NRZ + RS-FEC 纠错);单 link 25 GB/s 双向 → SXM5 18 link → 900 GB/s 总;NVSwitch 3 内部 SHARP engine 的 reduce 数据通路(把多 GPU 的 partial sum 在交换机内 reduce 成 1 份);DGX H100 8-GPU 全连接拓扑(每 GPU 18 link 分到 4 NVSwitch);NVL36/NVL72 跨 chassis 拓扑(光纤 + spine switch)。新增 Mermaid `flowchart LR` 画 DGX H100 8 GPU + 4 NVSwitch 全连接(每 GPU 与每 NVSwitch 多 link)。
- **§3 接口深化**: `cudaDeviceEnablePeerAccess(peerDev, 0)`(0 是 flag,reserved);`cudaMemcpyPeerAsync`;`cudaMemAdvise(...cudaMemAdviseSetAccessedBy, deviceId)`;NCCL 自动通过 `ncclTransport` 选 NVLink;`nvidia-smi nvlink --status` / `--throughput`。
- **§4 性能数字**: NVLink 4 单向 50 GB/s × 18 = 900 GB/s 总(理论);DGX H100 实测 P2P 带宽 ~880 GB/s(头部);NVSwitch SHARP allreduce 把 ring 算法的 `2(N-1)/N` 系数降到 1.0(即真实 sustained = aggregate / N);跨 chassis NVL72 NVSwitch 网络聚合带宽 PB/s 量级。
- **§5 代码深化**: 启用 P2P + cudaMemcpyPeerAsync 的完整代码;NCCL_TOPO_FILE 自定义拓扑覆盖默认探测;`nvidia-smi nvlink -gt c -i 0` 输出解读 + 流量监控脚本。
- **§7 反模式深化**: 忘 enablePeerAccess(DMA 退到 PCIe 100× 慢);跨 NUMA host pinned 内存导致 PCIe 路径慢;link 训练失败(DGX 启动时检查)忽略报错;NVSwitch SHARP 没正确配置(回退普通 ring)。
- **§8**: NVLink 4 Whitepaper + NVSwitch 3 Whitepaper + Programming Guide §3.2.5 + DGX H100 reference architecture。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/14-nvlink-nvswitch.md
git commit -m "docs(cuda-zh): 14 NVLink/NVSwitch 深度 — link 编码 + SHARP engine + DGX/NVL72 拓扑"
```

---

## Task 19: 扩深 15 NCCL 集合通信

**Files:**
- Modify: `docs/cuda-zh/15-nccl-collectives.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: NCCL ring 算法的 chunk size 自适应(根据 message size 选择,`NCCL_BUFFSIZE` 控制);tree 算法的 binary tree 拓扑 + log N latency;SHARP allreduce 把 reduce 卸到 NVSwitch in-network ALU;NCCL 内部用 `cuStreamWriteValue32` + `cuStreamWaitValue32` 与对端同步;straggler 现象在大集群(数百 GPU)的常见性。新增 Mermaid `sequenceDiagram` 画 ring allreduce 一次 step:N rank 把 buffer 分 N chunk,每 step 每 rank send chunk[(rank-step) % N] 给右邻、recv 给左邻 → reduce-scatter N-1 步 → allgather N-1 步。
- **§3 接口深化**: `ncclCommInitRank(&comm, world_size, uniqueId, rank)` + `ncclGetUniqueId`;`ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum, comm, stream)`;`ncclGroupStart` / `ncclGroupEnd` 批量 op 异步合并;`ncclSend` / `ncclRecv` (P2P);算法选择环境变量 `NCCL_ALGO=Ring/Tree/CollNet`;调试 `NCCL_DEBUG=TRACE` + `NCCL_DEBUG_SUBSYS=ALL`。
- **§4 性能数字**: ring allreduce 拐点 — message < 64 KB 用 tree(log N latency 优势)、≥ 64 KB 用 ring(带宽);SHARP allreduce 实测带宽提升 1.7-2.0×;DGX H100 8-GPU bf16 4 GB allreduce ~5 ms(ring NVLink saturate);跨节点 NDR IB 100 GB/s saturate 也类似数字。
- **§5 代码深化**: 标准 NCCL allreduce 模板 + ncclGroupStart 批多 op + 错误处理;PyTorch DDP 内部如何调 NCCL(`torch.distributed.all_reduce` → C++ ProcessGroupNCCL → ncclAllReduce);多 stream 跨 NCCL op 的同步坑。
- **§7 反模式深化**: 不同 rank 调 NCCL 顺序不一致(死锁);跨 group 用错 comm(数据流串);忘 `ncclGroupStart/End` 多 op 串行化;`NCCL_BUFFSIZE` 默认值在大集群上不够用导致带宽降。
- **§8**: NCCL User Guide + NCCL paper + SHARP Whitepaper + nccl-tests github + PyTorch ProcessGroupNCCL 源码。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/15-nccl-collectives.md
git commit -m "docs(cuda-zh): 15 NCCL 深度 — ring/tree 拐点 + SHARP 实测 + straggler debug"
```

---

## Task 20: 扩深 16 CUDA Graphs

**Files:**
- Modify: `docs/cuda-zh/16-cuda-graphs.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: capture mode 三种隔离 — global(进程内任何 stream activity 都被捕获,禁止其他 stream 走非 capture-safe API)、thread(只本线程)、relaxed(允许部分 race);`cudaGraphInstantiateFlagAutoFreeOnLaunch` 用于动态形状 + 自动释放;conditional graph node 12.4+ — 设备侧条件 (`cudaGraphAddNode` + `cudaGraphConditionalHandleCreate`) vs 主机侧条件 (Host node + cb)。新增 Mermaid `flowchart TD` 画 capture → instantiate → launch 流程:stream 命令序列被 capture 为 cudaGraph_t → cudaGraphInstantiate 编译 → cudaGraphLaunch 复用。
- **§3 接口深化**: `cudaStreamBeginCapture(stream, mode)`、`cudaStreamEndCapture(stream, &graph)`、`cudaGraphInstantiateWithFlags(&exec, graph, flags)`、`cudaGraphLaunch(exec, stream)`、`cudaGraphExecUpdate(exec, newGraph, &updateInfo)`(in-place 更新);conditional handle 完整 API 序列。
- **§4 性能数字**: 单次 cudaLaunchKernel ~5 µs / 单次 cudaGraphLaunch ~1 µs;训练 step 100 launch 的 host 开销节省 400 µs/step;capture overhead ~50 µs(一次性);graph instantiate ~200 µs 一次性。
- **§5 代码深化**: stream capture training step 完整模板 + ExecUpdate 处理动态形状;PyTorch `torch.cuda.graph()` context manager 内部如何管理 capture stream + memory pool。
- **§7 反模式深化**: capture 期间调 `cudaMalloc`(应用 mempool);conditional node 在 < 12.4 上调用直接 launch fail;capture stream 与 NCCL collective stream 同步遗漏;graph instantiate 后忘 destroy 导致内存泄漏。
- **§8**: Programming Guide §3.2.8 + Driver API `cuGraph*` + Best Practices §11 + PyTorch CUDA Graph 源码 + CUDA Sample `simpleGraph`。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/16-cuda-graphs.md
git commit -m "docs(cuda-zh): 16 CUDA Graphs 深度 — 三种 capture mode + conditional 12.4+ + ExecUpdate 实战"
```

---

## Task 21: 扩深 17 Persistent + Dynamic Parallelism

**Files:**
- Modify: `docs/cuda-zh/17-persistent-and-dynamic-parallelism.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: persistent kernel 占满 SM 后,priority stream 的饿死现象 — 必须用 `cudaFuncAttributePreferredSharedMemoryCarveout` 控制 occupancy 留资源;dynamic parallelism 2.0 的 launch queue 默认深度 2048(`cudaLimitDevRuntimePendingLaunchCount`),OOM 时 child launch 卡住;cluster 内的 DP launch 受限(target 必须非 cluster);CUTLASS 3.x 的 sm90 persistent GEMM 用 1 grid 长生命周期 + work tile 队列。新增 Mermaid `sequenceDiagram` 画 persistent kernel 工作流(host work push → atomic pop → process → 循环);+ Mermaid `flowchart LR` 画 DP parent → child grid 路径(driver-side launch queue 深度 2048)。
- **§3 接口深化**: persistent 自己写 grid-stride + atomic pop 队列(无专门 API);DP `cudaLaunchKernelEx(&attrs, child_fn, n, ...)` (推荐);`cudaStreamFireAndForget`(不等 child);`cudaDeviceGetLimit/SetLimit(cudaLimitDevRuntimeSyncDepth, ...)` 控制 nested DP 层数(默认 24);`cudaDeviceSynchronize` 在 device 端 deprecated(改用 cooperative_groups + cudaStreamFireAndForget)。
- **§4 性能数字**: persistent grid 占满 SM 后 launch 摊销 ~0;DP child launch ~10 µs(host launch 5 µs);DP 深度 24 + 每层 2048 child = 队列爆炸;CUTLASS 3.x persistent GEMM 实测 90%+ TC peak。
- **§5 代码深化**: persistent server 完整 PTX 风格代码(work queue + while loop + atomic pop);CUTLASS 3.x `gemm_sm90_kernel.hpp` 风格 persistent template skeleton;`cudaLaunchKernelEx` 完整 attrs(cluster dim + cooperative + ...)。
- **§7 反模式深化**: persistent 占满 SM 后 priority stream 饿死;DP 深度递归触发 24 层限制(可调但增加 stack);`__syncthreads` 在 child kernel 内不同步 parent;child launch queue OOM 时 silent block。
- **§8**: Programming Guide §6.5 + §3.2.8.7.10 + CUTLASS 3.x sm90 persistent kernel + ThunderKittens primitives + cdpSimpleQuicksort sample。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/17-persistent-and-dynamic-parallelism.md
git commit -m "docs(cuda-zh): 17 Persistent+DP 深度 — SM 饿死 + launch queue 深度 + CUTLASS 3.x persistent"
```

---

## Task 22: DG4 验证 + tag

```bash
for f in docs/cuda-zh/1[2-7]-*.md; do
    sec=$(grep -c '^## [1-8]\. ' "$f"); mer=$(grep -c '^```mermaid' "$f")
    zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text('utf-8')
t = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', t, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', t)))
" "$f")
    grep -qi 'gpusim' "$f" && g="FOUND" || g="ok"
    printf "%s: sec=%d mer=%d zh=%d %s\n" "$f" "$sec" "$mer" "$zh" "$g"
done

git tag cuda-zh-deep-DG4-complete
```

---

# DG5 — 内存管理 + Driver API + 工具链 + 编译(18-22)

## Task 23: 扩深 18 Stream-ordered Allocator

**Files:**
- Modify: `docs/cuda-zh/18-stream-ordered-allocator.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: PyTorch CUDACachingAllocator 在 `cudaMallocAsync` 之上的 buddy + size-segregated free-list 实现;`expandable_segments`(CUDA 11.4+)避免 OOM;跨 stream 引用计数 race(同一块被两 stream 引用,free 触发条件);`PYTORCH_CUDA_ALLOC_CONF` 环境变量解析(`max_split_size_mb`、`garbage_collection_threshold`、`expandable_segments`)。新增 Mermaid `stateDiagram-v2` 画 PyTorch allocator block 状态:Allocated → ToBeFreed (refcount=0) → FreedOnStream → AfterEvent → ReturnedToPool → ReturnedToOS。
- **§3 接口深化**: `cudaMemPoolCreate(&pool, &props)` props 字段(allocType / handleTypes / location);`cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &v)`;`cudaMemPoolGetAttribute(...cudaMemPoolAttrUsedMemCurrent...)`;`cudaMallocFromPoolAsync(&p, n, pool, stream)`;`cudaMemPoolExportToShareableHandle`(IPC pool)。
- **§4 性能数字**: 同 stream alloc-free-alloc 立即重用 ~0 µs;首次 alloc 走 cudaMallocAsync 慢路径 ~50 µs;trim_to release_threshold 调小让显存紧张但抖动;`max_split_size_mb=128` 实测降碎片。
- **§5 代码深化**: PyTorch 训练 step 用 `cudaMallocAsync` 自动重用 activation 的实测对比;CUDACachingAllocator stat dump (`torch.cuda.memory._snapshot`) 解读;手写 IPC mempool 跨进程共享。
- **§7 反模式深化**: alloc 在 streamA / free 在 streamB 但没 event sync(race);release_threshold 0 频繁 trim(抖动);`expandable_segments` 与传统 split 模式混用导致碎片。
- **§8**: Programming Guide §3.2.5.5 + Driver API `cuMemAlloc*` + Best Practices §15.3 + PyTorch CUDACachingAllocator 源码。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/18-stream-ordered-allocator.md
git commit -m "docs(cuda-zh): 18 Pool 深度 — PyTorch CachingAllocator 内部 + expandable_segments + IPC"
```

---

## Task 24: 扩深 19 Unified Memory

**Files:**
- Modify: `docs/cuda-zh/19-unified-memory.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: GPU MMU 处理 page fault 的硬件路径;HMM(Heterogeneous Memory Management)在 GH200 上的 zero-copy 行为(GPU 直接走 CPU 页表);ATS(Address Translation Service)绕过 IOMMU 直接读 system memory;`cudaMemAdviseSetReadMostly` 触发 page 复制(每访问 GPU 都有副本);page granularity Hopper 上 64 KB(可调)。新增 Mermaid `sequenceDiagram` 画 page fault 完整流程:GPU LD/ST → MMU 报 fault → driver 拿到 fault → 决定迁移方向 → 把 page 从 CPU 迁到 GPU(或反向) → 更新 GPU 页表 → 重新发起 LD/ST。
- **§3 接口深化**: `cudaMallocManaged(&p, n, flag)` flag (cudaMemAttachGlobal / cudaMemAttachHost);`cudaMemPrefetchAsync(p, n, dev, stream)` 的 device 参数 (`cudaCpuDeviceId` 拉回 CPU);`cudaMemAdvise(p, n, cudaMemAdviseSetPreferredLocation, dev)`;`cudaStreamAttachMemAsync(stream, p, n, cudaMemAttachSingle)` per-stream 可见性;`cudaMemRangeGetAttribute` 查 advice 状态。
- **§4 性能数字**: page fault ~50 µs / 4KB page;HMM 在 GH200 上免迁移直接读 100-200 ns(经 NVLink-C2C);`SetReadMostly` + 多 GPU 并发读 同 page 提升 10×;page-fault 风暴(乒乓迁移)让 kernel 慢 100×。
- **§5 代码深化**: scientific simulation case — initial cudaMallocManaged + cudaMemPrefetchAsync 提前迁移消除 fault,实测 5× 加速;`SetReadMostly` lookup table 多 GPU 并发的代码;HMM 在 GH200 上 stride access CPU 数据无 fault 的实战。
- **§7 反模式深化**: 频繁 CPU/GPU 交替写同 page(乒乓迁移);忘 prefetch 让首 launch 慢百倍;`SetReadMostly` 然后写(强制取消 advice + 失效所有副本);跨 device read-mostly 但忘 SetAccessedBy(每次 fault)。
- **§8**: Programming Guide §3.2.4 + §K.2 + Best Practices §9.2.2.4 + GH200 Whitepaper § HMM + ATS 文档。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/19-unified-memory.md
git commit -m "docs(cuda-zh): 19 UM 深度 — page fault 硬件路径 + HMM/ATS in GH200 + 乒乓 case"
```

---

## Task 25: 扩深 20 CUDA Driver API

**Files:**
- Modify: `docs/cuda-zh/20-cuda-driver-api.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: primary context 的多线程引用计数(`cuDevicePrimaryCtxRetain` / `Release`);explicit context 的 push/pop 栈(thread-local);`cuModuleLoadDataEx` JIT 编译缓存默认在 `~/.nv/ComputeCache/`(`CUDA_CACHE_PATH` 可改);Lazy context init 在 cuda runtime 隐式触发的延迟;TensorRT / Triton 直接调 driver API 避免 lazy init。新增 Mermaid `classDiagram` 画 Runtime API ↔ Driver API ↔ kernel-mode driver 的层级 + dual-API 调用映射(`cudaLaunchKernel` → `cuLaunchKernel`)。
- **§3 接口深化**: `cuInit(0)`(必须最早);`cuDeviceGet(&dev, 0)`;`cuDevicePrimaryCtxRetain(&ctx, dev)` + `cuCtxSetCurrent(ctx)`(推荐路径,避免 push/pop 复杂度);`cuModuleLoadData(&mod, ptx_str)` JIT;`cuModuleGetFunction(&fn, mod, "kernel_name")`;`cuLaunchKernel(fn, gx,gy,gz, bx,by,bz, smem, stream, params, extra)` 的 extra 参数固定 `CU_LAUNCH_PARAM_BUFFER_POINTER` + `CU_LAUNCH_PARAM_BUFFER_SIZE` + `CU_LAUNCH_PARAM_END`。
- **§4 性能数字**: driver API 调用 +1-2 µs vs runtime;primary context init ~100 ms 一次性;JIT PTX→SASS ~100-300 ms / kernel(可缓存);TensorRT 启动调 driver API 节省 lazy init 几百 ms。
- **§5 代码深化**: 最小 driver-only kernel launch 完整代码(cuInit→cuDeviceGet→PrimaryCtxRetain→ModuleLoad→GetFunction→LaunchKernel);Triton 编译产物如何通过 `cuModuleLoadData` 加载;PyTorch eager 用 driver API(`torch._C._cuda_initEnable*`)的位置。
- **§7 反模式深化**: forgot `cuInit(0)`(后续全 fail);Runtime + Driver 共享 context 但 retain/release 不平衡(销毁后 runtime 还在用);`cuLaunchKernel` extra 参数误传 NULL 之外的东西;多 device 切换没 push/pop 导致跨 context 操作。
- **§8**: CUDA Driver API Reference + Programming Guide §3.4 + CUDA Sample `vectorAddDrv` + Triton driver loader 源码。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/20-cuda-driver-api.md
git commit -m "docs(cuda-zh): 20 Driver API 深度 — primary context refcount + JIT cache + Triton 实战"
```

---

## Task 26: 扩深 21 Profiling 工具栈

**Files:**
- Modify: `docs/cuda-zh/21-profiling-toolchain.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: NSight Compute 的 kernel replay 语义 — 每个 metric 重跑同一 kernel 一次,改变 cache 状态(所以 sustained 数字 ≠ 单次实测);Compute 的 `--target-processes all` 多进程 attach;CUPTI Activity API 的 ring buffer + flush 时机;NVTX 3 在 nsys 自动捕获并显示在时间线;PyTorch profiler 的 `with_stack` Python 帧采集 overhead。新增 Mermaid `flowchart TD` 画 profile 工具栈层级:用户代码 + NVTX → CUDA Runtime → Driver → GPU;CUPTI 钩子从 driver 收 activity → 第三方 profiler / Compute / Systems 都基于 CUPTI。
- **§3 接口深化**: `nvtxRangePushA("name") / nvtxRangePop`、`nvtxMarkA`、`nvtx3::scoped_range r("name")` C++ RAII;CUPTI `cuptiActivityRegisterCallbacks` 完整 setup(callback / activity API 区别);`cudaProfilerStart` / `cudaProfilerStop` 控制 profile 范围(配合 nsys `-c cudaProfilerApi`);PyTorch profiler `record_function`、export chrome trace。
- **§4 性能数字**: nsys overhead < 5%;Compute kernel replay 让 profile 慢 10-100×(每 metric set 重跑);NVTX push/pop ~50 ns / call;PyTorch profiler `with_stack=True` overhead 30-100%(适合 dev,不适合 prod)。
- **§5 代码深化**: nsys + Compute + NVTX 的完整组合工作流(nsys 找瓶颈 kernel → cudaProfilerStart 圈定 → ncu 单测该 kernel);PyTorch profiler 输出 chrome trace 的实战 + 关键事件解读。
- **§7 反模式深化**: 在生产环境留 Compute(replay 改语义);用 nsys 分析单 kernel(应 Compute);忘 `cudaProfilerStart/Stop` 全程 profile 数据爆炸;PyTorch profiler `with_stack=True` 留生产(性能损失)。
- **§8**: NSight Systems / Compute 用户指南 + NVTX 3 github + CUPTI Reference + PyTorch profiler 文档。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/21-profiling-toolchain.md
git commit -m "docs(cuda-zh): 21 Profiling 深度 — kernel replay 语义 + CUPTI Activity + 工具协同"
```

---

## Task 27: 扩深 22 PTX → SASS 编译链

**Files:**
- Modify: `docs/cuda-zh/22-ptx-to-sass.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2)**

要点:
- **§2 微架构深化**: ptxas register allocator 的 SSA-based 算法 + spill-to-local-memory 决策;`-O3` 关键 pass(constant prop / loop unroll / instruction scheduling / lifetime analysis);`-arch=sm_90` vs `sm_90a`(后者启用 wgmma/TMA/cluster TMA store 等 sm_90a-only feature)的 PTX 指令集差异;fatbin 多 arch 打包结构;driver 接 PTX 时 JIT 触发条件(无匹配 SASS)。新增 Mermaid `flowchart LR` 画 nvcc 完整 pipeline:.cu → cudafe (host/device 拆) → cicc (cu→PTX) → ptxas (PTX→SASS) → fatbinary → 链接 host obj → executable。
- **§3 接口深化**: nvcc flag 矩阵 — `-arch=compute_90/sm_90/sm_90a`、`-gencode arch=compute_90,code=sm_90,sm_90a`、`-Xptxas -O3 / -v / --maxrregcount=N`、`-Xcicc -O3`、`--use_fast_math`、`--device-debug` 与 sass 调试关系;cuobjdump / nvdisasm 完整 flag(`--dump-sass --function 'kernel*'`);`cuModuleLoadDataEx` 的 JIT options。
- **§4 性能数字**: ptxas O3 vs O0 性能差 5-10×;spill-to-local 一次 access ~100 cycle (L1 命中);JIT PTX→SASS 一次 ~100-300 ms / kernel(缓存后跳过);`--maxrregcount=32` 强压寄存器导致大量 spill 反而慢 30%。
- **§5 代码深化**: 编译 + 反汇编完整命令链(`nvcc -arch=sm_90a -ptx kernel.cu -o k.ptx && ptxas -arch=sm_90a -O3 k.ptx -o k.cubin && cuobjdump --dump-sass k.cubin | head -40`);用 inline PTX `asm volatile("mma.sync...")` 嵌入手写 PTX 的 case;`__nv_isglobal()` 等编译期 builtin。
- **§7 反模式深化**: `-arch=sm_90` 写 wgmma kernel(应 sm_90a);关掉 -O3 跑 benchmark(数字假);忘 `--maxrregcount` 反推 occupancy;JIT cache `~/.nv/ComputeCache` 不清导致老 SASS 持久化。
- **§8**: PTX ISA Reference + CUDA Compiler Driver NVCC + Inline PTX in CUDA C++ + ptxas option 完整列表。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/22-ptx-to-sass.md
git commit -m "docs(cuda-zh): 22 PTX→SASS 深度 — ptxas register allocator + sm_90 vs sm_90a + JIT cache"
```

---

## Task 28: DG5 验证 + tag

```bash
for f in docs/cuda-zh/1[8-9]-*.md docs/cuda-zh/2[0-2]-*.md; do
    sec=$(grep -c '^## [1-8]\. ' "$f"); mer=$(grep -c '^```mermaid' "$f")
    zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text('utf-8')
t = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', t, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', t)))
" "$f")
    grep -qi 'gpusim' "$f" && g="FOUND" || g="ok"
    printf "%s: sec=%d mer=%d zh=%d %s\n" "$f" "$sec" "$mer" "$zh" "$g"
done

git tag cuda-zh-deep-DG5-complete
```

---

# DG6 — Capstone(23 训练、24 推理) + 全集验证 + push

## Task 29: 扩深 23 模型训练全栈串联

**Files:**
- Modify: `docs/cuda-zh/23-training-end-to-end.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2,§7 保持"训练侧优化方法体系"标题)**

要点:
- **§2 微架构深化**: 加 70B 参数 LLaMA-style transformer 单 step 的组件触发详细分解(每层 attention + FFN 的 GEMM size + KV / activation 内存占用);FSDP + Megatron-LM TP=8 PP=4 配置下 NCCL allreduce / reduce-scatter 的发生节点。新增 Mermaid `sequenceDiagram` 画 FSDP 一次 forward + backward + optimizer:per-layer 的 all-gather params → forward → reduce-scatter grads(与下层 backward 重叠)→ optimizer 更新 sharded states。
- **§3 接口深化**: 完整 FSDP 配置 — `MixedPrecision(param=bf16, reduce=bf16, buffer=bf16)`、`ShardingStrategy.FULL_SHARD`、`backward_prefetch=BACKWARD_PRE`、`auto_wrap_policy=transformer_auto_wrap_policy`;Megatron-LM `--tensor-model-parallel-size 8 --pipeline-model-parallel-size 4`;DeepSpeed ZeRO-3 + offload config json;Transformer Engine `fp8_autocast(enabled=True, fp8_recipe=DelayedScaling)`。
- **§4 性能数字**: 70B Llama bf16 训练 H100 SXM5×64 实测 MFU ~50%;405B Llama bf16 训练 H100×4096 实测 MFU ~40%;FP8 training 训练 throughput vs bf16 1.7×(论文实测);grad bucket 调到 25 MB 让 NCCL 最佳;activation checkpointing 让 80B 模型可以单 H100 跑(原本 OOM)。
- **§5 代码深化**: 标准 FSDP 训练 step 完整代码(autocast + backward + scaler) + cudaGraph capture 的 boundary 注意点(NCCL collective 的 capture mode 要求);Megatron 1F1B pipeline schedule 的 micro-batch 注入伪代码;FP8 Transformer Engine 的 weight + grad 配置代码。
- **§7 训练侧优化方法体系深化(原优化清单基础上每条加 production 数据 + 何时不要用)**:
  - FlashAttention-3:实测让 attention forward 5×、backward 7×;但 short seq (< 1024) overhead 不值得
  - FP8 training:H100 TC FP8 1979 TFLOPS,实测训练加速 1.7-2.0×;但需要 Transformer Engine + DelayedScaling 防 underflow
  - Activation checkpointing:80B Llama 单 H100 80 GB 必备;计算开销 ~30%(forward 重算);selective recompute 只重算 attention 把开销降到 5%
  - Gradient accumulation:micro_batch=1 + accum=64 等价 batch=64,显存省;但 step time 翻倍
  - ZeRO-3 / FSDP full shard:N GPU 显存 1/N;但通信 3×(forward all-gather + backward all-gather + reduce-scatter)
  - TP / PP / SP 三维并行:TP 在 NVLink 内(同节点 8 GPU),PP 跨节点,SP 沿 seq(把 layernorm + attention 沿 seq 切);3D 并行最佳实测在 405B 上需要全部
  - Compute-comm 重叠:FSDP backward_prefetch=BACKWARD_PRE 让 reduce-scatter 与下一层 backward 重叠,NCCL bus bandwidth saturate
  - CUDA Graph training step capture:节省 ~400 µs / step(100 launch × 4 µs);PyTorch `torch.cuda.graph()` 必须 mempool 配合;NCCL collective capture 需 `record_stream` 处理
- **§8**: 加 PyTorch FSDP paper、Megatron-LM paper、DeepSpeed paper、FlashAttention-3 paper、Transformer Engine 文档、Llama 3 405B training report。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/23-training-end-to-end.md
git commit -m "docs(cuda-zh): 23 训练全栈深度 — FSDP/Megatron 实战 + 70B/405B MFU 实测 + FP8 training"
```

---

## Task 30: 扩深 24 模型推理全栈串联

**Files:**
- Modify: `docs/cuda-zh/24-inference-end-to-end.md`

- [ ] **Step 1: Read + 加深(目标 4000-5000 字 + Mermaid ≥ 2,§7 保持"推理侧优化方法体系"标题)**

要点:
- **§2 微架构深化**: prefill 与 decode 不同的硬件压力 — prefill bound by TC(GEMM peak,实测 80% TC peak),decode bound by HBM bandwidth(实测 60-80% HBM peak,GEMV memory-bound);PagedAttention block table 在 GPU 端用 GMEM 索引,每 attention head 一个 block table;Sarathi-Serve 的 chunked prefill 把 prefill 分小块插入 decode 流。新增 Mermaid `sequenceDiagram` 画 continuous batching 一个调度周期:多请求异步加入 → scheduler 选 batch → 拼 prefill chunks + decode tokens → 一次 forward → sample → 完成请求 release 资源。
- **§3 接口深化**: vLLM `LLM(model="meta-llama/Llama-3-70B", tensor_parallel_size=8, gpu_memory_utilization=0.95, enable_prefix_caching=True)`、`SamplingParams(temperature, top_p, max_tokens)`;TensorRT-LLM 完整 build pipeline (`trtllm-build --checkpoint_dir ... --output_dir ... --gemm_plugin float16`);SGLang frontend DSL + RadixAttention;FlashInfer 库的 paged attention kernel 直调。
- **§4 性能数字**: vLLM Llama-3-70B INT8 KV H100×8 实测 1500 tokens/sec/GPU(prefill 占 30%);PagedAttention 让 batch size 比 vanilla attention 大 4×;FP8 inference H100 vs bf16 实测 decode 1.8×;speculative decoding(EAGLE-2)实测加速 3-4×;Llama-3-405B INT4 weight-only TensorRT-LLM 单 H100×8 ~50 tokens/sec/req。
- **§5 代码深化**: vLLM 完整 serve 启动命令 + python 客户端;PagedAttention block table 的 attention kernel 调用代码片段(FlashInfer 风格);CUDA Graph decode capture 在 vLLM 中的实现位置(`vllm/worker/model_runner.py`)。
- **§7 推理侧优化方法体系深化**:
  - PagedAttention(vLLM):block table + free pool,batch size 提升 2-4×;实测 Llama-2-70B serving throughput 提升 24×(vs HF baseline)
  - FlashAttention-3:Hopper warp-specialization,memory-bound attention kernel 提升 2×;但 short seq 不显著
  - Speculative decoding(EAGLE-2 / Medusa):draft 4 token + target verify,实测 3-4× decode throughput;但 batch size 大时增益降
  - Continuous batching(vLLM iteration-level):请求异步加入,GPU utilization 60% → 90%
  - Sarathi-Serve chunked prefill:把 prefill 分小块插入 decode 流,end-to-end p99 latency 降 50%
  - Disaggregated serving(DistServe):prefill GPU(compute-optimized)+ decode GPU(memory-optimized)分离,SLA 达成提升 4×
  - INT4 weight-only(GPTQ / AWQ):decode HBM read 减半,decode throughput 1.8×;精度损失 1-3% MMLU
  - SmoothQuant(W8A8):INT8 weight + INT8 activation,prefill TC 加速 2×;但需要校准 scale
  - FP8 inference(Hopper TC):2× decode throughput vs bf16;NVIDIA TRT-LLM 默认推荐
  - KV cache 量化(INT8/FP8 KV):KV cache 显存减半,batch 翻倍;精度损失轻微
  - Multi-LoRA serving:base 共享 + 多 LoRA adapter 切换,单 GPU 服务多个 finetune 模型
  - Tensor parallel inference:大模型横向切跨 GPU,NCCL allreduce after attention/FFN
  - Prefix caching(RadixAttention):多用户共 system prompt 时 prefill 几乎免费,实测 chat 场景 2× throughput
- **§8**: 加 vLLM paper、FlashAttention-3 paper、GPTQ/AWQ paper、SmoothQuant paper、DistServe paper、Sarathi-Serve paper、SGLang paper、EAGLE-2 paper、TensorRT-LLM github。

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/24-inference-end-to-end.md
git commit -m "docs(cuda-zh): 24 推理全栈深度 — vLLM/TRT-LLM 实战 + Llama-70B/405B 实测 + 12+ 优化方法"
```

---

## Task 31: 全集验证 + tag + push

- [ ] **Step 1: 全集验证脚本**

```bash
cd docs/cuda-zh
ls -1 *.md | wc -l                              # expect 25
echo ""

for f in $(ls -1 *.md | sort); do
    sec=$(grep -c "^## [1-8]\. " "$f")
    mer=$(grep -c '^```mermaid' "$f")
    zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text('utf-8')
t = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', t, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', t)))
" "$f")
    grep -qi 'gpusim' "$f" && g="GPUSIM_FOUND" || g="ok"
    expect_mer=2; [[ "$f" == 00-* ]] && expect_mer=3
    if (( zh < 4000 || zh > 5000 )); then zh_status="OUT_OF_RANGE"; else zh_status="ok"; fi
    if (( mer < expect_mer )); then mer_status="LOW"; else mer_status="ok"; fi
    printf "%-50s sec=%d mer=%d(%s) zh=%d(%s) %s\n" "$f" "$sec" "$mer" "$mer_status" "$zh" "$zh_status" "$g"
done

echo ""
echo "=== 总 mermaid(应 ≥ 51)==="
grep -c '^```mermaid' *.md | awk -F: '{s+=$2} END {print s}'
```

预期:25 文件;每章 sec=8、mer 在范围内、zh ∈ [4000,5000]、gpusim=ok;总 mermaid ≥ 51。

如果某章不达标,先回到对应任务修复(扩 §2/§5/§7 或裁剪)。

- [ ] **Step 2: Tag DG6 + 总 ship tag**

```bash
git tag cuda-zh-deep-DG6-complete
git tag cuda-zh-deep-complete
```

- [ ] **Step 3: Push 到 GitHub**

```bash
git push origin master
git push origin cuda-zh-deep-DG1-complete cuda-zh-deep-DG2-complete cuda-zh-deep-DG3-complete cuda-zh-deep-DG4-complete cuda-zh-deep-DG5-complete cuda-zh-deep-DG6-complete cuda-zh-deep-complete
```

预期输出:6 个 DG tag + 1 个 ship tag,master 推送成功。

---

## 验收准则

教程深度扩展完成的标准:

- [ ] 25 个 markdown 字数全部 [4000, 5000]
- [ ] 每章 Mermaid ≥ 2(00 章 ≥ 3),全集总数 ≥ 51
- [ ] 全部章节零 gpusim 引用
- [ ] 6 个 DG milestone tag + 1 个 ship tag 全到位:`cuda-zh-deep-DG1-complete` ... `cuda-zh-deep-DG6-complete` + `cuda-zh-deep-complete`
- [ ] master + 全部新 tag 推送到 origin
- [ ] 每章包含五类深度内容:微架构机制 / 真实生产数字 / 失败模式 / 实现导读 / 设计权衡
