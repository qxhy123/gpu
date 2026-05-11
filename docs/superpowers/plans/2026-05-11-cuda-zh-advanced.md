# cuda-zh Advanced — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 cuda-zh 教程加 10 章 advanced 进阶补充集(`docs/cuda-zh/advanced/`),每章 4000-5000 中文字 + Mermaid ≥ 2,覆盖 senior AI Infra 真实卡壳点。

**Architecture:** 10 章按主题分 4 个 milestone(AG1 LLM kernel 系列 / AG2 基础设施 + 框架 / AG3 下一代 + 部署 / AG4 索引 + push)。每个 AG 单 subagent 一次 dispatch 内部串行写多章。沿用 cuda-zh 既定规范(8 节结构、Mermaid 强制、零 gpusim、UTF-8 LF)。最后 push 到 GitHub origin。

**Tech Stack:** Markdown,Mermaid。无代码,无测试 — 验证靠 grep + 字数。

---

## 全局规则(适用于所有任务)

### 目录与文件
- 新章节全部入 `docs/cuda-zh/advanced/`(第一个任务里 `mkdir -p docs/cuda-zh/advanced`)
- 文件命名:`aNN-<slug>.md`(a01 ~ a10)
- 修改文件:`docs/cuda-zh/00-index.md`(最后任务追加 advanced 表格)

### 每章标准结构(强制)

```markdown
# aNN · <中文标题>

> **一句话总结**

## 1. 是什么 / 为什么有它
## 2. 硬件 / 系统视角(微架构 / 拓扑 / 协议)
## 3. CUDA / 框架编程接口
## 4. 关键性能指标
## 5. 代码示例
## 6. 实测手段
## 7. 常见反模式
## 8. 延伸阅读
```

### 字数 + Mermaid 配额
- 每章 **4000-5000 中文字**(代码块 + Mermaid 不计)
- 每章 Mermaid ≥ 2(架构 / 流程类章节 a01 MoE、a02 CUTLASS、a05 RDMA、a09 Blackwell 建议 ≥ 3)
- 全集 Mermaid 总数应从 51 升至 **≥ 73**

### 五类必加内容(每章)
1. **微架构 / 协议机制**
2. **真实生产数字**(H100 SXM5 / B200 / 大规模训练实测,不是 spec 上限)
3. **失败模式 + 调试**(production 真踩过的坑 + 检测手段)
4. **实现导读**(vLLM、CUTLASS、DeepEP、Triton、Megatron、NCCL 等真实源码位置)
5. **设计权衡**(为什么选 A 不选 B)

### 内容质量约束
- 零 gpusim 引用(`grep -i gpusim` 空命中)
- 真实数字必须可追溯:Hopper/Blackwell Whitepaper、论文、官方文档,引用注明
- 真实代码:`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`、`@triton.autotune(configs=[...], key=['M','N','K'])`、`cublasLtMatmulAlgoGetHeuristic` 等真名
- 无营销语,无友商对比(无 AMD / Intel)
- UTF-8 LF;`#` 仅章名;`##` 仅 §1-§8

### 验证脚本(每章写完跑)
```bash
F=docs/cuda-zh/advanced/aNN-xxx.md
sec=$(grep -c '^## [1-8]\. ' $F)
mer=$(grep -c '^```mermaid' $F)
zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text('utf-8')
t = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', t, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', t)))
" $F)
grep -qi 'gpusim' $F && g="FOUND" || g="ok"
echo "$F: sec=$sec mer=$mer zh=$zh gpusim=$g"
# 必须: sec=8, mer ≥ 2 (a01/a02/a05/a09 建议 ≥ 3), zh ∈ [4000,5000], g=ok
```

字数若 < 4000 或 > 5000,优先调 §2(微架构)+ §5(代码示例后说明)+ §7(优化分类)直到达标。

### CRITICAL — 环境
所有 `python` 调用必须用 `.venv/bin/python`。Conda python 缺 `ml_dtypes`,会触发误报。

---

# AG1 — LLM kernel 系列(a01-a04)

## Task 1: a01 MoE + Expert Parallelism

**Files:**
- Create: `docs/cuda-zh/advanced/a01-moe-expert-parallelism.md`

- [ ] **Step 1: 准备目录 + 写章节(目标 4000-5000 字 + Mermaid ≥ 3)**

`mkdir -p docs/cuda-zh/advanced`

按下面要点写,严格 8 节结构:

**章名:** `# a01 · MoE 模型与 Expert Parallelism — 训练 + 推理 + 通信瓶颈`

**§1 是什么 / 为什么有它(200-300 字):**
- MoE = Mixture of Experts。FFN 层切成 N 个并行 expert(8 / 64 / 256 个),每 token 通过 router 选 top-k(通常 k=2)发到对应 expert,激活参数远小于总参数
- 代表模型:Mixtral 8×7B(56B 总,12B 激活)、DeepSeek-V3 671B(37B 激活)、GShard、Switch Transformer
- 为什么 senior 必须懂:Frontier lab 都在做、训练通信瓶颈完全不同(all-to-all 代替 allreduce)、推理稀疏激活路径

**§2 系统视角 + Mermaid(400-500 字 + Mermaid ≥ 2):**
- **Mermaid `flowchart TB`** 画 MoE block:输入 token → router/gate(小 linear)→ softmax top-k → permute(token → expert)→ N 个 expert FFN 并行 → unpermute(expert → token)→ weighted sum 输出
- **Mermaid `flowchart LR`** 画 Expert Parallelism 通信:8 个 GPU,每 GPU 持 1 个 expert,token routing 触发 all-to-all(每 GPU send 自己 token 到目标 expert + recv 别的 GPU 发来的 token)
- 关键概念:
  - capacity factor:`tokens_per_expert = (total_tokens / num_experts) × capacity_factor`(1.0-1.25)
  - load balancing aux loss:`L_aux = N × Σ_i (f_i × P_i)`,鼓励均衡
  - EP(expert parallelism)与 TP / DP / PP 正交
  - all-to-all 是 MoE 主导通信(allreduce 在 attention,all-to-all 在 FFN)

**§3 框架接口(300-400 字):**
- Megatron-LM: `--num-experts 64 --moe-expert-model-parallel-size 8 --moe-router-topk 2 --moe-aux-loss-coeff 0.01`
- DeepSpeed-MoE: `deepspeed.moe.layer.MoE(hidden_size, expert, num_experts, k=2, capacity_factor=1.25)`
- DeepEP(DeepSeek 开源 EP all-to-all 库):`deep_ep.Buffer.dispatch / combine`
- Megablocks(MIT):scatter-gather kernel 替代标准 all-to-all,在 sparse 模式更快
- Tutel(微软):adaptive routing + 动态 capacity

**§4 关键性能指标 + 反模式提示(400-500 字):**
- 实测数字:
  - DeepSeek-V3 671B 在 H100×2048 训练 MFU ~50%(FP8 加 TE)
  - Mixtral 8×7B 推理:vLLM + MoE kernel 单 H100 ~80 tokens/sec/req
  - All-to-all 占 MoE step time **30-40%**(vs allreduce dense 模型 10-15%)
  - capacity factor 1.25 时 token drop rate < 1%
  - DeepEP vs 标准 NCCL all-to-all:H800 集群上 small message 提升 3×,large message 提升 1.3×
- 反模式提示:capacity 设太小 token drop / 太大显存爆;忘记 aux loss 让 routing 退化成集中到少数 expert;EP 数与 expert 数不能整除;all-to-all 没 overlap compute

**§5 代码示例(代码块,不计字数):**
- Megatron MoE config + DeepEP dispatch/combine 调用
- router top-k + aux loss 代码

**§6 实测手段(200-300 字):**
- NSight Systems 看 all-to-all 与 expert compute 重叠率
- Megatron 内置 `mlflow` 记录 expert load distribution(检查是否倾斜)
- `nvidia-smi nvlink -gt c` 监控 NVLink 流量(MoE 训练通常 saturate)

**§7 常见反模式(400-500 字):**
1. capacity factor 1.0 严格不丢 token(实际推理就是不允许溢出)→ 训练倾向 1.25 容错
2. aux loss 权重设 0(集中度高、router 训不动)
3. EP 数与 expert 数不能整除导致 padding 浪费 25%
4. 用标准 NCCL all-to-all 不上 DeepEP(latency 高 3×)
5. MoE 训练用 ZeRO-3(参数已经 EP 切了,ZeRO-3 重复切再 over-communicate)
6. capacity 用 token-level 而非 expert-level(每 GPU 看到不均衡)
7. expert 数太多(routing overhead 占比上升)

**§8 延伸阅读(代码块,不计字数):**
- Mixtral paper(`arxiv.org/abs/2401.04088`)
- DeepSeek-V3 paper(`arxiv.org/abs/2412.19437`)
- Switch Transformer paper(`arxiv.org/abs/2101.03961`)
- GShard paper(`arxiv.org/abs/2006.16668`)
- DeepEP(`github.com/deepseek-ai/DeepEP`)
- Megablocks(`github.com/databricks/megablocks`)
- Tutel(`github.com/microsoft/tutel`)

- [ ] **Step 2: 验证 + Commit**

```bash
F=docs/cuda-zh/advanced/a01-moe-expert-parallelism.md
sec=$(grep -c '^## [1-8]\. ' $F); mer=$(grep -c '^```mermaid' $F)
zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text('utf-8')
t = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', t, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', t)))
" $F)
grep -qi 'gpusim' $F && g="FOUND" || g="ok"
echo "$F: sec=$sec mer=$mer zh=$zh gpusim=$g"
# 必须 sec=8 mer≥2 zh in [4000,5000] g=ok

git add docs/cuda-zh/advanced/a01-moe-expert-parallelism.md
git commit -m "docs(cuda-zh/advanced): a01 MoE + Expert Parallelism — 训推 + DeepEP/Megablocks"
```

---

## Task 2: a02 CUTLASS 3.x + CuTe

**Files:**
- Create: `docs/cuda-zh/advanced/a02-cutlass-3x-and-cute.md`

- [ ] **Step 1: 写章节(目标 4000-5000 字 + Mermaid ≥ 3)**

**章名:** `# a02 · CUTLASS 3.x + CuTe — Hopper kernel 工程的事实标准`

**§1(200-300 字):** CUTLASS 是 NVIDIA 官方 GPU kernel template 库;3.x 为 Hopper SM90 完全重写;CuTe 是底层 Layout / Tensor 抽象代数;FlashAttention-3、Transformer Engine、vLLM PagedAttention 内核都基于 CUTLASS 3.x;senior 写自定义 kernel 不可能绕开。

**§2 + Mermaid ≥ 2(500-600 字):**
- **Mermaid `classDiagram`** 画 CUTLASS 3.x 类层级:`gemm::kernel::Sm90*` → `CollectiveMainloop` + `CollectiveEpilogue`;`CollectiveMainloop` 内部 `TiledMMA` + `SmemLayoutA/B/C` + `TmaCopy`
- **Mermaid `flowchart LR`** CuTe Layout 代数核心操作:`composition / inverse / partition / coalesce / right_inverse`
- 关键抽象:
  - `Layout<Shape, Stride>` 描述 ND tensor 在线性内存的布局
  - `Tensor<Engine, Layout>` 把数据指针 + Layout 绑定
  - `TiledMMA` 描述 wgmma fragment 的 thread × elem 映射
  - `partition_S(thr_mma, gA)` 把全局 tensor 按线程切分

**§3(300-400 字):**
- CuTe: `Layout<Shape<_64,_128,_16>, Stride<_128,_1,_8192>>`、`make_tensor / make_layout`
- `SM90_TMA_LOAD` / `SM90_TMA_STORE` copy op
- `KernelTraits` template 定制 mainloop
- CUTLASS Python interface(2.x 有,3.x 也支持)

**§4(400-500 字):**
- CUTLASS 3.x H100 GEMM 实测 87-92% TC peak(bf16);FP8 ~85%
- FlashAttention-3 用 CUTLASS 3.x 重写后比 FA-2 快 2×
- CUTLASS 2.x vs 3.x:Hopper async features(wgmma + TMA + cluster)必须 3.x
- CuTe Layout 编译期常数让 ptxas 完全 unroll(0 运行时开销)
- 反模式提示:CuTe Layout 误用 stride、忘 swizzle 与 wgmma 对齐、用 2.x API 写 3.x

**§5 代码:** 一段 CuTe Layout 实例 + CollectiveMainloop skeleton

**§6 实测:** ncu 看 `sm__pipe_tensor_op_hmma_qmma.sum` 占比

**§7(400-500 字):**
1. CuTe Layout stride 写反(row-major / col-major 算错)
2. forgot swizzle 对齐 wgmma fragment(silent slow,SMEM bank conflict)
3. 用 CUTLASS 2.x API 写在 3.x 编译器报错或退到 mma.sync(失去 wgmma)
4. CollectiveMainloop / Epilogue 类型不匹配
5. CuTe `partition_S` 用错 thr_mma(切分模式不对)

**§8:** CUTLASS github(`github.com/NVIDIA/cutlass`,特别 `include/cutlass/gemm/collective/sm90_*.hpp`)、CuTe paper、Hopper GEMM 官方 examples、NVIDIA GTC CUTLASS talk

- [ ] **Step 2: 验证 + Commit**

```bash
F=docs/cuda-zh/advanced/a02-cutlass-3x-and-cute.md
# ... 同 Task 1 验证脚本 ...
git add docs/cuda-zh/advanced/a02-cutlass-3x-and-cute.md
git commit -m "docs(cuda-zh/advanced): a02 CUTLASS 3.x + CuTe — collective mainloop + Layout 代数"
```

---

## Task 3: a03 量化算法原理

**Files:**
- Create: `docs/cuda-zh/advanced/a03-quantization-algorithms.md`

- [ ] **Step 1: 写章节(目标 4000-5000 字 + Mermaid ≥ 2)**

**章名:** `# a03 · 量化算法原理 — GPTQ / AWQ / SmoothQuant / FP8 scaling`

**§1(200-300 字):** 量化让 LLM 推理 / 训练用更少 bit;主体教程 §7 列了名字 + 加速比,这一章讲算法本身 — Hessian-based 最优 rounding、activation outlier 处理、per-channel migration、FP8 scaling 策略。

**§2 + Mermaid ≥ 2(500-600 字):**
- **Mermaid `flowchart LR`** 画三种量化数据流:
  - GPTQ:逐 col 选最优量化点 + 用 Hessian 逆传播误差
  - AWQ:activation-aware,保留 1% salient weight FP16
  - SmoothQuant:per-channel scale 把 activation outlier 迁到 weight
- **Mermaid `stateDiagram-v2`** FP8 delayed scaling 状态机:Forward 用 prev scale → 记 max → Backward 后更新 scale → 下一 iter 用新 scale
- 关键概念:
  - per-tensor / per-channel / per-group scale(精度 vs 开销)
  - symmetric vs asymmetric quantization
  - GPTQ 用 Cholesky 分解 Hessian(避免数值不稳)
  - AWQ 用 activation 幅度选 1% salient weight 保 FP16
  - SmoothQuant 把 `Y = X · W` 等价转 `Y = (X/s) · (s·W)`,X 变小 W 变大

**§3(300-400 字):**
- TensorRT-LLM: `quantization_modes = [W4A16_AWQ, W8A8_SQ, FP8]`、`--use_smooth_quant`、`--enable_kv_cache_fp8_quantization`
- `torch.ao.quantization` PyTorch native
- Transformer Engine: `Float8Tensor`、`DelayedScaling`、`HYBRID` format(forward E4M3 + backward E5M2)
- AutoGPTQ / AutoAWQ python API

**§4(400-500 字):**
- 实测数字:
  - Llama-3-70B GPTQ INT4 MMLU 79.1 vs FP16 79.5(0.4 pt 损失)
  - AWQ INT4 与 GPTQ 相当,但代码更简单(无 Hessian)
  - SmoothQuant W8A8 比 FP8 慢 10-15%(无原生 INT8 TC,需要走 dp4a)
  - FP8 training delayed scaling 与 current scaling 在 overflow 处理差(delayed 略保守)
  - KV cache FP8 量化:精度损失 < 0.5 pt,显存减半 batch 翻倍
- 反模式提示:校准集太小 / per-tensor 在长尾 activation 崩溃 / FP8 忘 DelayedScaling 触发 underflow NaN

**§5 代码:** GPTQ 逐 col 量化伪代码 + TE Float8Tensor 配 DelayedScaling

**§6:** TRT-LLM benchmark + MMLU 验证 + log loss 跟踪

**§7(400-500 字):**
1. 校准集 < 256 样本 outlier 没覆盖(精度爆)
2. per-tensor scale 在长尾 activation(Mixtral 在某些 expert 上活动量级差 100×)
3. FP8 用 E4M3 训练(应 HYBRID:forward E4M3 + backward E5M2)
4. forgot DelayedScaling(loss NaN 在 step 100 突然出现)
5. 量化后没重测下游(MMLU / GSM8K 必跑)
6. KV cache 全 INT4(精度损失大,应 INT8 / FP8 KV)
7. 用 PTQ on 不稳定模型(应先 finetune 再量化或 QAT)

**§8:** GPTQ paper(`arxiv.org/abs/2210.17323`)、AWQ paper(`arxiv.org/abs/2306.00978`)、SmoothQuant paper(`arxiv.org/abs/2211.10438`)、FP8 Training paper(NVIDIA)、TRT-LLM quantization docs

- [ ] **Step 2: 验证 + Commit**

```bash
F=docs/cuda-zh/advanced/a03-quantization-algorithms.md
# ... 验证 ...
git add docs/cuda-zh/advanced/a03-quantization-algorithms.md
git commit -m "docs(cuda-zh/advanced): a03 量化算法原理 — GPTQ/AWQ/SmoothQuant/FP8 scaling"
```

---

## Task 4: a04 Triton 工程化

**Files:**
- Create: `docs/cuda-zh/advanced/a04-triton-kernel-engineering.md`

- [ ] **Step 1: 写章节(目标 4000-5000 字 + Mermaid ≥ 2)**

**章名:** `# a04 · Triton 工程化 — compiler stack + autotune + torch.compile`

**§1(200-300 字):** Triton 是 OpenAI 推出的 Python DSL for GPU kernel;PyTorch 2.x inductor 默认生成 Triton;FlashAttention 早期 / PagedAttention 部分实现都用 Triton;senior 写自定义 kernel 90% 用 Triton(只在压榨最后 5-10% 性能时下沉 CUTLASS / inline PTX)。

**§2 + Mermaid ≥ 2(500-600 字):**
- **Mermaid `flowchart LR`** Triton compiler pipeline:`.py @triton.jit` → AST → ttir(triton dialect IR)→ ttgir(triton GPU IR,layout 决定)→ llir(LLVM IR with PTX intrinsic)→ PTX → SASS
- **Mermaid `classDiagram`** Triton runtime cache + autotune key:`JITCache` keyed by(arg dtypes、const args、autotune key vals)→ ProgramCache → cubin
- 关键概念:
  - `@triton.jit` 标记 device 函数
  - `tl.program_id(axis)` 取 grid id
  - `tl.load(ptr + offsets, mask=mask)` mask 支持避免越界
  - `tl.dot(a, b)` 调用 wgmma / mma.sync
  - autotune key:每个 key 组合编译独立 cubin

**§3(300-400 字):**
- `@triton.jit` + `@triton.autotune(configs=[...], key=['M','N','K'])` + `@triton.heuristics`
- `tl.program_id / tl.load / tl.store / tl.dot / tl.zeros / tl.cumsum / tl.where`
- `triton.language as tl`
- PyTorch 2.x: `torch.compile` 默认走 inductor + Triton backend
- 何时手写:小 op fusion / 自定义 ML 算子;何时下沉 CUTLASS:GEMM 最后 5%;何时下沉 PTX:wgmma 特殊变体

**§4(400-500 字):**
- 实测数字:
  - Triton vs CUTLASS H100 GEMM:Triton 80%、CUTLASS 90%(差 10%)
  - autotune key 选错(忘 dtype)→ 每 dtype 重 tune ~30s
  - `torch.compile` 在 PyTorch 2.4+ 是默认,生成 Triton 几乎覆盖 90% inductor lowering
  - FlashAttention 2 Triton 实现单 H100 70B 推理 attention 用 ~10% 时间
- 反模式提示:autotune key 漏 dtype / Triton 不支持 wgmma 部分变体需 inline asm / SMEM 大小硬编码超 cap silent fail

**§5 代码:** 完整 Triton GEMM kernel(M/N/K autotune)+ PyTorch inductor 生成代码片段

**§6:** `TRITON_DEBUG=1`、`TRITON_PRINT_AUTOTUNING=1`、ncu profile Triton 生成的 cubin

**§7(400-500 字):**
1. autotune key 漏 dtype(每 dtype 重 tune 30s)
2. Triton SMEM 用 `tl.zeros` 超 SMEM cap 时静默 fail
3. forgot `BLOCK_SIZE` 是编译期常数,运行时变会重编译
4. 用 Triton 写 wgmma 但实际编译生成 mma.sync(部分 wgmma 特性 Triton 还不支持)
5. autotune configs 太多(编译时间爆,key=(M,N,K,dtype)× 20 configs = 几分钟启动)
6. `torch.compile` 与 Triton 自定义 op 混用时 dispatch 顺序错

**§8:** Triton github(`github.com/openai/triton`)、Triton paper、FlashAttention-2 Triton 实现、PyTorch inductor docs、torch.compile 调试指南

- [ ] **Step 2: 验证 + Commit**

```bash
F=docs/cuda-zh/advanced/a04-triton-kernel-engineering.md
# ... 验证 ...
git add docs/cuda-zh/advanced/a04-triton-kernel-engineering.md
git commit -m "docs(cuda-zh/advanced): a04 Triton 工程化 — compiler stack + autotune + torch.compile"
```

---

## Task 5: AG1 验证 + tag

- [ ] **Step 1: 批量验证 4 章**

```bash
for f in docs/cuda-zh/advanced/a0[1-4]-*.md; do
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
```

Expected: 4 章全部 sec=8、mer ≥ 2、zh ∈ [4000,5000]、g=ok。

- [ ] **Step 2: Tag**

```bash
git tag cuda-zh-adv-AG1-complete
```

---

# AG2 — 基础设施 + 框架(a05-a08)

## Task 6: a05 RDMA + NCCL transport

**Files:**
- Create: `docs/cuda-zh/advanced/a05-rdma-nccl-transport.md`

- [ ] **Step 1: 写章节(目标 4000-5000 字 + Mermaid ≥ 3)**

**章名:** `# a05 · RDMA + NCCL transport — 跨节点通信完整路径`

**§1(200-300 字):** 主体教程 14 章 NVLink/NVSwitch 在单节点 8 GPU 内,跨节点(InfiniBand NDR 400G、GPUDirect RDMA、NCCL transport plugin)从来没讲;多节点训练必须懂。

**§2 + Mermaid ≥ 3(500-600 字):**
- **Mermaid `flowchart LR`** 画跨节点 GPU-to-GPU 完整路径:GPU A → GPU mem → 同节点 NIC(经 PCIe / NVLink-C2C 到 RDMA verbs)→ IB switch → 对端 NIC → 对端 GPU mem → GPU B
- **Mermaid `flowchart TB`** rail-optimized topology:8 GPU 节点 × 8 IB rail(每 GPU 一条独立 rail)→ spine switch → 跨节点
- **Mermaid `sequenceDiagram`** GPUDirect RDMA bypass CPU:无 GDR 时数据经 CPU memory 中转,GDR 直 DMA GPU memory ↔ NIC
- 关键概念:
  - InfiniBand verbs API(`ibv_post_send`)
  - GPUDirect RDMA(GDR):允许 NIC 直读 / 写 GPU memory
  - NCCL transport plugin layer:`NCCL_NET_PLUGIN` 指定后端
  - rail-optimized:同 PCIe switch 下 GPU + NIC pair,避免共享 rail bandwidth halved

**§3(300-400 字):**
- 环境变量:`NCCL_DEBUG=INFO`、`NCCL_NET=IB / Socket`、`NCCL_IB_HCA=mlx5_0:1`、`NCCL_IB_DISABLE=0`、`NCCL_TOPO_FILE=/path/topo.xml`、`NCCL_NET_GDR_LEVEL`
- Mellanox HCA OFED 安装 + verify:`ibstat`、`ibv_devinfo`、`mlxlink -d /dev/mst/...`
- OpenMPI / UCX 集成
- 监控:`perfquery -P 1`(IB counters)

**§4(400-500 字):**
- 实测数字:
  - NDR 400G IB 单 link 50 GB/s 双向
  - rail-optimized 8-rail 节点跨节点总 400 GB/s
  - GPUDirect RDMA P2P 接近 NVLink 速率(同 PCIe switch ~80 GB/s)
  - 无 GDR 时 P2P 退到 ~40 GB/s(经 CPU memory)
  - NCCL allreduce 跨节点 256 GPU bf16 4 GB ~10 ms(rail-optimized)
  - 跨集群(cross-DC)latency ~50 µs / hop,allreduce 灾难性
- 反模式提示:rail 不对齐让 8 GPU 共享 1 rail bandwidth halved / IB Q-counter 不监控 silent link error / NCCL fallback TCP 但用户没发现

**§5 代码:** NCCL 启动完整环境变量 + Mellanox NIC PCIe 拓扑(同 NUMA GPU + NIC)

**§6:** NCCL_DEBUG=INFO 解析 transport 选择行;`ib_send_bw / ib_write_bw` 跑链路 benchmark

**§7(400-500 字):**
1. rail-unaware 部署(同 PCIe switch 下多 GPU 共享 rail)
2. IB Q-counters 没监控(silent link error 让 NCCL fallback TCP)
3. forgot GDR(P2P 退 50%)
4. NCCL TOPO_FILE 拓扑错(NCCL 选错 transport)
5. 跨集群同步 collective(latency 灾难)
6. NCCL_BUFFSIZE 默认在大集群不够用

**§8:** NCCL User Guide(transport 章)、NVIDIA GPUDirect RDMA docs、Mellanox OFED docs、Pathways paper(Google 跨主机大规模训练)、nccl-tests github

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/advanced/a05-rdma-nccl-transport.md
git commit -m "docs(cuda-zh/advanced): a05 RDMA + NCCL transport — IB NDR + GDR + rail-optimized"
```

---

## Task 7: a06 训练可靠性 + SDC

**Files:**
- Create: `docs/cuda-zh/advanced/a06-fault-tolerance-and-sdc.md`

- [ ] **Step 1: 写章节(目标 4000-5000 字 + Mermaid ≥ 2)**

**章名:** `# a06 · 训练可靠性 + SDC — checkpoint / failover / 静默数据损坏`

**§1(200-300 字):** 1k+ GPU 训练时 GPU 故障率 + SDC(silent data corruption)是头号 production issue,Llama 3 / Gemini 论文都点名;主体教程零提及,senior 必须懂的工程实践。

**§2 + Mermaid ≥ 2(500-600 字):**
- **Mermaid `sequenceDiagram`** GPU 故障完整路径:HBM ECC double-bit / SM xid → CUDA error → process abort → Slurm restart → checkpoint load → resume training
- **Mermaid `flowchart TD`** SDC 检测策略:replicated check(双跑 + 对比)、hash verify(periodic)、activation outlier scan、loss spike + 跨 rank 同步检测
- 关键概念:
  - GPU annualized failure rate(AFR)~2-3% / year(SXM5)
  - 1k GPU 训练:平均 30 GPU 失败 / 年
  - SDC:位翻转但 ECC 没报错(罕见但 1k 卡 ~1 / 周)
  - Loss spike 与 NaN propagation 跨 rank 传染

**§3(300-400 字):**
- `torch.distributed.checkpoint` (DCP):sharded state dict、`async_save`、`fsspec` 后端
- `torch.cuda.cudart()` 查 ECC error counter
- Megatron `--checkpoint-path` + `--load` 模式
- Pathways failover:detect → relayout → resume(无需 process restart)
- Slurm `--requeue` 自动重启

**§4(400-500 字):**
- 实测数字:
  - DCP async checkpoint:70B 模型 ~5s vs sync 2 min
  - Megatron 1k GPU 训练:平均每 6 小时一次 GPU 故障(SDC + ECC + xid)
  - SDC 在 Gemini Ultra 训练中触发 ~20 次/周
  - Cooldown 重启策略:同卡反复 fail → 隔离 30 min 自检
- 反模式提示:同步 checkpoint 阻塞训练 / 单点 checkpoint 不分片 / 忽略 SDC 信号当抖动

**§5 代码:** DCP async checkpoint 完整代码 + loss spike 检测 + 跨 rank NaN 同步

**§6:** NVIDIA DCGM + ` dcgm-exporter` Prometheus 监控;`nvidia-smi -q --xml-format` 查 ECC + xid;Megatron 训练日志解析 loss spike

**§7(400-500 字):**
1. 同步 checkpoint(每 1000 step 阻塞 2 分钟,job efficiency 降 5%)
2. 单点 checkpoint(load 10× 慢,80B 模型 30 分钟)
3. 忽略 SDC 信号(loss spike 当随机抖动,实际是 1 张卡位翻转)
4. 立即重启 + 同卡反复 fail(应 cooldown 30 min 自检)
5. 跨 rank NaN 不同步检测(只本 rank 报错,其他 rank 已生成无效数据)
6. checkpoint frequency 太低(失败丢 4 小时进度)
7. forgot DCGM(故障发生后才查日志,无 metric)

**§8:** Llama 3 405B training report(讨论 SDC + interruption)、Gemini paper、Megatron DCP 文档、Pathways paper、Slurm + NCCL 实战、NVIDIA DCGM docs

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/advanced/a06-fault-tolerance-and-sdc.md
git commit -m "docs(cuda-zh/advanced): a06 训练可靠性 + SDC — AFR/DCP async/Pathways failover"
```

---

## Task 8: a07 数据流水线工程化

**Files:**
- Create: `docs/cuda-zh/advanced/a07-data-pipeline-engineering.md`

- [ ] **Step 1: 写章节(目标 4000-5000 字 + Mermaid ≥ 2)**

**章名:** `# a07 · 数据流水线工程化 — CPU 瓶颈 / DALI / Ray Data`

**§1(200-300 字):** 主体教程全在 GPU 内,但真实训练 30% 时间在 host code(DataLoader、tokenize、shuffle、pinned memory copy);senior 调优必须懂的层。

**§2 + Mermaid ≥ 2(500-600 字):**
- **Mermaid `sequenceDiagram`** 训练数据完整路径:disk read → decompress → tokenize → shuffle(per-epoch)→ batch → pinned memory → H2D async → GPU 训练,标出每段瓶颈
- **Mermaid `flowchart LR`** PyTorch DataLoader worker 进程模型:main process → `num_workers` 子进程并行 → `pin_memory` 线程后台 H2D
- 关键概念:
  - DataLoader 单线程 GIL 限制 → `num_workers` 并行
  - `pin_memory=True` 让 H2D 走 DMA(否则同步 memcpy)
  - GPU-side decoding(DALI):JPEG decode、resize、normalize 在 GPU(释放 CPU)
  - Streaming dataset:不全加载到 RAM,边读边训

**§3(300-400 字):**
- PyTorch `DataLoader(num_workers=N, pin_memory=True, prefetch_factor=2, persistent_workers=True, sampler=...)`
- NVIDIA DALI(`nvidia.dali.pipeline.Pipeline`):`fn.readers.file / fn.decoders.image / fn.crop_mirror_normalize`
- FFCV:写 `.beton` 高效格式 + 多 worker 读
- Ray Data:分布式预处理 + `.map_batches`
- HuggingFace `datasets.load_dataset(..., streaming=True)`

**§4(400-500 字):**
- 实测数字:
  - PyTorch DataLoader 在 batch=1024 image:`num_workers=8` 比 1 快 6×
  - `pin_memory=True` 让 H2D 快 3-4×
  - DALI GPU-side decode 释放 CPU,H100 训练加速 8-15%
  - Ray Data 在多节点线性 scale(1k worker / 1k 节点)
  - `persistent_workers=False` 每 epoch 重启进程(50 epoch × 5s overhead = 4 min 浪费)
- 反模式提示:`num_workers=0` 单线程 / 漏 pin_memory / tokenize 在 DataLoader CPU bound

**§5 代码:** PyTorch DataLoader + pin_memory + prefetch + persistent_workers 完整模板;DALI image pipeline 配置;Ray Data 分布式 map_batches

**§6:** PyTorch profiler 看 DataLoader idle 占比;`nvidia-smi dmon -s u` 监控 GPU idle 等数据时间;perf top 看 CPU 热点

**§7(400-500 字):**
1. `num_workers=0`(主线程串行,GPU idle 50%)
2. `num_workers` 太多(>16,文件 IO contention + memory 压力)
3. forgot `pin_memory=True`(H2D 慢 3-4×)
4. `persistent_workers=False`(每 epoch 重启 5s × 50 = 4 min)
5. tokenize 在 DataLoader CPU 而非离线预处理(瓶颈)
6. `prefetch_factor=1` 默认(下一 batch 没就绪)
7. shuffle 用 full-dataset shuffle 而非 buffer shuffle(内存爆)

**§8:** PyTorch DataLoader docs、DALI docs、FFCV paper、Ray Data docs、HuggingFace datasets streaming、Webdataset(POSIX tar 流式)

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/advanced/a07-data-pipeline-engineering.md
git commit -m "docs(cuda-zh/advanced): a07 数据流水线 — DataLoader / DALI / Ray Data / pinned memory"
```

---

## Task 9: a08 cuDNN + cuBLAS + cuBLASLt 高级

**Files:**
- Create: `docs/cuda-zh/advanced/a08-cudnn-cublas-advanced.md`

- [ ] **Step 1: 写章节(目标 4000-5000 字 + Mermaid ≥ 2)**

**章名:** `# a08 · cuDNN + cuBLAS + cuBLASLt 高级 — algorithm heuristic + backend graph`

**§1(200-300 字):** cuDNN / cuBLAS / cuBLASLt 是 PyTorch / TensorRT / TE 的 fallback path,生产必碰;教程提了名字但没讲 algorithm heuristic、backend API graph、何时绕开调 Triton/CUTLASS。

**§2 + Mermaid ≥ 2(500-600 字):**
- **Mermaid `classDiagram`** 三者关系:`cuBLAS legacy` 单算子 GEMM → `cuBLASLt` fused matmul + epilogue(bias、ReLU、GELU)→ `cuDNN backend API` graph(任意 op 融合)
- **Mermaid `flowchart LR`** algorithm heuristic 选择路径:输入(dtype + shape + workspace)→ heuristic predicate → 候选 algo list → benchmark / model-based 选最优 → cubin
- 关键概念:
  - cuBLASLt fused epilogue:matmul + bias + activation 一个 kernel
  - cuDNN backend API:graph-based,任意 op 拼接(forward + backward 一起)
  - heuristic mode:`CUBLASLT_MATMUL_PREF_HEURISTIC_MODE`(default vs aggressive)
  - workspace size 影响 algo 选择

**§3(300-400 字):**
- cuBLASLt: `cublasLtMatmulAlgoGetHeuristic(handle, computeDesc, ATy, BTy, CTy, DTy, preference, K, algos)` + `cublasLtMatmulDescSetAttribute(EPILOGUE=GELU)`
- cuDNN backend graph: `cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_GRAPH_DESCRIPTOR)` + Finalize + Execute
- cuBLAS legacy: `cublasGemmEx`(deprecated for new use)
- 何时手写:cuDNN 没覆盖的 op pattern;cuBLASLt heuristic 选差时;新 dtype 等 cuDNN 跟进

**§4(400-500 字):**
- 实测数字:
  - cuBLASLt vs cuBLAS legacy:matmul + bias 融合让 Llama-70B forward 加速 10-15%
  - cuDNN backend API 在 attention forward 与 FlashAttention 持平 ±5%(取决于 seq_len)
  - heuristic 内部用 ML 模型选(cuDNN 8.6+ predicate-based)
  - workspace 32 MiB 让 GEMM heuristic 选 split-K 算法(快 20%)
  - 新 dtype(FP8)cuBLASLt 8.0+ 才有 fused epilogue(此前走 legacy)
- 反模式提示:forgot workspace / heuristic 没 set preference 选差 algo / cuDNN backend graph 没 finalize 直接 execute(undefined)

**§5 代码:** cuBLASLt matmul + bias + GELU epilogue 完整 C++ 调用;cuDNN backend graph 构造 attention forward;workspace 配 ~32 MiB

**§6:** cuDNN logging(`CUDNN_LOGINFO_DBG=1`);cuBLAS heuristic enum 查询;ncu profile 看 algorithm 选择

**§7(400-500 字):**
1. forgot workspace(每次 alloc 慢 + 选差 algo)
2. cuBLASLt heuristic 没 set preference(选 default 不一定最优)
3. cuDNN backend graph 没 finalize 直接 execute(UB)
4. 新 dtype(FP8 / FP4)走 legacy API(没 fused epilogue,慢 15%)
5. 忽略 cuDNN log(silent fall back 到 slow path)
6. attention 自己写但实际 cuDNN backend graph 已有 + 经过 NVIDIA tune(自己实现慢)
7. cuBLASLt epilogue 设置不全(bias + activation 漏一个 → 退到无 epilogue 慢路径)

**§8:** cuBLAS docs、cuBLASLt heuristics docs、cuDNN backend API docs、Transformer Engine 源码(用 cuBLASLt + cuDNN)、PyTorch ATen 的 cuBLASLt 集成代码

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/advanced/a08-cudnn-cublas-advanced.md
git commit -m "docs(cuda-zh/advanced): a08 cuDNN/cuBLAS/cuBLASLt 高级 — heuristic + backend graph"
```

---

## Task 10: AG2 验证 + tag

- [ ] **Step 1: 批量验证 + tag**

```bash
for f in docs/cuda-zh/advanced/a0[5-8]-*.md; do
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

git tag cuda-zh-adv-AG2-complete
```

---

# AG3 — 下一代 + 部署(a09-a10)

## Task 11: a09 Blackwell B200 / GB200 NVL72

**Files:**
- Create: `docs/cuda-zh/advanced/a09-blackwell-b200-gb200.md`

- [ ] **Step 1: 写章节(目标 4000-5000 字 + Mermaid ≥ 3)**

**章名:** `# a09 · Blackwell B200 / GB200 NVL72 — 下一代架构`

**§1(200-300 字):** Blackwell SM_100 是 NVIDIA 2024-2025 主推架构;2nd gen Transformer Engine + FP4 + 5th gen NVLink 1.8 TB/s + HBM3e 192 GB;GB200 NVL72 用 36 Grace CPU + 72 Blackwell GPU 全 NVLink 互连;senior 必关注。

**§2 + Mermaid ≥ 3(500-600 字):**
- **Mermaid `flowchart TB`** 画 B200 SM_100 全景:2 die NVLink-C2C 互连 → 208 SM(单 die 104)→ 192 GB HBM3e × 8 TB/s → 5th gen NVLink 1.8 TB/s/GPU
- **Mermaid `flowchart LR`** GB200 NVL72 拓扑:36 Grace CPU + 72 Blackwell GPU,全 NVLink 互连(1.4 EFLOPS FP8)
- **Mermaid `classDiagram`** TE 2nd gen 数据通路:FP4 + FP6 + FP8 + BF16 多精度 mma,sparsity 2:4
- 关键 spec:
  - 2 die x 104 SM = 208 SM per chip(数字以 NVIDIA 公告为准)
  - HBM3e 192 GB / 8 TB/s
  - 5th gen NVLink 1.8 TB/s / GPU(vs Hopper 900 GB/s 双倍)
  - GB200 NVL72:36 Grace + 72 B200 全连接,Coherent CPU-GPU memory
  - FP4 TC peak ~20 PFLOPS / GPU(2:4 sparsity 40 PFLOPS)
  - 内置 decompression engine 加速 LZ4/Snappy GMEM 解压

**§3(300-400 字):**
- CUDA 12.4+ 支持 SM_100;`-arch=sm_100`、`-arch=sm_100a`(部分 wgmma 变体)
- Transformer Engine v1.10+ 自动 FP4 + FP8 选择
- GB200 cross-Grace memory:`cudaMallocAsync` 支持 CPU-GPU 统一池
- NVLink-C2C 900 GB/s 双向(Grace ↔ Blackwell)
- `cuda::std::memcpy_async` 接 NVLink-C2C
- Decompression engine API(读 LZ4 / Snappy 压缩 GMEM)

**§4(400-500 字):**
- 实测 / 预期数字:
  - B200 FP8 TC peak ~10 PFLOPS / GPU(vs H100 ~2 PFLOPS,5×)
  - B200 FP4 TC peak ~20 PFLOPS;+2:4 sparsity ~40 PFLOPS
  - HBM3e 8 TB/s vs Hopper 5 TB/s(1.6×)
  - NVL72 系统 1.4 EFLOPS FP8 / 720 PFLOPS FP4
  - Grace CPU + Blackwell GPU 通过 NVLink-C2C 900 GB/s 双向 vs PCIe 64 GB/s
  - GB200 NVL72 训练 Llama-3-70B 比 H100 DGX SuperPOD 快 ~3×(NVIDIA 公告)
- 反模式提示:FP4 直接训练(应 QAT 或推理);SM_100 写 SM_90a 代码(部分 wgmma 兼容但 mma_async size 变);GB200 单 GPU 当 H100 用(漏 NVLink-C2C)

**§5 代码:** Blackwell FP4 inference 代码(TE v1.10+);GB200 cross-CPU-GPU memory NVLink-C2C 例子;decompression engine API 示例

**§6:** NSight Compute SM_100 metric(`sm__pipe_tensor_op_fp4_*`);GB200 + NVLink fabric 监控

**§7(400-500 字):**
1. FP4 直接训练(精度损失爆;应 QAT 或仅推理)
2. SM_100 编译 SM_90a 代码(部分 wgmma 兼容但 mma_async tile size 变)
3. GB200 单 GPU 配置(漏 NVLink-C2C 让 CPU offload 退到 PCIe)
4. 用 cudaMallocAsync 但 mempool 没指定 location(default 在 GPU,跨 Grace 退化)
5. forgot decompression engine 在 GMEM 压缩数据上(慢路径)
6. TE v1.9 跑 B200(没有 FP4 路径,退 FP8)
7. NVL72 训练设 PP 跨 chassis(NVLink 内 vs 跨 chassis latency 显著)

**§8:** Blackwell Whitepaper、GB200 NVL72 reference architecture、Transformer Engine v1.10+ release notes、NVLink-C2C 文档、NVIDIA GTC 2024 Blackwell keynote

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/advanced/a09-blackwell-b200-gb200.md
git commit -m "docs(cuda-zh/advanced): a09 Blackwell B200/GB200 NVL72 — 2nd gen TE + FP4 + 5th NVLink"
```

---

## Task 12: a10 MIG + confidential + vGPU

**Files:**
- Create: `docs/cuda-zh/advanced/a10-mig-confidential-vgpu.md`

- [ ] **Step 1: 写章节(目标 4000-5000 字 + Mermaid ≥ 2)**

**章名:** `# a10 · MIG + Confidential Compute + vGPU — 多租户部署`

**§1(200-300 字):** 生产部署里多租户共享 GPU 必须懂:MIG 硬件级切分、Confidential Compute(SEV-SNP / TDX)合规、vGPU 虚拟化、容器化(NVIDIA Container Toolkit)。

**§2 + Mermaid ≥ 2(500-600 字):**
- **Mermaid `flowchart TB`** MIG 切分:H100 SXM5 max 7 instances,每 instance 含 SM × 16 + L2 partition + HBM partition,硬件隔离
- **Mermaid `sequenceDiagram`** confidential compute(SEV-SNP)kernel 加密路径:user code → encrypted boundary → driver → encrypted memcpy → GPU MMU 解密 → kernel 执行 → 加密返回
- 关键概念:
  - MIG profiles:`1g.10gb / 2g.20gb / 3g.40gb / 4g.40gb / 7g.80gb`(7 = 全 GPU)
  - MIG instance 独立 SM + L2 + HBM partition,硬件隔离
  - SEV-SNP / TDX:CPU 端机密计算扩到 GPU
  - vGPU:虚拟化(Time-Slice / Multi-Instance vGPU)
  - NVIDIA Container Toolkit:`--gpus all` Docker / containerd 集成

**§3(300-400 字):**
- MIG:
  - `nvidia-smi mig -cgi 9,9,9,9,9 -C`(创建 5 个 1g.10gb instance)
  - `nvidia-smi -L` 列 instance
  - `CUDA_VISIBLE_DEVICES=MIG-GPU-xxx` 选 instance
- Confidential Compute:
  - `NVIDIA_REQUIRE_CC=on` + AMD SEV-SNP / Intel TDX host
  - `nvidia-smi conf-compute -f` 查状态
- vGPU:NVIDIA grid driver + VMware / KVM;`vGPU profile A40-12Q`
- NVIDIA Container Toolkit:
  - `docker run --gpus all`
  - `docker run --gpus 'device=0,1'`
  - Kubernetes Device Plugin

**§4(400-500 字):**
- 实测数字:
  - MIG 7-instance 总吞吐 ~90% 单卡(隔离开销 10%)
  - Confidential compute 性能开销 5-15%(加密 path)
  - vGPU H100 max 8 user / GPU(Time-Slice)或 7 user(MIG-based)
  - MIG 实例之间 NCCL 不支持(P2P disabled)
- 反模式提示:MIG 配置后 NCCL 跨 instance(报错)/ confidential kernel 用 CPU 共享 mem(打破隔离)/ vGPU 漏 license

**§5 代码:** MIG 切分 + container 部署完整命令;confidential GPU 拉起 + verify;Kubernetes NVIDIA Device Plugin + MIG 配置(`gpuClient: 1g.10gb`)

**§6:** `nvidia-smi mig --list-gpu-instance-profiles`;DCGM-exporter 暴露 MIG metric;`kubectl describe nodes` 看 MIG resource

**§7(400-500 字):**
1. MIG 配置后 NCCL 跨 instance 通信(P2P disabled,报错)
2. Confidential kernel 误用 CPU 共享内存(打破加密隔离)
3. vGPU license 漏配(driver fallback 单卡 6 小时)
4. MIG 切了 7 instance 但 workload 需要 80GB(应不切)
5. Container 漏 `--gpus all`(看到 GPU 但跑不了)
6. Kubernetes Device Plugin 配置错(pod 起不来或共享)
7. MIG instance 间 KV cache 跨 不支持(LLM serving 必须考虑)

**§8:** NVIDIA MIG User Guide、Confidential Compute Whitepaper、NVIDIA vGPU 文档、NVIDIA Container Toolkit github、Kubernetes Device Plugin docs

- [ ] **Step 2: 验证 + Commit**

```bash
git add docs/cuda-zh/advanced/a10-mig-confidential-vgpu.md
git commit -m "docs(cuda-zh/advanced): a10 MIG + Confidential + vGPU — 多租户部署"
```

---

## Task 13: AG3 验证 + tag

- [ ] **Step 1: 批量验证 + tag**

```bash
for f in docs/cuda-zh/advanced/a0[9]-*.md docs/cuda-zh/advanced/a10-*.md; do
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

git tag cuda-zh-adv-AG3-complete
```

---

# AG4 — 索引 + 全集验证 + push

## Task 14: 更新 00-index.md 加 advanced 表

**Files:**
- Modify: `docs/cuda-zh/00-index.md`

- [ ] **Step 1: 在 §8 节末尾追加 advanced 表格**

打开 `docs/cuda-zh/00-index.md` §8 节。在现有"### 本教程章节索引"表格之后(以及其他子表之后),追加新的子节:

```markdown
### 进阶专题(advanced/)

适合 senior AI Infra,真实生产中常卡壳的 10 个深度主题。

| # | 标题 | 主题 |
|---|---|---|
| [a01](advanced/a01-moe-expert-parallelism.md) | MoE + Expert Parallelism | Mixtral/DeepSeek-V3 训推 + DeepEP/Megablocks |
| [a02](advanced/a02-cutlass-3x-and-cute.md) | CUTLASS 3.x + CuTe Layout | collective mainloop + Layout 代数 |
| [a03](advanced/a03-quantization-algorithms.md) | 量化算法原理 | GPTQ / AWQ / SmoothQuant / FP8 scaling |
| [a04](advanced/a04-triton-kernel-engineering.md) | Triton 工程化 | compiler stack + autotune + torch.compile |
| [a05](advanced/a05-rdma-nccl-transport.md) | RDMA + NCCL transport | NDR 400G IB + GDR + rail-optimized |
| [a06](advanced/a06-fault-tolerance-and-sdc.md) | 训练可靠性 + SDC | AFR + DCP async + loss spike debug |
| [a07](advanced/a07-data-pipeline-engineering.md) | 数据流水线工程化 | DALI/FFCV/Ray Data + GPU-side decode |
| [a08](advanced/a08-cudnn-cublas-advanced.md) | cuDNN/cuBLAS/cuBLASLt 高级 | algorithm heuristic + backend graph |
| [a09](advanced/a09-blackwell-b200-gb200.md) | Blackwell B200 / GB200 NVL72 | 2nd gen TE + FP4 + 5th gen NVLink |
| [a10](advanced/a10-mig-confidential-vgpu.md) | MIG + confidential + vGPU | 多租户 + SEV-SNP + 容器化 |
```

- [ ] **Step 2: 验证 + Commit**

```bash
# 00-index.md 字数应仍在 4000-5000 范围(advanced 表只增 ~80 字)
F=docs/cuda-zh/00-index.md
sec=$(grep -c '^## [1-8]\. ' $F); mer=$(grep -c '^```mermaid' $F)
zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text('utf-8')
t = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', t, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', t)))
" $F)
echo "$F: sec=$sec mer=$mer zh=$zh"
# 期望 sec=8 mer≥3 zh ∈ [4000, 5200](允许 advanced 表小幅超过 5000 上限)

git add docs/cuda-zh/00-index.md
git commit -m "docs(cuda-zh): 00 索引追加 advanced 进阶专题 10 章链接表"
```

---

## Task 15: 全集验证 + tag + push

- [ ] **Step 1: 全集验证脚本**

```bash
echo "=== cuda-zh 主体 25 章 ==="
ls -1 docs/cuda-zh/*.md | wc -l                # expect 25

echo ""
echo "=== cuda-zh advanced 10 章 ==="
ls -1 docs/cuda-zh/advanced/*.md | wc -l       # expect 10

echo ""
echo "=== 每章详细统计 ==="
for f in $(ls -1 docs/cuda-zh/*.md docs/cuda-zh/advanced/*.md | sort); do
    sec=$(grep -c "^## [1-8]\. " "$f")
    mer=$(grep -c '^```mermaid' "$f")
    zh=$(.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text('utf-8')
t = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', t, flags=re.DOTALL)
print(len(re.findall(r'[一-鿿]', t)))
" "$f")
    grep -qi 'gpusim' "$f" && g="GPUSIM_FOUND" || g="ok"
    expect_mer=2
    [[ "$f" == */00-* ]] && expect_mer=3
    if (( zh < 4000 || zh > 5500 )); then zh_status="OUT"; else zh_status="ok"; fi
    if (( mer < expect_mer )); then mer_status="LOW"; else mer_status="ok"; fi
    printf "%-55s sec=%d mer=%d(%s) zh=%d(%s) %s\n" "$f" "$sec" "$mer" "$mer_status" "$zh" "$zh_status" "$g"
done

echo ""
echo "=== 总 mermaid (应 ≥ 73) ==="
grep -c '^```mermaid' docs/cuda-zh/*.md docs/cuda-zh/advanced/*.md | awk -F: '{s+=$2} END {print s}'
```

预期:
- 主体 25 文件 + advanced 10 文件 = 35 文件
- 每章 sec=8、mer ≥ 2(00 章 ≥ 3 + advanced 章节自定)
- 字数 [4000, 5500](允许 00-index 因为 advanced 表小幅超 5000)
- 全部 gpusim=ok
- 总 mermaid ≥ 73

如果有不达标,先回到对应任务修复。

- [ ] **Step 2: Tag**

```bash
git tag cuda-zh-adv-complete
```

- [ ] **Step 3: Push 到 GitHub**

```bash
git push origin master
git push origin cuda-zh-adv-AG1-complete cuda-zh-adv-AG2-complete cuda-zh-adv-AG3-complete cuda-zh-adv-complete
```

预期输出:4 个新 tag + master 同步;新远端 ref 总数比之前多 4。

---

## 验收准则

- [ ] 10 个 advanced 章节全部存在,8 节齐全,字数 ∈ [4000, 5000],Mermaid ≥ 2(a01/a02/a05/a09 ≥ 3),零 gpusim
- [ ] `00-index.md` §8 包含 10 章 advanced 链接表
- [ ] cuda-zh 全集 35 个 markdown
- [ ] 全集 Mermaid 总数 ≥ 73
- [ ] 4 个 milestone tag + 1 ship tag(`cuda-zh-adv-AG1-complete` ... `cuda-zh-adv-AG3-complete` + `cuda-zh-adv-complete`)
- [ ] master + 全部新 tag 推送到 origin
