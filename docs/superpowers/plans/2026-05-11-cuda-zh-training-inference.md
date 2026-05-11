# CUDA-zh v2 训练 + 推理章节 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 cuda-zh 教程加 2 章 capstone(23 训练全栈、24 推理全栈)+ 索引更新 + push 到 GitHub。

**Architecture:** 沿用 cuda-zh 既定规范(8 节结构、Mermaid 强制、1500-2500 中文字、零 gpusim、UTF-8 LF)。本批次的 §7 节内容从"反模式"改写为"优化方法体系"(本批 capstone 章特色),原反模式简化为 §4 末尾的"反模式提示"段。

**Tech Stack:** Markdown,Mermaid。无代码,无测试 — 验证靠 grep + 字数。

---

## 全局规则

### 文件目录
- 新增章节写入 `docs/cuda-zh/23-training-end-to-end.md`、`docs/cuda-zh/24-inference-end-to-end.md`
- 修改 `docs/cuda-zh/00-index.md`(§1 加第三条阅读路径,§8 表格补 2 行)

### 每章标准结构(强制)

```markdown
# NN · <中文标题>

> **一句话总结。**

## 1. 是什么 / 为什么有它
## 2. 硬件视角(组件触发链)
## 3. CUDA / 框架编程接口
## 4. 关键性能指标
## 5. 代码示例
## 6. 实测手段
## 7. <训练侧 / 推理侧>优化方法体系   ← capstone 特色:写优化方法,非反模式
## 8. 延伸阅读
```

注意 §7 标题为本批 capstone 特化(原"常见反模式"改为"优化方法体系")。原反模式内容并入 §4 末尾的"反模式提示"段,保持信息覆盖完整。

### Mermaid 强制
- 每章 §2 必须有 1 个 `flowchart TB`(组件触发链)
- §7 推荐再加 1 个(可选,展示优化方法分类)
- 全集 mermaid 总数应从 24 升至 ≥ 26

### 内容质量
- 1500-2500 中文字(代码块 + Mermaid 不计)
- 零 gpusim 引用(`grep -i gpusim` 空命中)
- 真实 API / 论文名(PyTorch DDP、`torch.cuda.graph()`、`ncclAllReduce`、PagedAttention、FlashAttention-3、ZeRO 等真名)
- 无营销语,无友商对比

### 验证脚本(每章写完跑)

```bash
F=docs/cuda-zh/NN-xxx.md
echo "8 sections: $(grep -c '^## [1-8]\. ' $F) (expect 8)"
echo "mermaid: $(grep -c '^```mermaid' $F) (expect ≥ 1)"
.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding='utf-8')
text = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', text, flags=re.DOTALL)
zh = re.findall(r'[一-鿿]', text)
print(f'  {len(zh)} 中文字 (expect 1500-2500)')
" $F
! grep -i 'gpusim' $F && echo "no gpusim ref OK"
```

---

## Task 1: 第 23 章 — 模型训练全栈串联

**Files:**
- Create: `docs/cuda-zh/23-training-end-to-end.md`

- [ ] **Step 1: 写章节内容**

按下面要点写,严格遵循 8 节结构 + 1500-2500 字 + ≥ 1 Mermaid:

**章名行:** `# 23 · 模型训练全栈串联 — 一次 step 如何调度全部 GPU 组件`

**§1 是什么 / 为什么有它(150-250 字):**
- 训练 step = forward + backward + optimizer + 梯度同步,组件触发模式与单 kernel benchmark 完全不同
- 一次 step 几乎触发前 22 章覆盖的所有内容
- 理解串联是定位训练性能瓶颈的前提

**§2 硬件视角 — 组件触发链(300-450 字 + 1 个必有 Mermaid):**
- Mermaid `flowchart TB`,展示一次 transformer training step 的组件触发链:
  ```
  权重/激活/梯度都在 HBM
  → forward 一层:
    HBM read (weights via TMA → SMEM)
    → wgmma (TC) compute on tile
    → mbarrier sync
    → activation 写回 mempool 块
  → loss + backward:类似但反向
  → grad allreduce (NCCL via NVLink)
  → optimizer kernel (FP32 master + bf16 cast)
  → 重复
  ```
- 文字部分配套解释为何 forward 是 compute-bound、backward 多一倍 mma、optimizer 是 memory-bound

**§3 CUDA / 框架编程接口(200-300 字):**
- PyTorch DDP / FSDP(`torch.nn.parallel.DistributedDataParallel`、`torch.distributed.fsdp.FullyShardedDataParallel`)
- DeepSpeed(ZeRO-3 一键化)、Megatron-LM(TP + PP)
- `torch.cuda.graph()` 把 step capture 为 cudaGraph
- `torch.cuda.amp.autocast(dtype=torch.bfloat16)` + `torch.cuda.amp.GradScaler`(混精)
- `torch.distributed.all_reduce` 直接 NCCL 入口

**§4 关键性能指标(150-250 字):**
- MFU(Model FLOPs Utilization)= `flops_executed / (time × peak_flops)`,Hopper bf16 训练良好水平 50-60%
- HBW(HBM Bandwidth Utilization)
- Scaling efficiency = `actual_throughput / (single_gpu_throughput × N)`
- NCCL bus bandwidth(单 step 的 sustained allreduce 带宽 / NVLink 峰值)
- tokens / sec / GPU
- **反模式提示:** 没有 graph capture(每 step ~5 µs launch 开销 × 上百 launch)、grad sync 阻塞下一 forward、fp32 master 与 bf16 compute 混乱导致精度漂移

**§5 代码示例(代码块,不计字数):**
一段 PyTorch 风格 training step 伪代码,每行右注释命中组件:
```python
for batch in loader:                              # H2D copy → HBM
    with torch.cuda.amp.autocast(torch.bfloat16):
        out = model(batch.input)                   # forward: TMA→SMEM→wgmma→HBM
        loss = criterion(out, batch.target)        # 小 kernel + reduce
    scaler.scale(loss).backward()                  # backward: 双倍 mma + grad 写 HBM
    # FSDP 下: reduce_scatter grads 与最后一层 backward 重叠
    scaler.step(optimizer)                         # NCCL allreduce + optimizer kernel
    scaler.update()                                # FP32 master 更新
```

**§6 实测手段(150-200 字):**
- NSight Systems(`nsys profile -t cuda,nvtx,nccl python train.py`)看 NCCL 与 compute lane 是否重叠
- NSight Compute(`ncu --set full kernel`)看 wgmma fragment 利用率、TC throughput
- PyTorch profiler(`torch.profiler.profile(activities=[CUDA, CPU], with_stack=True)`)输出 chrome trace
- `torch.cuda.memory._snapshot()` 看 mempool 内 KV/activation 实时分布
- MFU 计算 = `6 × P × tokens / (step_time × peak_bf16_flops)`(P 为参数量,6 ≈ fwd+bwd 系数)

**§7 训练侧优化方法体系(400-600 字):**
分组列出 8-12 个方法,每个标注「命中组件 + 何时用」:

**算子层:**
- 算子融合 — FlashAttention-3 / Apex fused layer-norm:减少 HBM 读写次数。命中:HBM、SMEM、TC
- FP8 training(Hopper TC FP8) — `torch.float8_e4m3fn` weight + `e5m2` grad,2× TC throughput vs bf16
- 混精训练 — bf16 compute + fp32 master,平衡精度与吞吐

**显存层:**
- Activation checkpointing — 用算力换显存,backward 时重算 activation
- Gradient accumulation — 累 N 个 micro-batch 再 sync,降低有效通信频率
- Selective recompute — 只重算 attention(FLOPs 占比小的层),平衡显存 vs FLOPs

**分布式层:**
- ZeRO-1/2/3 与 FSDP — 优化器状态 / 梯度 / 参数分别 shard,显存 N× 节省
- Tensor parallelism(Megatron) — 矩阵乘按行/列切,allreduce 拼接
- Pipeline parallelism(GPipe / 1F1B) — 跨 GPU 切层 + micro-batch 流水
- Sequence parallelism — 沿 seq 切分 attention/layernorm,减小 activation 显存

**调度层:**
- Compute-comm 重叠 — FSDP 把 reduce-scatter 与上一层 backward 同 stream issue,通信完全隐藏
- CUDA Graph training step capture — 固定 shape 训练每 step 节省百次 launch overhead

**§8 延伸阅读(代码块,不计字数):**
- PyTorch DDP / FSDP 文档(`pytorch.org/docs/stable/distributed.html`)
- Megatron-LM(`github.com/NVIDIA/Megatron-LM`)
- DeepSpeed(`deepspeed.ai`)
- FlashAttention-3 paper(`arxiv.org/abs/2407.08608`)
- ZeRO paper(`arxiv.org/abs/1910.02054`)
- Hopper FP8 Training Guide(`docs.nvidia.com` 上的 Transformer Engine docs)

- [ ] **Step 2: 本地验证**

```bash
F=docs/cuda-zh/23-training-end-to-end.md
grep -c '^## [1-8]\. ' $F          # expect 8
grep -c '^```mermaid' $F            # expect ≥ 1
.venv/bin/python -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding='utf-8')
text = re.sub(r'\`\`\`[^\n]*\n.*?\n\`\`\`', '', text, flags=re.DOTALL)
zh = re.findall(r'[一-鿿]', text)
print(f'  {len(zh)} 中文字 (expect 1500-2500)')
" $F
! grep -i 'gpusim' $F && echo "no gpusim ref OK"
```

字数若不在 1500-2500,扩 §1/§4/§7 或精简 §7 直到达标。

- [ ] **Step 3: Commit**

```bash
git add docs/cuda-zh/23-training-end-to-end.md
git commit -m "docs(cuda-zh): 23 模型训练全栈串联 — capstone 章"
```

---

## Task 2: 第 24 章 — 模型推理全栈串联

**Files:**
- Create: `docs/cuda-zh/24-inference-end-to-end.md`

- [ ] **Step 1: 写章节内容**

按下面要点写,严格遵循 8 节结构 + 1500-2500 字 + ≥ 1 Mermaid:

**章名行:** `# 24 · 模型推理全栈串联 — prefill 与 decode 如何调度全部 GPU 组件`

**§1 是什么 / 为什么有它(150-250 字):**
- LLM serving 由 prefill(处理输入 prompt)+ decode(逐 token 生成)两阶段组成
- prefill 是 compute-bound(GEMM 主导),decode 是 memory-bound(GEMV + KV cache HBM read)
- 大多数生产 serving 时间在 decode,优化点与训练显著不同

**§2 硬件视角 — prefill vs decode(300-450 字 + 1 个必有 Mermaid):**
- Mermaid `flowchart TB`,展示两路径:
  ```
  请求 → tokenize → embed (HBM lookup)
    → prefill 路径(若新请求): TMA load weight tile → wgmma TC compute → KV cache 写入 HBM(整段)→ ...
    → decode 路径(每步 1 token):
       HBM read weights (memory-bound) → 小 GEMV
       → 读 KV cache(paged 寻址)→ flash-attention(SMEM 主导)
       → output projection (TC GEMM)
       → sample(host)→ 下一步
  → continuous batching:多请求 prefill / decode 异步交错(同 stream 池)
  ```

**§3 CUDA / 框架编程接口(200-300 字):**
- vLLM(`vllm.LLM`,自带 PagedAttention)、TensorRT-LLM(NVIDIA 官方)、TGI(HuggingFace)、SGLang、DeepSpeed-MII
- `torch.cuda.graph()` 在固定 batch 下 capture decode step
- PagedAttention 自定义 CUDA kernel(每 KV block 16 token / 32 token)
- `cudaMallocAsync` 用作 KV cache slab 池
- TensorRT-LLM `Engine.refit()` + INT8/FP8 quantization API

**§4 关键性能指标(150-250 字):**
- First-Token Latency(FTL)— prefill 完成时间
- Time-Per-Output-Token(TPOT)— decode 平均每 token 延迟
- Throughput(tokens/sec/GPU,通常 batch 越大越高)
- KV cache bytes / token(取决于 num_layers × hidden × dtype)
- HBM 利用率(decode 多数 cycle 在 HBM read,目标 70%+)
- **反模式提示:** 静态 batch 浪费 GPU(decode batch=1 利用率 5%)、KV cache 不分页 → 内存碎片严重、漏 flash-attention(标准 attention 反复读写 HBM)、INT8 weight 但漏 activation 量化(精度损失大)

**§5 代码示例(代码块,不计字数):**
一段 vLLM-风格 decode step 伪代码,每行右注释命中组件:
```python
while not stop_token:
    # 1. attention layer
    q = q_proj(x)                       # GEMV (HBM read weights, memory-bound)
    k_new, v_new = kv_proj(x)           # GEMV
    kv_cache.append_paged(k_new, v_new) # paged HBM write
    # 2. flash attention(SMEM 高效复用)
    o = flash_attn_paged(q, kv_cache, block_table)  # SMEM tile + softmax
    # 3. output proj + FFN
    x = o_proj(o); x = ffn(x)           # 大 GEMM,TC GEMM
    # 4. sample next token
    logits = lm_head(x); tok = sample(logits)  # host CPU 控制
```

**§6 实测手段(150-200 字):**
- NSight Systems 看 prefill / decode 时间分布,统计 TPOT
- TensorRT-LLM benchmark 工具(`benchmarks/cpp`)给定 dataset + batch 测吞吐
- vLLM 内置 metrics endpoint(`/metrics` 暴露 KV cache 利用率、queue length)
- `nvidia-smi` 看 KV cache HBM 占用
- NSight Compute 单测 attention kernel 看 SMEM/flash 利用与 wave occupancy
- `nccl-tests` 验证 TP allreduce 带宽

**§7 推理侧优化方法体系(400-600 字):**
分组列出 8-12 个方法:

**Attention 优化:**
- PagedAttention(vLLM)— KV cache 按 block 分页,消除碎片,提升 batch size 2-4×
- FlashAttention-2/3 — fused softmax + tile,memory I/O 降到 O(N)
- Speculative decoding — 用小模型 draft 数 token,大模型一次 verify,2-4× 速度

**Batching 优化:**
- Continuous / dynamic batching(vLLM iteration-level scheduling)— 请求异步加入,GPU 永不空闲
- Disaggregated serving(DistServe)— prefill 与 decode 拆机器,分别用 compute / memory-optimized 配置
- Multi-LoRA serving — 共享 base + 多 LoRA adapter 切换,单 GPU 服务多模型

**量化优化:**
- INT4 / INT8 weight-only(GPTQ / AWQ)— 模型权重压缩,decode 阶段 HBM read 减半
- SmoothQuant 激活量化 — INT8 权重 + INT8 激活,prefill TC 也加速
- FP8 inference(Hopper TC FP8)— 训练 / 推理同精度路径,2× decode 吞吐
- KV cache 量化(INT8/FP8 KV)— KV cache 显存减半,batch 更大

**调度优化:**
- CUDA Graph decode capture — 固定 shape 重放,launch 开销几乎为 0
- Tensor parallel inference(NCCL allreduce after attention/FFN)— 大模型横向切,跨 GPU 协同
- Prefix caching(共享 prompt 前缀复用 KV)— 多用户同 system prompt 时 prefill 几乎免费

**§8 延伸阅读(代码块,不计字数):**
- vLLM(`github.com/vllm-project/vllm`)+ PagedAttention paper(`arxiv.org/abs/2309.06180`)
- FlashAttention-3 paper
- TensorRT-LLM(`github.com/NVIDIA/TensorRT-LLM`)
- SGLang(`github.com/sgl-project/sglang`)
- GPTQ paper(`arxiv.org/abs/2210.17323`)、AWQ paper(`arxiv.org/abs/2306.00978`)、SmoothQuant paper(`arxiv.org/abs/2211.10438`)
- DistServe paper(`arxiv.org/abs/2401.09670`)

- [ ] **Step 2: 本地验证**(同 Task 1 Step 2,文件名换 `24-inference-end-to-end.md`)

- [ ] **Step 3: Commit**

```bash
git add docs/cuda-zh/24-inference-end-to-end.md
git commit -m "docs(cuda-zh): 24 模型推理全栈串联 — capstone 章"
```

---

## Task 3: 索引更新 + 全集验证 + tag + push

**Files:**
- Modify: `docs/cuda-zh/00-index.md`(§1 加第三条阅读路径,§8 表追加 2 行)

- [ ] **Step 1: 更新 §1 阅读路径**

打开 `docs/cuda-zh/00-index.md`,定位 §1 节末尾原本的两条阅读路径。在它们后面追加第三条:

```markdown
**3. 训练 / 推理实战路径(读完任何基础后)**

熟悉硬件层级 / 软件抽象层任一路径后,直接读 [23 模型训练全栈串联](23-training-end-to-end.md) 与 [24 模型推理全栈串联](24-inference-end-to-end.md) 看一次 step 如何调度前 22 章的全部组件,以及训练 / 推理两侧的优化方法体系。
```

(若 §1 目前没有"两条阅读路径"小节,在 §1 末尾或 §8 开头插入这个第三条即可,不破坏现有结构。)

- [ ] **Step 2: 更新 §8 章节索引表**

定位 `00-index.md` §8 节里"### 本教程章节索引"下的表格。在表格末尾(`[22]` 行之后)追加 2 行:

```markdown
| [23](23-training-end-to-end.md) | 模型训练全栈串联 | training step 端到端 + 优化方法体系 |
| [24](24-inference-end-to-end.md) | 模型推理全栈串联 | prefill/decode 端到端 + 优化方法体系 |
```

- [ ] **Step 3: 全集验证**

```bash
cd docs/cuda-zh
echo "=== 文件清单(应 25)==="
ls -1 *.md | wc -l

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
    if grep -qi 'gpusim' "$f"; then gpusim="GPUSIM_FOUND"; else gpusim="ok"; fi
    printf "%-50s sections=%d mermaid=%d zh_chars=%d %s\n" "$f" "$sections" "$mermaid" "$zh" "$gpusim"
done

echo ""
echo "=== 总 mermaid 计数(应 ≥ 26)==="
grep -c '^```mermaid' *.md | awk -F: '{s+=$2} END {print s}'
```

预期输出:
- 25 个 md 文件
- 每个文件 sections=8、mermaid ≥ 1、zh_chars 在 1500-2500
- 全部 gpusim=ok
- 总 mermaid ≥ 26

如果验证失败,先修复对应章节再继续。

- [ ] **Step 4: Commit + tag**

```bash
git add docs/cuda-zh/00-index.md
git commit -m "docs(cuda-zh): 00 索引补 23/24 链接 + 第三条阅读路径"
git tag cuda-zh-v2-complete
```

- [ ] **Step 5: Push 到 GitHub**

```bash
git push origin master
git push origin cuda-zh-v2-complete
```

预期输出包含 `master -> master` 与 `[new tag] cuda-zh-v2-complete -> cuda-zh-v2-complete`。

---

## 验收准则

- [ ] `docs/cuda-zh/23-training-end-to-end.md` 与 `24-inference-end-to-end.md` 都存在,8 节齐全,1500-2500 字,≥ 1 mermaid,无 gpusim
- [ ] `docs/cuda-zh/00-index.md` §8 表含 23/24 链接;§1(或 §8 开头)有第三条「训练 / 推理实战路径」
- [ ] 全集 25 个 md;总 mermaid ≥ 26
- [ ] tag `cuda-zh-v2-complete` 到位且已推送
- [ ] master 已推送到 origin
