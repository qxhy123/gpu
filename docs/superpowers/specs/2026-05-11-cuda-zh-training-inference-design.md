# CUDA-zh v2 — 训练 + 推理全栈串联章节

> **Status:** Brainstormed 2026-05-11. Implementation TBD.

## 1. Goals

为 `docs/cuda-zh/` 添加 2 章 capstone:
- **23 模型训练全栈串联** — 一次 LLM training step 如何端到端调度前 22 章覆盖的所有 GPU 组件,以及训练侧优化方法体系
- **24 模型推理全栈串联** — 一次 LLM serving step(prefill + decode)如何调度组件,以及推理侧优化方法体系

更新 `00-index.md` 增加这 2 章入口 + 第三条阅读路径(训练 / 推理实战)。

## 2. Non-goals

- 不引入新组件章节(只串联现有 22 章)
- 不写 PyTorch / vLLM 实现源码导读
- 不覆盖训练框架对比(只举主流框架名 + 入口 API)
- 不覆盖非 LLM 模型(扩散、推荐系统等)
- 沿用 cuda-zh 既定规范:零 gpusim、Mermaid 强制、1500-2500 字/章

## 3. 章节结构

两章遵循同 cuda-zh 系列的 8 节结构:
```
# NN · <标题>
## 1. 是什么 / 为什么有它
## 2. 硬件视角(微架构细节)
## 3. CUDA 编程接口
## 4. 关键性能指标
## 5. 代码示例
## 6. 实测手段
## 7. 常见反模式 ← 这里改写为「优化方法」(本章特色)
## 8. 延伸阅读
```

**§7 改写为「训练侧优化方法」/「推理侧优化方法」**:列出 8-12 个优化技术,每个一句话说明 + 命中哪些组件 + 何时用。原"反模式"内容并入 §4 性能指标节末尾的"反模式提示"小段。

## 4. 内容骨架

### 23 模型训练全栈串联

- **§1**: 训练 step 是 forward + backward + optimizer + comm 的混合工作负载;不同阶段触发不同组件主导;一次 step 涉及前 22 章几乎所有内容。
- **§2**: Mermaid `flowchart TB` 画一次 step 的组件触发链:HBM (weights/optim/grad) → L2 → SMEM tile → TC/wgmma (mma) → TMA (load) → mbarrier (sync) → cluster (cooperative GEMM) → CTA scheduler → Stream (compute / NCCL 重叠) → NCCL allreduce → optimizer kernel → mempool 复用 activation。
- **§3**: PyTorch DDP / FSDP / DeepSpeed / Megatron 入口;`torch.cuda.graph()`(capture training step);`torch.distributed.all_reduce`;`bf16/fp16 autocast`。
- **§4**: 关键指标 — model FLOPs utilization (MFU)、HBM bandwidth utilization (HBW)、scaling efficiency (vs ideal)、NCCL bus bandwidth、tokens/sec/GPU。反模式提示:无 graph capture、grad sync 阻塞下一 forward、mismatch fp32 master / bf16 compute。
- **§5**: 一个标注的 training step 伪代码 — 每行右侧注释命中的组件(HBM read / TC FMA / NCCL allreduce / mempool free)。
- **§6**: NSight Systems 看 NCCL/compute 重叠;NSight Compute 算 MFU = `flops_executed / (time × peak_flops)`;PyTorch profiler。
- **§7**: 训练侧优化方法体系(8-12 项):算子融合(FlashAttention-3、Apex fused layernorm)、混精训练(bf16/fp16 + fp32 master)、Activation checkpointing、Gradient accumulation、ZeRO-1/2/3 与 FSDP 分片、Tensor parallelism(Megatron)、Pipeline parallelism(GPipe/1F1B)、Sequence parallelism、Compute-comm 重叠(reduce-scatter overlapped with backward)、CUDA Graph training step capture、Selective recompute、FP8 training(Hopper TC)。
- **§8**: PyTorch 文档、Megatron-LM github、DeepSpeed docs、FlashAttention paper、ZeRO paper、Hopper FP8 training guide。

### 24 模型推理全栈串联

- **§1**: 推理是 prefill (compute-bound, GEMM 主导) + decode (memory-bound, GEMV + KV cache HBM read 主导) 两阶段;LLM serving 大多数时间在 decode;组件触发模式与训练显著不同。
- **§2**: Mermaid `flowchart TB` 画 prefill vs decode 两路径 — prefill 走 TC + wgmma + TMA(GEMM peak);decode 走 HBM read + 小型 GEMV + KV cache pagelist + flash-attention kernel(memory-bound);output 走 TC。说明 batch=1 vs continuous batching 的不同。
- **§3**: vLLM / TensorRT-LLM / TGI / SGLang / DeepSpeed-MII 入口;`torch.cuda.graph()`(capture decode step,固定 shape);PagedAttention 内核;`cudaMallocAsync` 用于 KV cache slab 池。
- **§4**: 关键指标 — first-token latency (FTL)、time-per-output-token (TPOT)、throughput (tokens/sec/GPU)、KV cache bytes、HBM 利用率。反模式提示:静态 batch、KV cache 不分页、没用 flash-attention、INT8 量化但漏激活量化。
- **§5**: 一个标注的 decode step 伪代码 — KV cache append (HBM write)、attention(flash kernel,SMEM)、FFN(TC GEMM)、sample(host)。
- **§6**: NSight Systems 看 prefill/decode 时间分布;TensorRT-LLM benchmark 工具;`nvidia-smi` 看 KV cache HBM 占用;NSight Compute 看 attention kernel SMEM/flash 利用。
- **§7**: 推理侧优化方法体系(8-12 项):PagedAttention(vLLM)、FlashAttention-2/3(memory I/O 优化)、Continuous / dynamic batching、Speculative decoding(draft + verify)、INT4/INT8 weight-only quantization(GPTQ/AWQ)、SmoothQuant 激活量化、FP8 inference(Hopper TC FP8)、CUDA Graph decode capture、KV cache quantization(INT8/FP8)、Multi-LoRA serving、Tensor parallel inference(NCCL allreduce after each block)、prefill/decode 拆机器(disaggregated serving,DistServe)。
- **§8**: vLLM paper、FlashAttention-3 paper、GPTQ/AWQ paper、SmoothQuant paper、TensorRT-LLM github、SGLang github、DistServe paper。

## 5. 索引更新

`docs/cuda-zh/00-index.md` §8 章节表追加 2 行:
- `[23](23-training-end-to-end.md) | 模型训练全栈串联 | training step + 优化方法`
- `[24](24-inference-end-to-end.md) | 模型推理全栈串联 | prefill/decode + 优化方法`

§1 末尾增加第三条阅读路径:
- 「训练 / 推理实战路径」: 00 → 01-22 选择性补漏 → 23 训练全栈 → 24 推理全栈

## 6. Mermaid 要求

每章 ≥ 1 个 Mermaid(§2 硬件视角必须有);推荐 §7 优化方法节再用一个 `flowchart LR` 或 `mindmap` 展示优化方法分类(可选)。

总 Mermaid 计数:全集应从 24 升至 ≥ 26。

## 7. 文件清单

### 新增
```
docs/cuda-zh/23-training-end-to-end.md
docs/cuda-zh/24-inference-end-to-end.md
```

### 修改
```
docs/cuda-zh/00-index.md      # §1 第三条阅读路径 + §8 表格 +2 行
```

## 8. 里程碑

| Milestone | Scope | Tag |
|---|---|---|
| **M1** 23+24 写完 + index 更新 + push | 2 新章 + 索引补全 + 全集验证 + 推送到 origin | `cuda-zh-v2-complete` |

单 subagent 一次完成。

## 9. 验收准则

- [ ] `docs/cuda-zh/23-training-end-to-end.md` 存在、8 节齐全、1500-2500 中文字、≥ 1 Mermaid、零 gpusim
- [ ] `docs/cuda-zh/24-inference-end-to-end.md` 同上
- [ ] `docs/cuda-zh/00-index.md` §8 表格含 23+24 链接
- [ ] `00-index.md` §1 提供 3 条阅读路径(原 2 条 + 训练/推理实战)
- [ ] 全集 25 个 md 文件;总 mermaid ≥ 26
- [ ] tag `cuda-zh-v2-complete` 到位
- [ ] master + 新 tag 推送到 `origin`(github qxhy123/gpu)
