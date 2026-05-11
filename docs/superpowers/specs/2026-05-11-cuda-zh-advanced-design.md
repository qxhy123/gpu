# cuda-zh Advanced — 10 章进阶补充集 design

> **Status:** Brainstormed 2026-05-11. Implementation TBD.

## 1. Goals

为现有 25 章 cuda-zh 教程补充 senior AI Infra engineer 真实工作中常卡壳但教程未覆盖的 8 个主题(A 集),并扩展到下一代硬件 + 部署形态(C 集 2 章)。共 **10 个新章节** 入 `docs/cuda-zh/advanced/`。

## 2. Non-goals

- 不改动现有 25 章
- 不写英文版
- 不覆盖非 NVIDIA 生态(AMD MI、Intel Xe、TPU)
- 不覆盖 OpenCL / SYCL / HIP
- 沿用 cuda-zh 既定规范:零 gpusim、8 节结构、Mermaid 强制、UTF-8 LF

## 3. 10 个章节定位

### A 集(8 章 — 真缺的 senior 主题)

| # | 文件名 | 主题 |
|---|---|---|
| a01 | `a01-moe-expert-parallelism.md` | MoE 模型(Mixtral / DeepSeek-V3)训练 + 推理:expert routing、all-to-all、capacity factor、DeepEP / Megablocks / Tutel |
| a02 | `a02-cutlass-3x-and-cute.md` | CUTLASS 3.x 架构 + CuTe Layout 代数:collective mainloop、Layout / Tensor / TiledMMA、sm90_collective_*、为什么 CUTLASS 3.x 完全重写 |
| a03 | `a03-quantization-algorithms.md` | 量化算法原理:GPTQ Hessian-based rounding、AWQ activation outlier、SmoothQuant per-channel migration、FP8 delayed vs current scaling、QAT for LLM |
| a04 | `a04-triton-kernel-engineering.md` | Triton 工程化:ttir → ttgir → llir → PTX pipeline、autotune key 策略、torch.compile inductor 生成 Triton 模式、何时 Triton 何时 CUTLASS 何时 PTX |
| a05 | `a05-rdma-nccl-transport.md` | 跨节点通信:InfiniBand NDR 400G、GPUDirect RDMA 路径、NCCL transport 选择(NVLink/IB/GDR/TCP)、`NCCL_NET_PLUGIN`、rail-optimized topology |
| a06 | `a06-fault-tolerance-and-sdc.md` | 训练可靠性:GPU AFR、SDC(silent data corruption)在 1k+ GPU 训练的概率、`torch.distributed.checkpoint` DCP async、loss spike / NaN debug、Pathways failover |
| a07 | `a07-data-pipeline-engineering.md` | 数据流水线:DataLoader CPU 瓶颈、DALI/FFCV/Ray Data、GPU-side decoding、prefetch + pinned memory、tokenize 在线 vs 离线 |
| a08 | `a08-cudnn-cublas-advanced.md` | cuDNN / cuBLAS / cuBLASLt 高级:algorithm heuristic、`cublasLtMatmulAlgoGetHeuristic`、cuDNN backend API graph、何时绕开调 Triton/CUTLASS |

### C 集(2 章 — 下一代 + 部署形态)

| # | 文件名 | 主题 |
|---|---|---|
| a09 | `a09-blackwell-b200-gb200.md` | Blackwell SM_100:B100 / B200 / GB200 NVL72,2nd gen TE,5th gen NVLink 1.8 TB/s,HBM3e 192 GB,FP4 推理,decompression engine |
| a10 | `a10-mig-confidential-vgpu.md` | 多租户部署:MIG 7 instance、confidential compute(SEV-SNP)、vGPU、容器化 GPU(NVIDIA Container Toolkit) |

## 4. 每章统一结构(沿用既定规范)

```
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
- 每章 **4000-5000 中文字**(同主体教程深度)
- 每章 Mermaid ≥ 2(架构 / 流程类章节如 a01 MoE、a02 CUTLASS、a05 RDMA、a09 Blackwell 推荐 ≥ 3)
- 总 Mermaid 预期 ≥ 22

### 五类必加内容(每章)
1. **微架构 / 协议机制** — 硬件 / 软件协议层面具体怎么工作
2. **真实生产数字** — H100 SXM5 / B200 / 大规模训练实测,而不是 spec 上限
3. **失败模式 + 调试** — production 真踩过的坑
4. **实现导读** — 真实开源源码位置(vLLM、CUTLASS、DeepEP、Triton、Megatron 等)
5. **设计权衡** — 为什么 NVIDIA / 框架选 A 不选 B

## 5. 索引更新

`docs/cuda-zh/00-index.md` §8 节末尾追加 "进阶专题(`advanced/`)" 表格:

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

## 6. 里程碑

| Milestone | Scope | Tag |
|---|---|---|
| **AG1** LLM kernel 系列 | a01 MoE + a02 CUTLASS+CuTe + a03 量化 + a04 Triton | `cuda-zh-adv-AG1-complete` |
| **AG2** 基础设施 + 框架 | a05 RDMA/NCCL + a06 SDC + a07 数据流水线 + a08 cuDNN/cuBLAS | `cuda-zh-adv-AG2-complete` |
| **AG3** 下一代 + 部署 | a09 Blackwell + a10 MIG | `cuda-zh-adv-AG3-complete` |
| **AG4** 索引 + 全集验证 + push | 更新 00-index + 验证 35 文件 + push | `cuda-zh-adv-complete` |

每个 AG 单 subagent 一次 dispatch 内部串行写。

## 7. 文件清单

### 新增
```
docs/cuda-zh/advanced/a01-moe-expert-parallelism.md
docs/cuda-zh/advanced/a02-cutlass-3x-and-cute.md
docs/cuda-zh/advanced/a03-quantization-algorithms.md
docs/cuda-zh/advanced/a04-triton-kernel-engineering.md
docs/cuda-zh/advanced/a05-rdma-nccl-transport.md
docs/cuda-zh/advanced/a06-fault-tolerance-and-sdc.md
docs/cuda-zh/advanced/a07-data-pipeline-engineering.md
docs/cuda-zh/advanced/a08-cudnn-cublas-advanced.md
docs/cuda-zh/advanced/a09-blackwell-b200-gb200.md
docs/cuda-zh/advanced/a10-mig-confidential-vgpu.md
```

### 修改
```
docs/cuda-zh/00-index.md     # §8 末尾追加"进阶专题"表格
```

## 8. 验收准则

- [ ] 10 个 advanced 章节全部存在,8 节齐全,4000-5000 字,Mermaid ≥ 2,零 gpusim
- [ ] `00-index.md` §8 包含 10 章 advanced 表格
- [ ] cuda-zh 全集:25 (现有) + 10 (advanced) = **35 个 markdown**
- [ ] 全集 Mermaid 总数从 51 升至 **≥ 73**(原 51 + 新 ≥ 22)
- [ ] 4 个 milestone tag 全到位:`cuda-zh-adv-AG1-complete` ... `cuda-zh-adv-AG3-complete` + `cuda-zh-adv-complete`
- [ ] master + 新 tag 推送到 origin

## 9. 章节内容要点(per-chapter 提示)

### a01 MoE + Expert Parallelism
- §2: Mixtral 8×7B / DeepSeek-V3 671B(37B 激活)架构;**Mermaid `flowchart TB`** 画 MoE block(input → router/gate → top-k expert select → expert FFN × N → weighted sum);capacity factor 公式;EP(expert parallelism)与 TP/DP 的正交;all-to-all 通信模式
- §3: Megatron `--num-experts 8 --moe-expert-model-parallel-size 4`;DeepSpeed-MoE;DeepEP github、Megablocks(scatter-gather kernel)、Tutel(adaptive routing);PyTorch MoE 自实现
- §4: DeepSeek-V3 671B 在 H100×2048 训练 MFU ~50%;all-to-all 通信占 step time 30-40%;capacity factor 1.25 时 token drop rate < 1%
- §5: Megatron-MoE config + DeepEP all-to-all kernel 调用片段;router top-k + load-balancing aux loss 代码
- §7: capacity 设太小(token drop)、太大(显存爆);忘记 aux loss 让 routing 退化;EP 数与 expert 数不能整除;all-to-all 没 overlap compute
- §8: Mixtral paper、Switch Transformer paper、GShard paper、DeepSeek-V3 paper、DeepEP github、Megablocks paper、Tutel paper

### a02 CUTLASS 3.x + CuTe
- §2: **Mermaid `classDiagram`** 画 CUTLASS 3.x 层级:`gemm::kernel::Sm90*` → `gemm::collective::CollectiveMainloop` + `CollectiveEpilogue`;CuTe 三个核心抽象 Layout、Tensor、TiledMMA;Layout 代数(composition / inverse / partition / coalesce)
- §3: CuTe `Layout<Shape, Stride>`、`Tensor<Engine, Layout>`、`make_tensor / partition_S / partition_D`;`SM90_TMA_LOAD` copy op;collective mainloop `KernelTraits` template
- §4: CUTLASS 3.x 在 H100 实测 GEMM 87-92% TC peak;FlashAttention-3 基于 CUTLASS 3.x sm90_collective;CUTLASS 2.x vs 3.x:Hopper async 必须 3.x
- §5: 一段 `cute::Layout<Shape<_64,_128,_16>, Stride<_128,_1,_8192>>` 实例;CollectiveMainloop 例子 skeleton
- §7: CuTe Layout 误用 stride(变 row-major 误算)、忘记 swizzle 与 wgmma 对齐、CUTLASS 2.x API 写在 3.x 没法用 wgmma
- §8: CUTLASS github(`include/cutlass/gemm/collective/sm90_*.hpp`)、CuTe paper、Hopper GEMM CUTLASS examples

### a03 量化算法原理
- §2: **Mermaid `flowchart LR`** 画三种量化数据流:GPTQ(Hessian-based optimal rounding)、AWQ(activation-aware salient weight)、SmoothQuant(per-channel migration);per-tensor / per-channel / per-group scale 对比;**Mermaid `stateDiagram-v2`** FP8 delayed scaling 状态机
- §3: TensorRT-LLM `quantization_modes = [W4A16_AWQ, W8A8_SQ, FP8]`;`torch.ao.quantization`;Transformer Engine `Float8Tensor`、`DelayedScaling`、`HYBRID` format;AutoGPTQ / AutoAWQ python API
- §4: Llama-3-70B GPTQ INT4 MMLU 79.1 vs FP16 79.5(0.4 pt 损失);AWQ INT4 与 GPTQ 相当但保留 1% salient weight FP16;SmoothQuant W8A8 比 FP8 慢 10-15%(无原生 TC);FP8 delayed scaling 与 current scaling 在 overflow 处理差
- §5: GPTQ 算法伪代码(逐 col 选最优量化 + Hessian inverse 误差传播);TE `Float8Tensor` 配 `DelayedScaling` 代码;AWQ 1% salient mask 代码
- §7: 校准集太小导致 outlier 没覆盖(精度爆);per-tensor scale 在长尾 activation 上崩溃(必须 per-channel);FP8 underflow forgot DelayedScaling(loss NaN);量化后没重测下游任务
- §8: GPTQ paper、AWQ paper、SmoothQuant paper、FP8 Training paper(NVIDIA)、TensorRT-LLM quantization docs

### a04 Triton 工程化
- §2: **Mermaid `flowchart LR`** Triton compiler pipeline:`.py @triton.jit` → AST → ttir(triton IR)→ ttgir(triton GPU IR) → llir(LLVM IR with PTX intrinsic) → PTX → SASS;**Mermaid `classDiagram`** Triton runtime kernel cache + autotune key
- §3: `@triton.jit`、`@triton.autotune(configs=[...], key=['M', 'N', 'K'])`;`tl.program_id / tl.load / tl.store / tl.dot / tl.zeros`;`triton.language as tl`;`torch.compile` 自动生成 Triton 内核(inductor)
- §4: Triton vs CUTLASS H100 GEMM:Triton 80%、CUTLASS 90%;autotune key 选错(忘记 dtype)导致每个 dtype 重 tune ~30s;`torch.compile` 生成 Triton 在 PyTorch 2.x 已经是默认
- §5: Triton 完整 GEMM kernel(M/N/K + autotune configs);FlashAttention 2 Triton 实现导读(`flash_attn_triton.py`);PyTorch inductor 生成的 Triton 实例
- §7: autotune key 漏 dtype(每 dtype 重 tune);Triton 不支持的指令(wgmma 部分需 inline asm);共享 SMEM 大小硬编码超 SMEM cap 静默 fail;forgot `BLOCK_SIZE` 是编译期常数
- §8: Triton github(`OpenAI/triton`)、Triton paper、FlashAttention-2 Triton 实现、PyTorch inductor docs

### a05 RDMA + NCCL transport
- §2: **Mermaid `flowchart LR`** 画跨节点通信完整路径:GPU → GPU memory → NIC(RDMA verbs)→ IB switch → 对端 NIC → 对端 GPU;**Mermaid `flowchart TB`** rail-optimized topology(8 GPU 节点 × 8 IB rail × spine switch);GPUDirect RDMA bypass CPU memory copy
- §3: NCCL transport 选择:`NCCL_NET_PLUGIN`、`NCCL_IB_DISABLE`、`NCCL_NET=Socket/IB`、`NCCL_TOPO_FILE`;OpenMPI / UCX 集成;Mellanox HCA OFED 安装;`ibstat` / `ibv_devinfo` 检查
- §4: NDR 400G IB 单 link 50 GB/s 双向;rail-optimized 8-rail 单节点 400 GB/s 跨节点;GPUDirect RDMA 让 P2P 接近 NVLink 速率;NCCL allreduce 跨节点 256 GPU bf16 4 GB ~10 ms
- §5: `NCCL_DEBUG=INFO NCCL_NET=IB NCCL_IB_HCA=mlx5_0:1 ...` 完整环境变量;Mellanox NIC PCIe 拓扑配置(同 NUMA 节点 GPU + NIC);NCCL backend 探测代码
- §7: rail 不对齐(同 PCIe switch 下多 GPU 共享 1 rail bandwidth halved);IB Q-counters 没监控 → 静默 link error;NCCL fallback 到 TCP 但用户没发现(打 INFO log 才看到);跨集群训练 cross-DC latency 在 sync collective 上灾难
- §8: NCCL User Guide transport 章、NVIDIA GPUDirect RDMA docs、Mellanox OFED docs、Pathways paper(Google 大规模训练 cross-host)

### a06 训练可靠性 + SDC
- §2: **Mermaid `sequenceDiagram`** GPU 故障 → MEM ECC → CUDA error → process abort → Slurm restart → checkpoint load 完整路径;**Mermaid `flowchart TD`** SDC 检测策略(replicated check / hash verify / activation outlier scan)
- §3: `torch.distributed.checkpoint` (DCP) sharded state dict、`async_save`;`torch.cuda.cudart()` ECC error query;Megatron checkpoint hierarchical;Pathways failover 模式
- §4: H100 SXM5 AFR ~2-3% / year;1k GPU 训练 ~30 GPU 失败 / 年;SDC 在 1k+ GPU 出现 ~1 次 / 周(Llama 3、Gemini 论文);DCP async checkpoint ~5s for 70B vs sync 2 min
- §5: DCP async checkpoint 代码;loss spike 检测 + 自动 revert 到上个 checkpoint;NaN propagation 跨 rank 同步检测(`torch.distributed.all_reduce` 个 flag)
- §7: 同步 checkpoint 阻塞训练(必须 async);单点 checkpoint 不分片(load 10× 慢);忽略 SDC 信号(loss spike 当随机抖动);硬件故障后立即 restart 而不 cooldown(同卡反复 fail)
- §8: Llama 3 405B training report、Gemini paper、Megatron DCP 文档、Pathways paper、Slurm + NCCL 实战

### a07 数据流水线工程化
- §2: **Mermaid `sequenceDiagram`** 训练数据从 disk → tokenize → shuffle → batch → pinned memory → H2D → GPU 完整路径,标出每段瓶颈;**Mermaid `flowchart LR`** PyTorch DataLoader worker 进程 + pin_memory 线程模型
- §3: PyTorch `DataLoader(num_workers, pin_memory, prefetch_factor, persistent_workers)`;DALI GPU-side decode + augment;FFCV;Ray Data 分布式;NVIDIA Tokenizer GPU 实现
- §4: CPU DataLoader 在 batch_size 1024 image 上单卡瓶颈 ~20%;DALI GPU-side decode 释放 CPU,H100 训练加速 8-15%;Ray Data 在多节点 scale 线性
- §5: PyTorch DataLoader + pin_memory + prefetch 模板;DALI image pipeline(`fn.readers.file` + `fn.decoders.image` + `fn.crop_mirror_normalize`);Ray Data 分布式预处理
- §7: `num_workers` 设错(0 主线程串行 / 太多 contention);忘 `pin_memory=True`(H2D 慢 3-4×);persistent_workers=False 每 epoch 重启进程;tokenize 在 DataLoader 里 CPU-bound
- §8: PyTorch DataLoader docs、DALI docs、FFCV paper、Ray Data docs、HuggingFace datasets streaming

### a08 cuDNN + cuBLAS + cuBLASLt 高级
- §2: **Mermaid `classDiagram`** cuBLAS / cuBLASLt / cuDNN 关系:cuBLAS legacy 单算子 → cuBLASLt fused matmul + epilogue(bias、ReLU、GELU 融合)→ cuDNN backend API graph(任意 op 融合);**Mermaid `flowchart LR`** algorithm heuristic 选择路径
- §3: `cublasLtMatmulAlgoGetHeuristic` API + `cublasLtMatmulDesc_t` + `cublasLtMatmulPreference_t`;`cudnnBackendCreateDescriptor / Finalize` cuDNN backend graph API;何时手写 Triton / CUTLASS(cuDNN 没覆盖时)
- §4: cuBLASLt vs cuBLAS legacy: FP16 matmul + bias 融合让 70B forward 加速 10-15%;cuDNN backend API 在 attention forward 与 FlashAttention 持平 ±5%;algorithm heuristic 内部用 ML 模型选(cuDNN 8.6+ predicate-based)
- §5: cuBLASLt matmul + bias + GELU epilogue 完整 C++ 调用;cuDNN backend graph 构造 attention forward;cuBLAS workspace 配 ~32 MiB
- §7: 忘记 cuBLAS workspace(每次 alloc 慢);cublasLt heuristic 没 set preference(选差 algo);cuDNN backend graph 没 finalize 直接 execute(undefined);新 dtype(FP8 / BF16)走 legacy API(没 fused epilogue)
- §8: cuBLAS docs、cuBLASLt heuristics docs、cuDNN backend API docs、Transformer Engine 源码(用 cuBLASLt + cuDNN)

### a09 Blackwell B200 / GB200 NVL72
- §2: **Mermaid `flowchart TB`** Blackwell SM_100 全景:2 die 4 NVLink-C2C / die、208 SM、192 GB HBM3e、8 TB/s、5th gen NVLink 1.8 TB/s/GPU;**Mermaid `flowchart LR`** GB200 NVL72 拓扑:36 Grace CPU + 72 Blackwell GPU,全 NVLink 互连;**Mermaid 第 3 个 `classDiagram`** TE 2nd gen + FP4 数据通路
- §3: CUDA 12.4+ 支持 SM_100;`-arch=sm_100`;Transformer Engine v1.10+ 自动 FP8/FP4 选择;`cudaMallocAsync` 在 GB200 上支持 cross-Grace memory;`cuda::std::memcpy_async` 接 NVLink-C2C
- §4: B200 FP8 TC peak ~10 PFLOPS(SXM5);FP4 TC peak ~20 PFLOPS(下一代 2:4 sparsity 40 PFLOPS);HBM3e 8 TB/s vs Hopper 5 TB/s;NVL72 系统 ~720 PFLOPS FP8;Grace CPU + Blackwell GPU 通过 NVLink-C2C 900 GB/s 双向
- §5: Blackwell FP4 inference 代码片段(TE v1.10+);GB200 cross-CPU-GPU memory 用 NVLink-C2C 例子;decompression engine API(读 LZ4/Snappy 压缩 GMEM)
- §7: FP4 直接训练(应推理 / QAT);SM_100 写 SM_90a 代码(部分 wgmma 兼容但 mma_async size 变);GB200 单 GPU 当作普通 H100 配置(漏 NVLink-C2C cross-CPU)
- §8: Blackwell Whitepaper、GB200 NVL72 reference architecture、Transformer Engine v1.10+ release notes、NVLink-C2C 文档

### a10 MIG + confidential + vGPU
- §2: **Mermaid `flowchart TB`** MIG 切分模式:H100 SXM5 max 7 instances(每 instance 含 SM × 16 + L2 + HBM partition);**Mermaid `sequenceDiagram`** confidential compute(SEV-SNP)kernel 加密路径;**Mermaid `flowchart LR`** vGPU + grid driver + container topology
- §3: MIG: `nvidia-smi mig -cgi 9,9,9,9,9 -C`(创建 5 个 1g.10gb instance);`CUDA_VISIBLE_DEVICES=MIG-...`;confidential compute: `NVIDIA_REQUIRE_CC=on` + AMD SEV-SNP host;vGPU: NVIDIA grid driver + VMware/KVM;NVIDIA Container Toolkit(`--gpus all`、`--gpus 'device=0,1'`)
- §4: MIG 7-instance 总吞吐 ~90% 单卡(隔离开销 10%);confidential compute 性能开销 5-15%(数据加密路径);vGPU H100 max 8 user / GPU
- §5: MIG 切分 + container 部署完整命令;confidential GPU 拉起验证;Kubernetes NVIDIA Device Plugin + MIG 配置
- §7: MIG 配置后 NCCL 跨 instance 报错(MIG 不支持 P2P);confidential compute kernel 用了 CPU 共享内存(打破隔离);vGPU 漏 license(driver fallback 单卡 6 小时)
- §8: NVIDIA MIG User Guide、Confidential Compute Whitepaper、NVIDIA vGPU 文档、NVIDIA Container Toolkit github
