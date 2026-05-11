# CUDA-zh Deep Expansion — 25 章扩深 design

> **Status:** Brainstormed 2026-05-11. Implementation TBD.

## 1. Goals

把现有 25 章 cuda-zh 教程从"smart graduate student 入门"深度,提升到"senior AI Infra engineer 参考手册"深度:
- 每章字数从 1500-2500 扩到 **4000-5000 中文字**
- 每章 Mermaid 数从 ≥ 1 提升到 **≥ 2**
- 全集 Mermaid 总数从 26 提升到 **≥ 50**
- 内容深度从"是什么 + 主流 API"提升到"微架构机制 + 生产数字 + 失败模式 + 实现导读"

## 2. Non-goals

- 不新增章节(沿用现有 25 章)
- 不改 8 节结构 + 既定规范(零 gpusim、UTF-8 LF、Mermaid 必有)
- 不替换现有内容,而是**在已有内容基础上加深**(保持读者两条阅读路径不破坏)
- 不写英文版

## 3. 深度提升原则(每章必须包含)

每章扩展后必须新增以下五类内容(可分散到 §2 / §4 / §5 / §7):

### 3.1 微架构机制级细节
不只说"用了 X",而是"X 的硬件怎么工作":
- 寄存器堆 bank 冲突避免、scoreboard 标志位结构
- WGMMA descriptor bit field(swizzle、leading dim、stride)
- TMA box descriptor 编码格式 + 5D 翻译路径
- mbarrier 64-bit 内部布局(phase / arrived / expected / pending tx)
- L2 set-aside 实际是 way-bias 还是 set-bias
- HBM3 row buffer 调度策略(bank-group rotation)

### 3.2 真实生产数字 + 案例
H100 SXM5 实测 / 论文实测,而不是 spec 上限:
- 70B / 405B 模型 TP×PP×DP 不同配置下的 MFU 实测分解
- bf16 vs FP8 训练在不同 batch 下的真实加速比
- vLLM PagedAttention 在 prefix-cache 命中率 90% 时的 throughput 提升
- NCCL ring vs tree allreduce 在不同 message size 的拐点
- TMA 跨 cluster 与单 SM 的延迟实测差(论文数据)

### 3.3 失败模式 + 调试手段
production 里真踩过的坑:
- NCCL deadlock(rank order mismatch、跨 group 数据)的诊断方法
- PagedAttention block table 与 KV-cache slab GC 的 race
- FP8 training loss scaling overflow 的工程处理
- HBM ECC double-bit error 处理
- CUDA Graph capture 期间误调 cudaMalloc 的诊断
- mbarrier expect_tx 字节数算错的死锁 + 检测

### 3.4 实现导读 / 当前前沿
指向 CUTLASS / vLLM / TensorRT-LLM / FlashAttention 真实源码位置:
- CUTLASS 3.x persistent kernel(`include/cutlass/gemm/kernel/sm90_*`)的 producer-consumer warp 拆分
- ThunderKittens tile primitives 的 SMEM 抽象
- vLLM PagedAttention(`csrc/attention/`)block table 索引
- TensorRT-LLM IFB(`cpp/tensorrt_llm/runtime/`)kernel 拼接
- FlashAttention-3 的 warp-specialization producer-consumer 模式
- DistServe / Sarathi-Serve 调度差异

### 3.5 替代方案 / 设计权衡
为什么 NVIDIA 选 A 不选 B:
- 为什么 wgmma 用 warp-group 不用 thread-block
- 为什么 cluster 上限 16 不是 32
- 为什么 mbarrier 单 phase bit 不是 multi-bit counter
- 为什么 cudaMallocAsync 不默认 split block
- 为什么 CUDA Graph 不支持 conditional 直到 12.4

## 4. 各章扩展重点(per-chapter 提示)

### G1 — 基础

- **00 索引**: 加"按 senior gap 阅读路径"(第 4 条);加 H100 SXM5 vs PCIe 关键差异表;Mermaid 增 1 个 GPC × 9 / SM × 132 拓扑图
- **01 SIMT**: 加 ITS 在 Volta+ 的 PC + RPC + 收敛栈实现;Hopper warp scheduler 的 4-issue scoreboard 机制;divergent loop 的最坏情形量化;`__shfl_sync` 实测 5-cycle 延迟
- **02 SM 内部**: 加 sub-partition 的 64K regs / 16K per scheduler 实际分配规则;LD/ST 单元 16-wide vs 32-wide warp 的拆分;TC 与 ALU 的 issue port 共享情况;真实 register pressure 案例(GPT 70B forward)

### G2 — 内存层级

- **03 SMEM+L1**: 加 SMEM swizzle 模式(32B/64B/128B)与 wgmma fragment 对齐的关系;double-buffer + producer-consumer 实测吞吐;async copy `cp.async.cg` vs `cp.async.ca` 选择
- **04 L2**: 加 L2 set-aside 实际机制(way-bias)+ persistence 与普通 LRU 的交互;`cudaCtxResetPersistingL2Cache` 何时调真有意义;hot embedding table 的实际配置案例(bytes / hit_ratio 调参)
- **05 HBM3+GMEM**: 加 row buffer hit / partial / miss 三态延迟差(50/100/150 ns);bank-group rotation 调度;async global load + L2 sector tracker;NSight Compute `dram__sectors_*` 全套 metric
- **06 atomics**: 加 L2 ALU 的 atomic 串行化粒度(per-line per-cycle);`red.async` 的 fire-and-forget 队列深度;BF16/FP8 atomic add 的硬件支持矩阵;histogram production case(SMEM bin → GMEM merge)

### G3 — Tensor Core + Hopper async + Cluster

- **07 TC**: 加 mma fragment 在 regfile 的具体 lane × elem 布局;FP8 E4M3 vs E5M2 数值范围 + 训练 / 推理使用建议;sparsity 2:4 + INT8 数据通路
- **08 wgmma**: 加 WGMMA descriptor 64-bit 字段(addr / leading_dim / stride / swizzle / base_offset);commit_group 队列深度(4 group);warp-specialization producer-consumer 模板(CUTLASS 3.x);wgmma + TMA + mbarrier 三件套真实实现导读
- **09 TMA**: 加 CUtensorMap 解码路径;5D box 越界 / 部分越界处理(zero-fill);跨 cluster TMA(`cp.async.bulk.tensor.x.shared::cluster.tile`)与单 SM 延迟差;TMA L2 cache 行为(promote / demote)
- **10 mbarrier**: 加 64-bit 内部布局(20-bit pending tx + 20-bit arrived + 20-bit expected + 1-bit phase + 余 reserve);phase 翻转后旧 wait token 的硬件检测;`mbarrier.try_wait.parity` vs `try_wait.token` 选择;ThunderKittens / CUTLASS 中的实际用法
- **11 Cluster**: 加 GPC 内 SM 选址(8 SM / GPC 在 SXM5);DSMEM 地址翻译的 GPC-local 限制;cluster TMA store 的 coalesce 规则;cluster size > 8 的 driver gating

### G4 — 调度 + Stream + 多 GPU + Graph + 持久化

- **12 调度+GigaThread**: 加 GigaThread 的 cluster grid dispatch 同步要求;tail effect 量化(grid % SM_count != 0);`cudaFuncAttributePreferredClusterDimension` 实际效果;preempt(MIG 切换 / TimeSlice)开销
- **13 Stream+Event**: 加 hyper-Q 32 hw queue 与 N stream 的 N:1 映射;default-per-thread vs legacy 行为差;CUDA 12 引入的 `cudaStreamGetCaptureInfo` 调试 capture mode 死锁;event 跨 device 的 P2P 要求
- **14 NVLink+NVSwitch**: 加 NVLink 4 link 编码(NRZ + RS-FEC);NVSwitch 3 SHARP engine 的 reduce 数据通路;DGX H100 8-GPU vs NVL36/NVL72 拓扑差;`nvidia-smi nvlink -gt c` 输出解读
- **15 NCCL**: 加 NCCL ring 算法的 chunk size 自适应(`NCCL_BUFFSIZE`);tree allreduce 在小 message 的 log N latency 优势量化;SHARP allreduce 的 in-network reduce 实测带宽提升;NCCL deadlock 调试(`NCCL_DEBUG=TRACE`)
- **16 CUDA Graph**: 加 capture mode 三种(global/thread/relaxed)的隔离级别;`cudaGraphInstantiateFlagAutoFreeOnLaunch` 用法;conditional graph node 12.4+ 设备侧 vs 主机侧;PyTorch `torch.cuda.graph` 与 NCCL collective 的兼容陷阱
- **17 Persistent+DP**: 加 persistent kernel 占满 SM 与 priority stream 的饿死问题;DP 的 launch queue 深度(默认 2048)与 OOM;`cudaLaunchKernelEx` 的 attribute 列表;CUTLASS 3.x persistent GEMM 实战导读

### G5 — 内存管理 + Driver API + 工具链 + 编译

- **18 Pool**: 加 PyTorch CUDACachingAllocator 在 cudaMallocAsync 之上的 buddy + free-list 实现;`PYTORCH_CUDA_ALLOC_CONF` 调参实战(expandable_segments、garbage_collection_threshold);跨 stream 引用计数 race
- **19 Unified Memory**: 加 page fault 处理的 GPU MMU 路径;HMM 在 GH200 / MI300 上的 zero-copy 行为;`cudaMemAdviseSetPreferredLocation` + `SetAccessedBy` 的页表预热;真实 case:scientific simulation 的 UM 性能 5 倍差异
- **20 Driver API**: 加 primary context 与 explicit context 的并发 race;`cuModuleLoadDataEx` JIT 编译缓存(`~/.nv/ComputeCache`);TensorRT 直接调 driver API 的原因(避免 runtime 的 LazyContext 开销);Triton 编译产物如何 `cuModuleLoad`
- **21 Profiling**: 加 NSight Compute 的 kernel replay 语义(每 metric 重跑一次,改变 cache 状态);Compute 的 `--target-processes all` 与多 GPU profile;CUPTI Activity API 实战(自己写最小 profiler);PyTorch profiler 的 `with_stack` overhead 量化
- **22 PTX→SASS**: 加 ptxas register allocator 提示(`.maxnreg`);spill-to-local 的 latency 实测;`ptxas -O3` 关键 pass(SSA、loop unroll、scheduling);SASS 反汇编看 wgmma + TMA 真实 ucode;JIT vs AOT 编译的 trade-off

### G6 — 训练 + 推理 capstone

- **23 训练全栈**: 加 70B / 405B 模型 TP×PP×DP 不同配置实测 MFU 表;FSDP gradient bucket 调参实战;FP8 training overflow handler(Transformer Engine `fp8_autocast`);`torch.compile` + `torch.cuda.graph` 组合的边界条件;ZeRO-3 的 all-gather pre-fetch 实战
- **24 推理全栈**: 加 prefill chunking(Sarathi-Serve)与 continuous batching 调度对比;PagedAttention block table 的 prefix-cache 命中加速实测;FP8 inference + INT4 weight-only 的精度损失矩阵;DistServe 的 prefill / decode 拆机器实测;ThunderKittens / Cutlass attention kernel 性能对比

## 5. 字数与 Mermaid 配额

- 每章 4000-5000 中文字(原 1500-2500;增量 ~2500-3000 字)
- 每章 Mermaid ≥ 2(原 ≥ 1;§2 与 §3/§5/§7 任一节再加 1 个)
- 00 索引 ≥ 3(原 2)
- 全集 Mermaid 总数:**≥ 51**(25 章 × 2 + 00 章额外 1)

## 6. 验收准则

- [ ] 25 章字数全部在 [4000, 5000] 区间
- [ ] 每章 Mermaid ≥ 2(00 章 ≥ 3)
- [ ] 全集 Mermaid 总数 ≥ 51
- [ ] 全部章节零 gpusim 引用
- [ ] 6 个里程碑 tag 到位:`cuda-zh-deep-DG1-complete` ... `cuda-zh-deep-DG6-complete` + `cuda-zh-deep-complete`
- [ ] master + 新 tag 推送到 origin

## 7. 里程碑

| Milestone | Scope | Tag |
|---|---|---|
| **DG1** | 00, 01, 02 (3 章) | `cuda-zh-deep-DG1-complete` |
| **DG2** | 03, 04, 05, 06 (4 章) | `cuda-zh-deep-DG2-complete` |
| **DG3** | 07, 08, 09, 10, 11 (5 章) | `cuda-zh-deep-DG3-complete` |
| **DG4** | 12, 13, 14, 15, 16, 17 (6 章) | `cuda-zh-deep-DG4-complete` |
| **DG5** | 18, 19, 20, 21, 22 (5 章) | `cuda-zh-deep-DG5-complete` |
| **DG6** | 23, 24 (2 章) + push + final tag | `cuda-zh-deep-complete` |

每个 DG 单 subagent 一次 dispatch 完成。

## 8. 文件清单

### 修改(仅修改,不新增)
```
docs/cuda-zh/00-index.md
docs/cuda-zh/01-simt-execution.md
... (全部 25 章)
docs/cuda-zh/24-inference-end-to-end.md
```

### 新增
无。

## 9. 实施策略要点

- **subagent 必须先读现有章节**,在原文基础上加深;不是从零重写
- 保持原有 8 节结构与 §7 capstone 特化(23/24 章 §7 仍是优化方法体系)
- 新增 Mermaid 主要在 §2(微架构图)+ §3 或 §5(实现/调用流程)
- 所有真实数字必须可对照 NVIDIA 官方文档或公开论文,引用注明出处
