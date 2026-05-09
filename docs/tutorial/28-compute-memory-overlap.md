# Chapter 28 — Compute-Memory Overlap

## HBM and SM Are Independent Resources

An H100 GPU is two largely independent machines bolted together: a **compute engine** (SMs, CUDA cores, Tensor Cores, register files, shared memory) and a **memory engine** (HBM channels, L2 cache, load/store units, DMA engines). When a kernel is purely compute-bound — its arithmetic instruction throughput is the bottleneck — the memory engine sits idle. When a kernel is purely memory-bound — warps stall waiting for HBM — the compute engine sits idle.

This asymmetry is exactly why two-stream overlap of a compute-heavy kernel and a memory-heavy kernel can yield better GPU utilization than running them serially: the compute engine is busy for stream 0 while the memory engine services stream 1's loads and stores in parallel.

This is sometimes called **compute-memory overlap** or **dual-engine overlap**. It is different from the multi-stream CTA interleaving discussed in Chapter 27: that interleaves CTAs from two compute kernels across the same SM pool. Compute-memory overlap targets a different bottleneck pair.

## 走通 compute_vs_memory_overlap

```bash
python examples/compute_vs_memory_overlap/run.py
```

The demo assigns two deliberately asymmetric kernels to two streams.

**`kernel_compute.ptx`** — the compute-heavy kernel executes eight chained `add.f32` instructions between a load pair and a store. All arithmetic is dependent on the previous result, so no instruction-level parallelism is possible, but the SM's floating-point units are occupied throughout:

```ptx
ld.global.f32 %f0, [%rd4];   // load A[tid]
ld.global.f32 %f1, [%rd4];   // load B[tid]
// 8 chained additions:
add.f32 %f2, %f0, %f1;
add.f32 %f2, %f2, %f0;
add.f32 %f2, %f2, %f1;
// ... (4 more)
st.global.f32 [%rd4], %f2;
```

**`kernel_memory.ptx`** — the memory-heavy kernel does exactly two loads, one addition, and one store. It has very low arithmetic density; the warp spends most of its time waiting for L2/HBM to return data:

```ptx
ld.global.f32 %f0, [%rd4];   // load A[tid]
ld.global.f32 %f1, [%rd4];   // load B[tid]
add.f32 %f2, %f0, %f1;
st.global.f32 [%rd4], %f2;
```

The run script dispatches them to separate streams:

```python
s0 = Stream()
s1 = Stream()

s0.launch(ptx_src=(here / "kernel_compute.ptx").read_text(),
          grid=(1,1,1), block=(32,1,1),
          params={"A": A, "B": B, "OUT": C},
          kernel_name="compute_heavy", config=cfg)

s1.launch(ptx_src=(here / "kernel_memory.ptx").read_text(),
          grid=(1,1,1), block=(32,1,1),
          params={"A": D, "B": E, "OUT": F},
          kernel_name="memory_heavy", config=cfg)

multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
print(multi_res.stream_summary())
```

On real hardware, the combined wall-clock time would be close to `max(T_compute, T_memory)` rather than `T_compute + T_memory`. The overlap ratio approaches 1.0 when the two kernels take exactly the same time and use fully disjoint hardware resources.

> **Simulator note:** Phase 7's sequential drain runs `compute_heavy` to completion before starting `memory_heavy`. The `overlap_ratio()` metric is wired and returns a value, but under sequential drain the realized overlap is 0. The API and metric are ready for a future scheduler that dispatches CTAs from both streams simultaneously across different SM partitions.

## 看模拟器

```python
print(multi_res.stream_summary())
print("Overlap ratio:", multi_res.overlap_ratio())
```

`overlap_ratio()` computes how much of the total elapsed time the two streams were running simultaneously. Under sequential drain it returns 0.0. Once CTA-level interleaving lands, it will return a value between 0 and 1 — closer to 1 means better overlap.

For the Perfetto trace:

```python
import gpusim
gpusim.to_perfetto(multi_res, path="trace.json")
```

Open `trace.json` in `ui.perfetto.dev`. Stream 0 and stream 1 appear as separate swimlanes. In the current simulator they appear as consecutive non-overlapping blocks. In future iterations, the two lanes will have overlapping time spans, which is the visual signal that compute-memory overlap is active.

The HTML report **§28 Compute vs Memory** table shows per-stream arithmetic intensity (instructions issued vs. memory bytes requested), which makes it easy to confirm that `compute_heavy` and `memory_heavy` are indeed in different resource regimes.

## 改一改

**Replace `compute_heavy` with another `memory_heavy` copy.** Change stream 0 to also use `kernel_memory.ptx`:

```python
s0.launch(ptx_src=(here / "kernel_memory.ptx").read_text(),
          grid=(1,1,1), block=(32,1,1),
          params={"A": A, "B": B, "OUT": C},
          kernel_name="memory_heavy_2", config=cfg)
```

Now both streams are memory-bound. On real hardware, this scenario typically shows poor overlap because both kernels contend for the same HBM channels and L2 bandwidth. There is nothing idle to fill: both streams want the same bottleneck resource. The `overlap_ratio()` will reflect this (or stay at 0 in the simulator for the same sequential-drain reason). Chapter 29 explores the L2/HBM contention angle in more detail.

## 真机对照

CUTLASS's multi-stage GEMM pipelines use compute-memory overlap internally — the epilogue stream stores results to HBM while a separate warp group issues the next GEMM tile's global memory loads. This is achieved by decoupling the GEMM kernel across two stream-like warp specializations (producers and consumers) using `cuda::pipeline` barriers.

At the application level, CUTLASS's `CudaHostAdapter` launches GEMM and epilogue-scaling kernels on separate CUDA streams when the two operations touch different memory regions, allowing the SM's load-store units for the epilogue and the CUDA cores for the GEMM to run concurrently.

Profile with `ncu --set full` and look at the **Compute Throughput** vs **Memory Throughput** utilization charts: a well-overlapped compute+memory pair will show both bars near their individual ceilings for the overlapping portion of the timeline. A memory+memory pair will show Memory Throughput near 100% but Compute Throughput near 0%.
