# Chapter 27 — Multi-Stream Concurrency Basics

## What Is a CUDA Stream?

Every GPU kernel launch is ordered within a **stream** — a software-level queue that the hardware processes in FIFO order. Within a single stream, kernels execute one after another: kernel B does not start until kernel A has drained every CTA. Across different streams there is no such ordering guarantee, which means the hardware is free to overlap their execution as SM occupancy allows.

Streams are the fundamental building block for GPU concurrency. The classic use cases are:

- **Compute + compute overlap:** two independent kernels share the SM pool if their combined CTA count does not exceed occupancy.
- **Compute + transfer overlap:** a CUDA memcpy on one stream can run while a kernel executes on another, using the DMA engine in parallel with the SM array.
- **Pipeline stages:** large workloads broken into tiles, where stream N processes tile K+1 while stream N-1 writes tile K back to host.

On the simulator, streams are first-class objects created with `gpusim.Stream()` and submitted via `Stream.launch()`. The collection is drained with `gpusim.synchronize()`, which returns a `MultiStreamResult` holding per-stream and per-kernel metrics.

## 走通 concurrent_vector_add_2stream

```bash
python examples/concurrent_vector_add_2stream/run.py
```

The demo creates two independent vector-add workloads — arrays `A+B→C` and `D+E→F` — and assigns each to its own stream:

```python
from gpusim.api import Stream
import gpusim

s0 = Stream()
s1 = Stream()

s0.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"A": A, "B": B, "OUT": C}, kernel_name="vec_add_a", config=cfg)

s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"A": D, "B": E, "OUT": F}, kernel_name="vec_add_b", config=cfg)

multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
```

`gpusim.synchronize()` is the barrier: it blocks until every pending launch across all listed streams has completed, then assembles a `MultiStreamResult`. The PTX kernel is the same entry point for both streams; the per-stream parameter dictionaries (`A`, `B`, `OUT`) distinguish the data.

The kernel itself is a straightforward load-add-store:

```ptx
ld.global.f32 %f0, [%rd4];   // load A[tid]
ld.global.f32 %f1, [%rd4];   // load B[tid]
add.f32 %f2, %f0, %f1;
st.global.f32 [%rd4], %f2;   // store C[tid]
```

Expected output:

```
Stream 0 cycles: <N>
Stream 1 cycles: <N>
C[0:4] = [0.0, 3.0, 6.0, 9.0]
F[0:4] = [0.0, 7.0, 14.0, 21.0]
```

Both streams verify correctly. The cycle counts will be similar — this is by design for a symmetric workload.

> **Simulator note:** Phase 7's `run_streams` uses **sequential drain**: it finishes all CTAs from stream 0 before advancing to stream 1. The total cycle count therefore equals the sum of the two individual kernel cycle counts. True cross-grid CTA interleaving — and the resulting cycle savings — is not yet implemented. The API and metric structure are wired for that future iteration; the concurrency story is architecturally correct.

## 看模拟器

```python
multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
print(multi_res.stream_summary())
```

`stream_summary()` prints a table with one row per stream: stream ID, number of kernels launched, and total cycles. You can also inspect individual results:

```python
res_s0 = multi_res.streams[0][0]   # stream 0, launch 0
print(res_s0.metrics["cycles"])
print(res_s0.kernel_name)          # "vec_add_a"
```

Open `report.html` and navigate to **§27 Multi-Stream Timeline**. The timeline lane shows stream 0 completing before stream 1 starts — consistent with sequential drain. When the scheduler gains CTA interleaving, the lanes will partially overlap.

## 改一改

**1 stream vs 2 streams.** Merge both launches onto a single stream and compare total cycles:

```python
s = Stream()
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
         params={"A": A, "B": B, "OUT": C}, kernel_name="vec_add_a", config=cfg)
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
         params={"A": D, "B": E, "OUT": F}, kernel_name="vec_add_b", config=cfg)
multi_res_1s = gpusim.synchronize(streams=[s], config=cfg)
```

In the current simulator both approaches produce the same total cycle count (sequential drain). On real H100/A100 hardware, 2 streams would allow the hardware multi-stream scheduler to interleave CTAs from both grids across the SM pool, potentially cutting wall-clock time by up to 50% when each grid alone uses fewer than half of available SMs.

## 真机对照

In CUDA C++:

```cpp
cudaStream_t s0, s1;
cudaStreamCreate(&s0);
cudaStreamCreate(&s1);

vec_add_kernel<<<1, 32, 0, s0>>>(A, B, C);
vec_add_kernel<<<1, 32, 0, s1>>>(D, E, F);

cudaStreamSynchronize(s0);
cudaStreamSynchronize(s1);
// or: cudaDeviceSynchronize();
```

`cudaStreamSynchronize(s)` blocks the calling CPU thread until all work queued on stream `s` is complete. `cudaDeviceSynchronize()` waits for all streams. On hardware, both kernels above are eligible for concurrent execution once the GPU's task manager has dispatched them; whether they actually overlap depends on whether there are enough free SMs to host both grids simultaneously.

Use `nvprof --print-gpu-trace` or Nsight Systems to confirm actual overlap: look for overlapping rows in the GPU activity timeline. The equivalent in the simulator is `report.html` §27 or the Perfetto trace (`gpusim.to_perfetto(multi_res)`), where each stream appears as its own swimlane.
