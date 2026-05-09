# Chapter 36 — Production Multi-Stream Pipeline

## Combining Priority, Events, and L2 Window

The previous five chapters introduced Phase 8's features individually. Real production workloads combine all of them. This chapter assembles a three-stage pipeline that uses priority, CUDA events, and L2 cache window partitioning together — the same architecture you would find in a CUTLASS persistent matmul kernel or a DNN inference serving loop.

The pipeline has three stages:

1. **Load** (`s_load`, priority normal) — reads input data from global memory into a staging buffer.
2. **Compute** (`s_compute`, priority high) — reads the staged data, performs computation, writes to an intermediate buffer. This is the critical stage; it gets the highest priority and an L2 window to protect its working set.
3. **Store** (`s_store`, priority normal) — writes the compute output to the final destination.

Two CUDA events chain the stages:

```
s_load → [kernel_load] → record ev1
                               ↓
s_compute → wait ev1 → [kernel_compute] → record ev2
                                                ↓
s_store → wait ev2 → [kernel_store]
```

## Running the Demo

```bash
python examples/multi_stream_pipeline_full/run.py
```

```python
s_load    = Stream(priority="normal")
s_compute = Stream(priority="high")
s_compute.set_l2_window(start_set=0, n_sets=16)   # protect compute's hot sets
s_store   = Stream(priority="normal")

ev1 = Event()
ev2 = Event()

s_load.launch(ptx_src=load_ptx,    kernel_name="load",    ...)
s_load.record(ev1)

s_compute.wait(ev1)
s_compute.launch(ptx_src=compute_ptx, kernel_name="compute", ...)
s_compute.record(ev2)

s_store.wait(ev2)
s_store.launch(ptx_src=store_ptx,  kernel_name="store",   ...)

multi_res = gpusim.synchronize(streams=[s_load, s_compute, s_store], config=cfg)
print(f"OUT[0:4] = {list(OUT[0:4])}")
print(f"event_chain_critical_path: {multi_res.event_chain_critical_path()}")
```

## 看模拟器

The key metric for this pattern is the **event-chain critical path**:

```python
critical_path = multi_res.event_chain_critical_path()
print(f"Critical path: {critical_path} cycles")
```

`multi_res.event_chain_critical_path()` traces the longest chain of event dependencies — from the first kernel launch through each `record → wait` edge to the last kernel completion — and returns the minimum cycle count that the pipeline could not have been shorter than, given the data dependencies. This is the GPU equivalent of the critical path in a DAG scheduler.

For the load → compute → store chain, the critical path is approximately:

```
load_cycles + wait1_overhead + compute_cycles + wait2_overhead + store_cycles
```

Where `wait_overhead` is the number of cycles a stream spent blocked. You can decompose this with:

```python
wait_cycles = multi_res.event_wait_cycles_per_stream()
for sid, w in wait_cycles.items():
    print(f"  stream {sid} waited {w} cycles")
```

Also check the L2 window metrics to confirm the compute stage's cache behavior:

```python
print(f"L2 window hit rate: {multi_res.l2_window_hit_rate()}")
print(f"L2 protection efficiency: {multi_res.l2_window_protection_efficiency():.3f}")
```

## 改一改

**Drop priority — observe critical path lengthening.** Set all three streams to `"normal"` priority and remove the L2 window:

```python
s_load    = Stream(priority="normal")
s_compute = Stream(priority="normal")   # was "high"
# s_compute.set_l2_window(...)          # removed
s_store   = Stream(priority="normal")
```

Rerun and compare `event_chain_critical_path()`. With equal priority, the compute stage no longer gets preferential CTA dispatch, so it takes longer to complete — the critical path grows. In the priority-enabled run, the scheduler allocated 4 CTA dispatch slots per cycle to `s_compute` versus 2 for the other streams; in the equal-priority run, all streams get 2 slots, halving the compute stage's throughput advantage.

Also observe that without the L2 window, the compute stage's hot lines compete with the load and store stages' traffic. The `l2_window_protection_efficiency` drops to near 0.0 (or returns a neutral value), and in a workload with a larger working set you would see compute-stage L2 hit rates degrade.

## 真机対照

This three-stage pattern appears in production GPU code under different names:

**CUTLASS persistent matmul with cooperative epilogue** (Chapter 24): the load stage prefetches tiles into shared memory using TMA, the compute stage runs the wgmma pipeline at high priority, and the epilogue (store) stage writes back with stream-ordered atomics. CUTLASS uses `cudaStreamCreateWithPriority` to elevate the compute stream and `cudaAccessPropertyPersisting` on the weight tensor's address range.

**DNN inference serving**: a model serving loop runs prefill at high priority (latency-sensitive) and decode at normal priority (throughput-sensitive). An event separates the two phases so decode can begin on one request while prefill runs on the next. L2 windows protect the KV-cache from being evicted by the prefill stream's large activations.

The simulator's three-stage pipeline captures the essential scheduling behavior of both production patterns. The toy kernels (`kernel_load.ptx`, `kernel_compute.ptx`, `kernel_store.ptx`) are small enough to run in milliseconds, but the metric relationships — critical path, wait cycles, priority dispatch share, L2 protection efficiency — scale directly to production workloads.

## Phase 8 Summary

Chapters 31–36 have covered all four Phase 8 features:

- **Chapter 31**: `ConcurrentStreamScheduler` — per-cycle WRR dispatch replacing sequential drain.
- **Chapter 32**: `Stream(priority=...)` — high/normal/low with 4:2:1 weighted dispatch.
- **Chapter 33**: `Event` + `stream.record` + `stream.wait` — kernel-granularity cross-stream synchronization.
- **Chapter 34**: Event fanout — one producer signal unblocks N consumers simultaneously.
- **Chapter 35**: `stream.set_l2_window` — L2 set reservation protecting critical-stream data.
- **Chapter 36**: Combined pipeline — priority + events + L2 window in a single production-shaped workload.

Phase 9 will add per-line L2 access counters for precise window-hit-rate metrics, arbitrary integer stream priorities matching the full CUDA API range, and cross-device event signaling.
