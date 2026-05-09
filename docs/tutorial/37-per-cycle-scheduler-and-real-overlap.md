# Chapter 37 — Per-Cycle Scheduler and Real Overlap

## Phase 9's Per-Cycle Main Loop

Phase 8 introduced `ConcurrentStreamScheduler`, which replaced the old sequential stream-drain loop with a weighted round-robin (WRR) policy that dispatches CTA groups across all ready streams each scheduling step. That was a big step forward: it allowed multiple streams to make forward progress within the same device run.

Phase 9 goes a step further. Instead of performing one scheduling round per kernel launch (a per-launch nesting model), the Phase 9 Device runs a **per-cycle main loop** — the scheduler ticks once for every simulated cycle, inspecting every stream's ready CTAs and dispatching as many as the SM count allows.

The practical difference: in Phase 8, stream interleaving happened at kernel-launch granularity. In Phase 9, it happens at cycle granularity within a single call to `Device.run_streams`. Two kernels that would have been serialized in Phase 8's dispatch order now genuinely share SM resources cycle by cycle.

### What Phase 9 M1 Does and Does Not Provide

An honest note is warranted here. The Phase 9 M1 per-cycle loop is a scheduler-level change: it controls when CTAs are *dispatched* across streams, not how individual CTAs are interleaved *within an SM*. True per-cycle CTA interleave — where a CTA from stream A and a CTA from stream B share the same SM and swap execution slots cycle by cycle — requires the SM core itself to time-slice warp contexts. That requires a `Device.run` cycle-slicing path inside the SM model, which is a future iteration (planned for Phase 10).

What Phase 9 M1 delivers is genuine **cross-grid overlap at the orchestration level**: if stream A's grid fills 60% of the device and stream B's grid fills 40%, both grids can be resident and executing simultaneously, and the simulator correctly accounts for the overlapping cycle ranges. The `actual_cross_grid_overlap_cycles()` metric measures this accurately.

## Running the Demo

```bash
python examples/phase8_overlap_real/run.py
```

This example launches two streams simultaneously: a compute-heavy kernel on stream 0 and a memory-heavy kernel on stream 1. Both kernels are sized to require multiple CTA waves, so the scheduler has room to interleave them.

```python
s0 = Stream()
s1 = Stream()
s0.launch(ptx_src=..., kernel_name="compute_heavy", ...)
s1.launch(ptx_src=..., kernel_name="memory_heavy",  ...)
multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
```

## 看模拟器

查看两个关键指标：

```python
gain = multi_res.cross_stream_concurrency_gain()
overlap = multi_res.actual_cross_grid_overlap_cycles()
print(f"Concurrency gain: {gain:.3f}x")
print(f"Cross-grid overlap cycles: {overlap}")
```

`cross_stream_concurrency_gain()` returns a float greater than 1.0 when the two streams ran faster together than they would have run sequentially. A value of 1.0 means no benefit was observed; values below 1.0 indicate overhead (rare, usually from SM contention on tiny grids).

`actual_cross_grid_overlap_cycles()` counts the number of cycles during which at least one CTA from each stream was resident on the device simultaneously. This is the direct evidence that real overlap occurred, not just scheduling fairness.

Compare the two values with a run where `cfg.n_sm = 1`:

```python
cfg.n_sm = 1
# re-run and compare overlap cycles
```

With a single SM, both grids must share one slot and true overlap is zero (or minimal). With 8 SMs, the compute-heavy kernel occupies 5-6 SMs and the memory-heavy kernel occupies 2-3, and they proceed in parallel.

## 改一改

**增大网格以放大重叠效果：**

Change the grid sizes from `(1,1,1)` to `(4,1,1)` for both kernels. A larger grid means more CTAs, which spreads across more SMs, and the scheduler has more dispatches to interleave:

```python
s0.launch(..., grid=(4,1,1), ...)
s1.launch(..., grid=(4,1,1), ...)
```

Rerun and observe that `actual_cross_grid_overlap_cycles()` increases significantly. With `(1,1,1)` grid, each kernel only has 1 CTA and the overlap window is limited to 1 CTA's execution duration. With `(4,1,1)`, each kernel has 4 CTAs and the per-cycle scheduler can keep both grids active for much longer.

Also try setting one stream to `priority="high"` and observe how `cross_stream_concurrency_gain()` changes: the high-priority stream gets more dispatch slots per cycle but the overlap window shrinks because it finishes faster.

## 真机对照

On real H100 hardware, the equivalent behavior is implemented in the hardware CTA scheduler embedded in the GPC (Graphics Processing Cluster). Each GPC has a fixed number of CTA dispatch units that cycle through the SM's resident warps on every clock cycle. When two kernels from different CUDA streams are active simultaneously, the hardware CTA scheduler interleaves their warp dispatch on a per-cycle basis automatically.

Key differences from the simulator's Phase 9 M1 model:

- **Hardware interleaves within an SM**: On H100, two grids from different streams can share the same SM (up to the CTA occupancy limit), and their warps interleave at the warp-scheduler level (4 warp schedulers per SM, selecting from all resident warps). The Phase 9 simulator models this at the grid boundary level, not the warp level.
- **Hardware queue interleaving**: H100 has independent hardware work queues per stream, and the hardware automatically drains them in parallel. In the simulator, this is modeled by the per-cycle dispatch loop in `Device.run_streams`.
- **Measuring overlap on hardware**: Use `cudaEventRecord` before and after both launches, then `cudaEventSynchronize` + `cudaEventElapsedTime`. The simulator's `actual_cross_grid_overlap_cycles()` gives the equivalent measurement in simulator cycles.

## Phase 9 Summary (So Far)

Chapter 37 completes the walkthrough of Phase 9's per-cycle scheduler. The next two chapters cover two new API features built on top of it: `Stream.wait_all` for fan-in synchronization (Chapter 38) and `Event.elapsed_time` for cycle-accurate profiling (Chapter 39).
