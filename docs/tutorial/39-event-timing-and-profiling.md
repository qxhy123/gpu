# Chapter 39 — Event Timing and Profiling

## Measuring Kernel Duration with Events

CUDA events serve two purposes: synchronization (Chapters 33–38) and timing. You can bracket any sequence of operations between two recorded events and then query the elapsed time. On hardware this returns wall-clock time in milliseconds; in the simulator it returns an integer cycle count.

Phase 9 adds `Event.elapsed_time(start, end)` to the public API:

```python
elapsed = Event.elapsed_time(ev_start, ev_end)
```

The return value is an `int` — the number of simulated cycles between when `ev_start` was signaled and when `ev_end` was signaled. Both events must have been recorded (signaled) before calling `elapsed_time`; calling it before synchronization raises a `RuntimeError`.

This is a static method: it does not require a stream or device instance. Both events carry their signal cycle as an attribute set at record time, and `elapsed_time` simply computes the difference.

## Running the Demo

```bash
python examples/event_timing_benchmark/run.py
```

The demo records a start event, launches a kernel, records an end event, synchronizes, and prints the elapsed cycle count:

```python
s = Stream()
ev_start = Event(); ev_end = Event()

s.record(ev_start)
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
         params={"A": A, "B": B, "OUT": C}, kernel_name="vec_add", config=cfg)
s.record(ev_end)

gpusim.synchronize(streams=[s], config=cfg)
print(f"Kernel took {Event.elapsed_time(ev_start, ev_end)} cycles")
```

The output is a single integer: the number of cycles from the stream's start event to its end event, inclusive of any scheduling latency between the `record` call and the kernel's first CTA dispatch.

## 看模拟器

**打印 elapsed cycles 并与 per-launch cycles 对比：**

After synchronization, compare the event-based elapsed time with the per-launch cycle count reported in the result:

```python
elapsed_ev = Event.elapsed_time(ev_start, ev_end)

# Per-launch cycle count from the result object
launch_result = multi_res.streams[s.stream_id][0]
launch_cycles = launch_result.metrics["cycles"]

print(f"Event elapsed time: {elapsed_ev} cycles")
print(f"Per-launch cycles:  {launch_cycles} cycles")
print(f"Overhead:           {elapsed_ev - launch_cycles} cycles")
```

The event elapsed time will be slightly larger than the per-launch cycle count. The difference is scheduling overhead: the cycles between when `ev_start` was recorded and when the first CTA was actually dispatched (the kernel had to wait in the stream's ready queue). On a lightly loaded device this overhead is 0–2 cycles; on a device with many concurrent streams it can be larger.

This comparison lets you distinguish between:
- **Kernel execution time** (`launch_cycles`): how long the kernel ran once it started.
- **End-to-end latency** (`elapsed_ev`): how long the operation took from enqueue to completion, including any queueing delay.

For latency-sensitive workloads (e.g., inference serving), the end-to-end latency is the metric that matters. For throughput optimization, kernel execution time is the primary lever.

## 改一改

**对多发射流水线的各阶段分别计时：**

Add events between each launch in a multi-stage pipeline to measure each phase individually:

```python
s = Stream()
ev0 = Event(); ev1 = Event(); ev2 = Event(); ev3 = Event()

s.record(ev0)
s.launch(..., kernel_name="load", ...)
s.record(ev1)
s.launch(..., kernel_name="compute", ...)
s.record(ev2)
s.launch(..., kernel_name="store", ...)
s.record(ev3)

gpusim.synchronize(streams=[s], config=cfg)

load_cycles    = Event.elapsed_time(ev0, ev1)
compute_cycles = Event.elapsed_time(ev1, ev2)
store_cycles   = Event.elapsed_time(ev2, ev3)
total_cycles   = Event.elapsed_time(ev0, ev3)

print(f"Load:    {load_cycles} cycles ({100*load_cycles/total_cycles:.1f}%)")
print(f"Compute: {compute_cycles} cycles ({100*compute_cycles/total_cycles:.1f}%)")
print(f"Store:   {store_cycles} cycles ({100*store_cycles/total_cycles:.1f}%)")
print(f"Total:   {total_cycles} cycles")
```

This gives a breakdown that would require four separate `cudaEventRecord` calls on hardware, which is exactly what you would use in production profiling before reaching for a tool like Nsight. In the simulator, the overhead of inserting events is zero (they are record objects, not hardware instructions), so you can profile as finely as desired without affecting timing accuracy.

Try changing the grid sizes to see how each phase scales differently: the compute kernel may scale super-linearly due to cache effects, while the load and store phases scale linearly with data volume.

## 真机対照

On real CUDA hardware, the equivalent API is `cudaEventElapsedTime`:

```c
float ms;
cudaEventRecord(ev_start, stream);
// ... launches ...
cudaEventRecord(ev_end, stream);
cudaEventSynchronize(ev_end);
cudaEventElapsedTime(&ms, ev_start, ev_end);
```

Key differences from the simulator's `Event.elapsed_time`:

| | Simulator | Hardware (H100) |
|---|---|---|
| **Return type** | `int` (cycles) | `float` (milliseconds) |
| **Resolution** | 1 cycle | ~0.5 µs (H100 clock resolution) |
| **Blocking** | Only after `gpusim.synchronize` | Only after `cudaEventSynchronize` |
| **API** | `Event.elapsed_time(start, end)` | `cudaEventElapsedTime(&ms, start, end)` |
| **Overhead** | Zero (metadata only) | ~1 µs per record on stream |

To convert simulator cycles to hardware milliseconds, divide by the target clock frequency. H100 SXM5 runs at 1980 MHz in compute mode:

```python
SXM5_GHZ = 1.98
elapsed_ms = elapsed_cycles / (SXM5_GHZ * 1e6)
```

This gives a rough correspondence, though the simulator's cycle model is not cycle-accurate at the instruction level — it models memory latencies and occupancy, not pipeline hazards. Use the simulator's elapsed_time for relative comparisons (which stage is the bottleneck?) and hardware cudaEventElapsedTime for absolute latency numbers.

## Phase 9 Feature Summary

Chapters 37–39 have covered the three Phase 9 additions:

- **Chapter 37**: Per-cycle scheduler — `Device.run_streams` ticks once per cycle, enabling genuine cross-grid overlap measurement with `actual_cross_grid_overlap_cycles()`.
- **Chapter 38**: `Stream.wait_all([events])` — fan-in synchronization, blocking a consumer until all listed events have been signaled (AND semantics).
- **Chapter 39**: `Event.elapsed_time(start, end)` — cycle-accurate duration measurement between any two recorded events, returning an integer cycle count.

These three features complete Phase 9's additions to the multi-stream programming model. Phase 10 will introduce SM-level warp interleaving — the final step toward true per-SM multi-tenancy simulation.
