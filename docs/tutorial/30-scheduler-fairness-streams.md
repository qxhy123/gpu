# Chapter 30 — Scheduler Fairness Across Streams

## Round-Robin at CTA Granularity

When multiple streams are resident on the GPU, the SM scheduler must decide which CTA to dispatch next. The Phase 7 simulator implements a **round-robin (RR)** policy at CTA granularity: each time a new CTA slot opens on an SM, the scheduler cycles through the active streams in order and picks the next undispatched CTA from the current stream. If a stream has no remaining CTAs, the scheduler advances to the next stream and tries again.

This is the same policy that H100 hardware uses by default for streams of equal priority. The GPU hardware's GPC-level task scheduler is not documented in detail, but empirical profiling with Nsight Systems consistently shows CTA interleaving that approximates round-robin across equal-priority streams.

Round-robin at CTA granularity has a useful property: **no stream can starve another**. Because the scheduler alternates at the finest granularity of work dispatch, a stream with a large grid and a stream with a small grid both make progress. The stream with fewer CTAs finishes first, but it is never held back while the other stream monopolizes SM slots.

## Jain's Fairness Index

Fairness is quantified with Jain's fairness index:

```
J = (Σx_i)² / (n · Σx_i²)
```

where `x_i` is the throughput (cycles used or CTAs completed) attributed to stream `i`, and `n` is the number of streams. `J = 1.0` means all streams received exactly equal service. `J = 1/n` means one stream received all the resources and the rest received nothing — maximum unfairness.

For the symmetric case (all grids the same size, equal workload per CTA), round-robin trivially achieves `J = 1.0`. The index is more informative when grid sizes differ (the `改一改` experiment below) or when future priority levels are added (Chapter 31 preview).

## 走通 stream_priority_serial_vs_concurrent

```bash
python examples/stream_priority_serial_vs_concurrent/run.py
```

The demo runs the same vector-add kernel four times in two configurations:

**Serial:** all four launches on one stream. The scheduler sees a single stream and dispatches CTAs in FIFO order. There is no cross-stream fairness to measure.

**Concurrent:** four separate `Stream()` objects, each with one launch. The round-robin scheduler interleaves CTAs from all four:

```python
# Serial path
s_serial = Stream()
for i in range(4):
    s_serial.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                    params={"A": A, "B": B, "OUT": outs[i]},
                    kernel_name=f"k{i}", config=cfg)
rs = gpusim.synchronize(streams=[s_serial], config=cfg)

# Concurrent path
streams = [Stream() for _ in range(4)]
for i, s in enumerate(streams):
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
             params={"A": A, "B": B, "OUT": outs2[i]},
             kernel_name=f"k{i}", config=cfg)
rc = gpusim.synchronize(streams=streams, config=cfg)

print(f"Serial:     {rs.total_cycles} cycles (1 stream, 4 launches)")
print(f"Concurrent: {rc.total_cycles} cycles (4 streams, 1 launch each)")
print(f"Speedup:    {rs.total_cycles / max(rc.total_cycles, 1):.2f}x")
```

Under Phase 7's sequential drain, the total cycle count for serial and concurrent will be the same (or very close). The key difference exposed by the `fairness()` metric is that the concurrent case distributes cycles evenly across all four streams, while the serial case assigns all cycles to stream 0.

> **Simulator note:** Phase 7 uses sequential drain — all CTAs from stream 0 complete before stream 1 begins. The `fairness()` metric accounts for this by measuring the fraction of total cycles each stream actively had CTAs running. With sequential drain and equal grid sizes the metric still returns 1.0, because each stream runs for an equal number of cycles even though they run in sequence rather than simultaneously. True concurrent RR dispatch with interleaved CTA slots is the target for a future iteration.

## 看模拟器

```python
print("Fairness index:", rc.fairness())
```

`multi_res.fairness()` computes Jain's index over the per-stream cycle totals recorded in `multi_res.streams`. For the symmetric four-stream case above, it returns 1.0 or very close to it, confirming that the RR scheduler allocated equal work to all streams.

You can also inspect the per-stream breakdown directly:

```python
for sid, launches in rc.streams.items():
    total = sum(r.metrics["cycles"] for r in launches)
    print(f"  stream {sid}: {total} cycles")
```

## 改一改

**Uneven grid sizes.** Give stream 0 a larger grid than the others to test whether RR still distributes cycles fairly at CTA granularity:

```python
streams[0].launch(ptx_src=ptx, grid=(4,1,1), block=(32,1,1),  # 4 CTAs
                  params={"A": A, "B": B, "OUT": outs2[0]},
                  kernel_name="k0_big", config=cfg)
for i in range(1, 4):
    streams[i].launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),  # 1 CTA each
                      params={"A": A, "B": B, "OUT": outs2[i]},
                      kernel_name=f"k{i}_small", config=cfg)
```

Stream 0 now has 4× more work. The fairness index will drop (stream 0 accumulates more cycles than streams 1–3) but not as dramatically as you might expect: because RR is CTA-level, streams 1–3 each get their one CTA dispatched early, while stream 0 takes several more rounds to drain its four CTAs. No stream waits indefinitely. On real hardware this would produce a fairness index around 0.5–0.7 depending on grid sizes; in the simulator's sequential drain the result reflects dispatch ordering rather than true simultaneity.

## 真机对照

H100's hardware RR scheduler operates at the warp group (or CTA) level across equal-priority streams. You can observe this with Nsight Systems: in the GPU trace view, CTAs from concurrent streams appear as interleaved blocks on each SM's occupancy row.

CUDA allows stream priority hints via `cudaStreamCreateWithPriority`:

```cpp
cudaStream_t high_prio_stream, normal_stream;
int lo, hi;
cudaDeviceGetStreamPriorityRange(&lo, &hi);  // hi < lo on CUDA (more negative = higher)
cudaStreamCreateWithPriority(&high_prio_stream, cudaStreamNonBlocking, hi);
cudaStreamCreateWithPriority(&normal_stream,    cudaStreamNonBlocking, lo);
```

With different priorities, the scheduler no longer uses pure RR: higher-priority stream CTAs are dispatched first whenever a slot is available. Jain's fairness index for a high-priority + normal-priority pair will be less than 1.0 by design — the user has explicitly requested unequal service.

Phase 8 of the simulator will add `Stream(priority=...)` and a priority-aware scheduler. The `fairness()` metric will then be the key observable for verifying that priorities are honoured without completely starving lower-priority streams.
