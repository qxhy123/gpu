# Chapter 10 — HBM Bandwidth Saturation

## Why coalescing is fast — the real answer

Chapter 03 said: "coalesced access is fast because it uses 1 transaction instead of 32." Phase 2 reveals the deeper reason: coalesced access distributes load across all HBM channels simultaneously. Strided access funnels everything into fewer channels, creating a queue.

## HBM architecture in the simulator

The Phase 2 HBM model has:

```yaml
hbm:
  channels: 8           # independent command queues
  banks_per_channel: 16 # row-buffer banks per channel
  row_size_bytes: 4096  # 4 KB DRAM row
  row_hit_latency: 10   # cycles for same-row access
  row_miss_latency: 30  # cycles to activate a new row
```

The 8 channels are physically distinct on the HBM die. Each channel has its own command queue and row-buffer array. Requests to **different channels run in parallel**. Requests to the **same channel serialize** — the second request waits until the first is served.

## Address layout and channel assignment

The simulator assigns channels using bits [9:7] of the byte address:

```
Byte address bits:
  [6:0]  = offset within cache line (128 B)
  [9:7]  = channel index (0–7)
  [14:10] = column within row
  [18:15] = bank index
  [30:19] = row index
```

A sequential stream of cache lines (addresses 0, 128, 256, ...) cycles through all 8 channels: line 0 → channel 0, line 1 → channel 1, ..., line 7 → channel 7, line 8 → channel 0 again. This is why coalesced access (1 cache line per warp) efficiently uses all channels when multiple warps run concurrently.

## The bandwidth saturation experiment

`examples/bw_saturation_demo/` streams data from HBM with varying amounts of concurrency. Each thread reads one float from `A` and writes it to `OUT`. The only work is the memory access itself.

```bash
python examples/bw_saturation_demo/run.py
```

Output:

```
# low concurrency (2 CTAs, 1 warp each):
  cycles=68
# high concurrency (64 CTAs, 1 warp each):
  cycles=229
```

**Low concurrency (2 CTAs × 1 warp × 32 threads = 64 threads):**
- Thread 0 accesses element 0 (byte 0, channel 0), thread 1 → element 1 (byte 4, channel 0), ..., thread 31 → element 31 (byte 124, channel 0). All 32 threads coalesce to 1 cache line → 1 HBM request to channel 0.
- CTA 1 accesses elements 32–63 → 1 cache line → channel 0.
- Total: 4 HBM requests (2 loads + 2 stores), channels 0–3 each used once.
- Channel utilization: ~44% for channels 0–3, 0% for channels 4–7.
- cycles=68.

**High concurrency (64 CTAs × 1 warp × 32 threads = 2048 threads):**
- 64 CTAs × 2 requests each = 128 HBM requests, spread across all 8 channels.
- With 128 requests / 8 channels = 16 requests per channel, each channel queue fills up. The second request to channel 0 must wait ~30 cycles (row-miss latency) for the first to complete. The 16th request waits much longer.
- Channel utilization: ~79% for all 8 channels.
- queue_wait statistics: min=0, max=17, mean=5.9 cycles.
- cycles=229.

The 3.4× increase in cycle count (68 → 229) with 32× more threads illustrates channel saturation: doubling the threads does not double the throughput once channels are fully loaded.

## Reading the channel utilization metric

`res.cache_metrics['channel_util']` is a list of 8 floats, one per channel. The value is `(total_cycles_channel_busy) / (total_simulation_cycles)`.

For the low-concurrency run:

```python
channel_util = [0.441, 0.441, 0.441, 0.441, 0.0, 0.0, 0.0, 0.0]
```

Only channels 0–3 are active (the 64 threads' addresses span 4 cache lines × 4 channels). Channels 4–7 are idle.

For the high-concurrency run:

```python
channel_util = [0.786, 0.786, 0.786, 0.786, 0.786, 0.786, 0.786, 0.786]
```

All 8 channels are busy for ~79% of the simulation time. A fully saturated channel would show 1.0. Getting from 79% to 100% would require more concurrent warps — the single-SM 64-warp limit prevents it.

## "Memory-bound" and "compute-bound"

A kernel is **memory-bound** when the bottleneck is HBM bandwidth: the compute units (FP32, INT) sit idle waiting for data. A kernel is **compute-bound** when the arithmetic units are fully occupied and memory latency is hidden by sufficient warp parallelism.

From the metrics:
- channel_util close to 1.0 for all channels → memory-bound.
- channel_util near 0 but high compute throughput → compute-bound (not visible in bw_saturation_demo since it has zero arithmetic).

The bw_saturation_demo at 64 CTAs is approaching the memory-bound regime: channels at 79% utilization and climbing.

## Channel queue wait — measuring queuing delay

`res.hbm_events_df['queue_wait']` records, for each HBM request, how many cycles the request spent waiting in the channel queue before being served. For the high-concurrency run:

```
min=0, max=17, mean=5.9 cycles
```

The maximum wait of 17 cycles means some requests waited half a row-miss latency (30 cycles) in the queue before being served. At fully saturated bandwidth, queue_wait grows proportionally to the number of concurrent requests per channel.

## 改一改 — Double the channels

In `default_hopper.yaml`:

```yaml
hbm:
  channels: 16
```

Re-run the high-concurrency demo. With 16 channels instead of 8, each channel receives half as many requests. The cycle count for the 64-CTA case should drop from ~229 to ~120. Channel utilization should halve to ~40%.

This directly illustrates the relationship: HBM throughput scales linearly with the number of channels, all else equal. Real H100s have 6 HBM3 stacks × 128-bit bus each, physically equivalent to roughly 12–16 simulator-style channels.

## 改一改 — Force channel imbalance with stride

Change the demo to use a stride that maps all threads to the same channel. With `STRIDE=8` (8-element = 1 cache-line separation), consecutive threads in the same warp access lines at channels 0, 1, 2, ..., 7, 0, 1, ... (cycling through all channels). With `STRIDE=1` (sequential), all 32 threads are in 1 cache line → 1 channel request.

Now change the launch to use 64 CTAs but modify the kernel so each CTA accesses elements from the same channel 0 only (e.g., use global thread ID × 8 as the address, skipping 7 channels). Channel 0 becomes the bottleneck; channels 1–7 are idle. You'll see channel_util = [~1.0, 0, 0, ..., 0].

## 真机对照 — Real-machine comparison

_No reference fixture committed (requires real-GPU run). The real H100 SXM5 has 80 GB HBM3 at 3.35 TB/s, using 6 HBM stacks with a 128-bit bus at 3.2 GT/s per pin. That is equivalent to approximately 16 independent 200 GB/s channels. The simulator models 8 channels at a simplified latency model. Absolute bandwidth numbers differ significantly (3.35 TB/s real vs. ~100 GB/s effective in the simulator's single-SM model), but the saturation dynamics — queue_wait growing with concurrency, all-channels-busy indicating memory-bound — are structurally identical._
