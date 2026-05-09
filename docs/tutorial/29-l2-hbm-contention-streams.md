# Chapter 29 — L2/HBM Contention Across Streams

## Shared Bandwidth, Multiple Streams

When two kernels run concurrently on the GPU they share every level of the memory subsystem: L2 cache capacity and bandwidth, HBM channel bandwidth, and the crossbar that connects the SM array to L2. There is no per-stream memory partition. A warp on SM 4 issuing a `ld.global` for stream 0 competes with a warp on SM 7 issuing a `ld.global` for stream 1 for exactly the same L2 tag-comparison slots and HBM row-buffer state.

Two concrete sharing mechanisms matter most:

**L2 set collisions.** The L2 is a set-associative cache. Both streams map their addresses into the same set-index bits. If both kernels touch enough unique cache lines, they thrash each other out of L2, reducing the effective hit rate for both — even if their working sets would fit individually.

**HBM channel arbitration.** HBM is organized into independent channels (H100: 16 stacks × 8 channels = 128 channels). Requests from both streams are multiplexed across the same channel queues. High concurrency from two streams saturates the channel faster than either stream alone; the result is increased HBM latency for both.

The interplay is asymmetric: if one stream is write-heavy and the other read-heavy, HBM write-combining can partially decouple them. If both streams read from overlapping address ranges, the L2 can actually serve both from a single fill — a rare benefit of sharing.

## 走通 l2_contention_2stream

```bash
python examples/l2_contention_2stream/run.py
```

The demo uses a single kernel entry `writer` that writes `1` to a slice of a shared `uint32` buffer. Two streams use different `OFFSET` values to target adjacent 32-element regions of the same 64-element allocation:

```ptx
ld.param.u64 %rd0, [OUT];
ld.param.u32 %r0, [OFFSET];   // 0 for s0, 32 for s1
mov.u32 %r1, %tid.x;
add.u32 %r2, %r1, %r0;
shl.b32 %r3, %r2, 2;          // byte offset = (tid + OFFSET) * 4
cvt.u64.u32 %rd1, %r3;
add.u64 %rd2, %rd0, %rd1;
mov.u32 %r4, 1;
st.global.u32 [%rd2], %r4;    // write 1 to this slot
```

Stream 0 writes elements `[0..31]` and stream 1 writes elements `[32..63]` of the same `SHARED` array. At 4 bytes each, the two regions together span a single 256-byte L2 cache line on H100. Both streams therefore write to the same L2 cache line — a direct demonstration of intra-line contention.

The run script:

```python
SHARED = np.zeros(64, dtype=np.uint32)
s0 = Stream()
s1 = Stream()
s0.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"OUT": SHARED, "OFFSET": 0}, kernel_name="writer_low", config=cfg)
s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"OUT": SHARED, "OFFSET": 32}, kernel_name="writer_high", config=cfg)
multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
print(multi_res.stream_summary())
```

## 看模拟器

```python
print(multi_res.stream_summary())
```

The `stream_summary()` output includes the `l2_bandwidth_per_stream` metric, which reports the number of bytes each stream pushed through the L2 interface. When both streams target the same L2 set (as here), the per-stream bandwidth is lower than when they use disjoint regions — the L2 queue serializes conflicting requests.

In the HTML report, navigate to **§28 L2/HBM Breakdown Table**. The table rows list each stream's L2 hit count, L2 miss count, and HBM fill count. With adjacent writes into the same cache line, you will see that one stream's stores trigger L2 invalidations that the other stream's stores then have to re-fetch — the hallmark of false sharing across streams.

## 改一改

**Separate the gmem regions to eliminate contention.** Increase `OFFSET` for stream 1 to place its writes far from stream 0's region:

```python
s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"OUT": SHARED, "OFFSET": 1024},  # was 32
          kernel_name="writer_high", config=cfg)
```

With a gap of 1024 elements (4 KiB), the two streams' writes are guaranteed to land in different L2 cache sets and different HBM row buffers. Rerun and compare `l2_bandwidth_per_stream`: both streams should now achieve higher per-stream bandwidth, and the L2 miss count for each stream should drop because there is no cross-stream eviction pressure.

This is the practical lesson: when designing multi-stream workloads on real hardware, pad output buffers so that each stream's working set maps to distinct L2 sets. A common heuristic is to align stream-private buffers to multiples of the L2 associativity × set size (often 4 MB on A100/H100) to guarantee independent set usage.

## 真机对照

CUDA 11.2 introduced the **L2 cache window** API (`cudaStreamSetAttribute` with `cudaStreamAttributeAccessPolicyWindow`) to give individual streams a reservation in the L2's persisting-data region. A stream can pin its working set into a fraction of L2 with a `hitRatio` hint, preventing other streams from evicting it:

```cpp
cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.base_ptr    = my_buffer;
attr.accessPolicyWindow.num_bytes   = 1ULL << 20;  // 1 MB
attr.accessPolicyWindow.hitRatio    = 0.6f;
attr.accessPolicyWindow.hitProp     = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp    = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

This is the hardware-side answer to the contention problem explored in this chapter: instead of hoping that two streams land in different sets by accident, the API lets you explicitly protect a region. Profile with `ncu` → **L2 Hit Rate** to confirm that the persisting window reduces cross-stream evictions.

The simulator's `l2_bandwidth_per_stream` metric is the conceptual analog: it lets you observe the bandwidth each stream achieves and reason about whether contention is limiting throughput, without requiring access to real hardware.
