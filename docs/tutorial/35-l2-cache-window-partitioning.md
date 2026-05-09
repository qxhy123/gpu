# Chapter 35 — L2 Cache Window Partitioning

## Protecting Critical-Stream Data in L2

The L2 cache is shared by all SM clusters on the GPU. When multiple concurrent streams access global memory, their cache lines compete for the same L2 sets. A low-priority background stream with a large working set can evict the hot data belonging to a high-priority critical stream, forcing expensive cache misses at exactly the wrong time.

Phase 8 addresses this with **L2 set-window partitioning**: a stream can reserve a contiguous range of L2 sets for its exclusive use. Lines loaded by that stream are installed into its reserved window, and the eviction policy gives those lines extra protection — they are the last candidates for eviction and are never displaced by traffic from other streams.

## The API

```python
stream = Stream(priority="high")
stream.set_l2_window(start_set=0, n_sets=32)
```

`set_l2_window` records the window parameters on the `Stream` object:

```python
# gpusim/api.py (simplified)
def set_l2_window(self, *, start_set: int, n_sets: int) -> None:
    self.l2_window = (start_set, n_sets)
```

When `Device.run_streams` initializes the L2 cache, it calls `L2Cache.register_stream_window(stream_id, start_set, n_sets)` for each stream that has a window configured. Inside the cache, each `L2Line` carries two new fields: `owner_stream_id` (set when the line is installed) and `in_window` (True if the installing stream has a registered window). The eviction policy checks `in_window` before choosing a victim: lines outside any window are evicted first.

**Honest implementation note.** Phase 8 ships the full L2 window API, the `L2Line.owner_stream_id / in_window` fields, the window registration call, and the window-aware eviction policy. The data plumbing that feeds `l2_window_hit_rate` and `l2_window_protection_efficiency` with per-stream hit/miss counters is **Phase 9 work**. In Phase 8, these methods return placeholder values derived from available trace data. The API surface is stable, and the eviction protection logic is functional; the metric accuracy will improve when Phase 9 wires the per-line access counters through to the analysis layer.

## 看模拟器

Run the L2 window demo:

```bash
python examples/l2_window_demo/run.py
```

```python
s_high = Stream(priority="high")
s_high.set_l2_window(start_set=0, n_sets=32)   # 32 sets reserved

s_low  = Stream(priority="low")                # no window — uses evictable sets

s_high.launch(ptx_src=ptx, kernel_name="critical",    ...)
s_low.launch(ptx_src=ptx,  kernel_name="background",  ...)

multi_res = gpusim.synchronize(streams=[s_high, s_low], config=cfg)
```

After synchronize, query the window metrics:

```python
hit_rate = multi_res.l2_window_hit_rate()
print(f"L2 window hit rate: {hit_rate}")

efficiency = multi_res.l2_window_protection_efficiency()
print(f"L2 window protection efficiency: {efficiency:.3f}")
```

`l2_window_hit_rate()` returns a dict mapping stream ID to hit-rate fraction within that stream's window region. `l2_window_protection_efficiency()` returns a scalar measuring how often the window-protected lines survived eviction pressure from other streams. Both metrics will improve in precision with Phase 9 instrumentation.

## 改一改

**Smaller window — less protection.** Reduce `n_sets` from 32 to 4:

```python
s_high.set_l2_window(start_set=0, n_sets=4)   # only 4 sets reserved
```

With a smaller window, fewer of the critical stream's lines benefit from eviction protection. Lines that spill outside the 4-set window are treated as ordinary evictable lines and can be displaced by the background stream's traffic. The `l2_window_protection_efficiency` metric should decrease compared to the 32-set case.

Try setting `n_sets=0` (effectively disabling the window) and confirm that the high-priority stream's lines are as vulnerable to eviction as those of the low-priority stream — the two streams compete on equal terms.

**Overlapping windows.** The register call does not currently validate that windows are non-overlapping. Setting two streams' windows to the same range will result in both streams tagging their lines as `in_window`, which doubles the pressure within that region. This is a configuration error, not a safety guarantee.

## 真机対照

CUDA exposes L2 cache partitioning via `cudaStreamSetAttribute` with the `cudaStreamAttributeAccessPolicyWindow` attribute:

```cpp
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = (void*)ptr;
attr.accessPolicyWindow.num_bytes = window_size_bytes;
attr.accessPolicyWindow.hitRatio  = 1.0f;          // fully persistent
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

The `hitProp = cudaAccessPropertyPersisting` tells the hardware L2 to treat lines in the specified address range as persistent — they resist eviction from competing traffic. `missProp = cudaAccessPropertyStreaming` marks all other lines as short-lived, giving the hardware permission to evict them first.

The simulator's `set_l2_window` abstracts over the address-range-based hardware API, mapping instead to cache-set ranges for simplicity. This is sufficient to model the key tradeoff — window size versus protection strength — without requiring the simulator to track virtual-to-physical address mappings for every kernel.
