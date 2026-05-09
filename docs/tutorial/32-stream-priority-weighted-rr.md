# Chapter 32 — Stream Priority and Weighted Round-Robin

## Three Priority Levels

Phase 8 adds three priority levels to streams: `"high"`, `"normal"`, and `"low"`. You set the priority when constructing a stream:

```python
from gpusim.api import Stream

s_high   = Stream(priority="high")
s_normal = Stream(priority="normal")   # default
s_low    = Stream(priority="low")
```

The priority field is validated at construction time — passing an unknown string raises `ValueError`.

## Weighted Round-Robin (4:2:1)

Priority maps to a dispatch weight. Each cycle, the `ConcurrentStreamScheduler` iterates over all active streams and dispatches up to `weight` CTAs per stream:

| Priority | Default weight |
|----------|---------------|
| high     | 4             |
| normal   | 2             |
| low      | 1             |

A high-priority stream therefore receives **four times** as many CTA dispatch slots per cycle as a low-priority stream, and twice as many as a normal-priority stream. This is a **weighted round-robin (WRR)** policy rather than strict priority: low-priority streams continue to make progress even when high-priority streams are saturated. No stream starves.

The weights come from `cfg.scheduler.priority_weights`, a dict stored in `CtaSchedulerConfig`:

```python
# gpusim/config/schema.py
@dataclass
class CtaSchedulerConfig:
    cta_policy: str = "rr"
    priority_weights: dict = field(
        default_factory=lambda: {"high": 4, "normal": 2, "low": 1}
    )
```

## 看模拟器

Run the priority demo:

```bash
python examples/priority_demo/run.py
```

The demo launches the same vector-add kernel on three streams of different priority:

```python
s_high   = Stream(priority="high")
s_normal = Stream(priority="normal")
s_low    = Stream(priority="low")

for s, out, name in [(s_high, out_h, "kh"),
                     (s_normal, out_n, "kn"),
                     (s_low, out_l, "kl")]:
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
             params={"A": A, "B": B, "OUT": out},
             kernel_name=name, config=cfg)

multi_res = gpusim.synchronize(streams=[s_high, s_normal, s_low], config=cfg)
```

After synchronize, query the dispatch-share breakdown:

```python
share = multi_res.priority_dispatch_share()
print(share)
# {"high": 0.57, "normal": 0.29, "low": 0.14}  (approximate for 4:2:1 weights)
```

`multi_res.priority_dispatch_share()` returns a dict keyed by priority string. Each value is the fraction of total kernel-launch events that were dispatched to that priority level. The 4:2:1 weight ratio produces shares of roughly 4/7 ≈ 0.57, 2/7 ≈ 0.29, and 1/7 ≈ 0.14 for the symmetric three-stream case.

You can also inspect the per-stream cycle totals to confirm that high-priority work finishes first when SM slots are limited:

```python
for sid, launches in multi_res.streams.items():
    total = sum(r.metrics["cycles"] for r in launches)
    print(f"  stream {sid}: {total} cycles")
```

## 改一改

**Configure custom priority weights.** The 4:2:1 ratio is a simulator default, not a law. Try a more extreme ratio to see the effect on the dispatch share:

```python
cfg = load_default()
cfg.scheduler.priority_weights = {"high": 8, "normal": 2, "low": 1}
```

Rebuild and rerun the demo. The high-priority share should increase toward 8/11 ≈ 0.73. Watch the total cycle count: if you give the high-priority stream a massive weight advantage and it happens to have the smallest grid, it completes so quickly that the weight advantage stops being useful — cycle count then converges back toward the sequential case.

Try the opposite: set all weights to 1 (pure round-robin, Chapter 30 behavior) and confirm that the fairness index approaches 1.0 while the dispatch-share dict becomes `{"high": 0.33, "normal": 0.33, "low": 0.33}`.

## 真机対照

CUDA exposes stream priorities via `cudaStreamCreateWithPriority`:

```cpp
int lo, hi;
cudaDeviceGetStreamPriorityRange(&lo, &hi);
// hi < lo on CUDA: more-negative = higher priority

cudaStream_t s_high, s_low;
cudaStreamCreateWithPriority(&s_high, cudaStreamNonBlocking, hi);
cudaStreamCreateWithPriority(&s_low,  cudaStreamNonBlocking, lo);
```

On H100, the hardware scheduler uses a similar weighted-priority policy at the GPC level. The exact weights are not published, but empirical measurements with Nsight Compute suggest a ratio comparable to 2:1 or 4:1 between adjacent priority levels. The simulator's 4:2:1 default is consistent with those observations.

CUDA priority values are integers in `[hi, lo]` with more-negative meaning higher priority. The simulator maps this to three named levels for simplicity; a future iteration could support arbitrary integer priorities to match the CUDA API exactly.
