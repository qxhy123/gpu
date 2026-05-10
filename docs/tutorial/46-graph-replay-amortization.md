# Chapter 46 — Graph Replay Amortization ⭐

## The Overhead Problem

Every kernel launch carries overhead: the runtime must validate arguments, schedule the kernel on the hardware queue, and handle synchronization bookkeeping. For short kernels that run in a few microseconds, this overhead is non-trivial — on real hardware, a CUDA kernel launch costs roughly 5–15 µs of CPU overhead, which can exceed the GPU execution time for small workloads.

Graph replay eliminates most of this overhead. Once a graph is instantiated, the execution plan is pre-validated and pre-optimized. Replaying it is nearly free on the CPU side: the driver submits the entire plan in a single call. For workloads that repeat the same kernel sequence — training loop bodies, inference batches, DSP filter chains — graphs provide a consistent cycle-count across replays, with no per-launch jitter.

## Replay Is Deterministic

The simulator makes the determinism property explicit and easy to verify. Because the graph DAG is fixed at instantiation time and the simulator has no hardware scheduling noise, every `exec.launch()` call returns exactly the same cycle count:

```python
from gpusim.api import Stream
from gpusim.config.loader import load_default

cfg = load_default()
s = Stream()
s.begin_capture()
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"A": A, "B": B, "OUT": OUT},
          kernel_name="vec_add", config=cfg)
g = s.end_capture()
exec = g.instantiate(cfg)

cycles = [exec.launch() for _ in range(5)]
print(cycles)
# [<N>, <N>, <N>, <N>, <N>] — all identical
```

On real hardware, consecutive graph replays show very low jitter (typically < 1%) compared to individual kernel launches (5–20% jitter from CUDA driver scheduling), making graphs the preferred approach for benchmarking.

## The graph_replay_perf Demo

The `examples/graph_replay_perf/run.py` demo captures a single vector-add kernel and replays it 5 times, printing the per-replay cycle count and average:

```bash
python examples/graph_replay_perf/run.py
```

Expected output:
```
Replay cycles per launch: [<N>, <N>, <N>, <N>, <N>]
Average cycles/replay: <N>.0
Final OUT[0:4]: [<values>]
```

The demo demonstrates three properties simultaneously:

1. **Determinism**: all five cycle counts are identical.
2. **Functional correctness**: `OUT` accumulates across replays because `params` holds a reference to the live NumPy array. Each replay reads the current array values and writes new ones.
3. **Amortization**: the `GraphExec` object is created once and reused — no re-validation cost on replays 2–5.

## 看模拟器

**用 `graph_replay_amortization` 量化加速比：**

The `graph_replay_amortization` metric compares graph replay cost against a single-kernel baseline. It answers: "how much cheaper is graph replay than issuing each kernel individually?"

```python
import pandas as pd
from gpusim.trace.recorder import Recorder
from gpusim.analysis.metrics import graph_replay_amortization

cfg = load_default()
rec = Recorder()
s = Stream()
s.begin_capture()
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"A": A, "B": B, "OUT": OUT},
          kernel_name="vec_add", config=cfg)
g = s.end_capture()
exec = g.instantiate(cfg)
exec._recorder = rec  # attach recorder

for _ in range(5):
    exec.launch()

# Build a DataFrame from recorded graph launches
rows = [{"start_cycle": ev.start_cycle, "end_cycle": ev.end_cycle}
        for ev in rec.graph_launch_events]
df = pd.DataFrame(rows)

single_kernel_cycles = 256  # baseline: one kernel launched individually
result = graph_replay_amortization(df, single_kernel_cycles)

print(f"Avg cycles/replay: {result['avg_cycles_per_replay']:.1f}")
print(f"Amortization factor: {result['amortization_factor']:.2f}x")
```

In the simulator, `amortization_factor` is always near 1.0 because there is no driver overhead model — the cycle count reflects pure execution time. On real hardware, `amortization_factor` for short kernels is often 5–20x, reflecting the elimination of per-launch driver work.

## 改一改

**把 replay 次数增加到 100 次，看 OUT 的变化：**

Change the replay count to 100 and observe how the output array drifts:

```python
exec = g.instantiate(cfg)
for i in range(100):
    exec.launch()

print(f"After 100 replays, OUT[0]: {OUT[0]:.1f}")
# Each replay adds B[0] to OUT[0], so OUT[0] = 100 * (A[0] + B[0])
```

This reveals an important graph semantic: **graphs hold references, not copies**. The `params` dict in each graph node points to the same NumPy arrays that were live during capture. Every replay reads the current values of those arrays and writes new results back. Modifying the arrays between replays changes the computation's inputs without rebuilding or re-instantiating the graph.

This is both useful (SGD training step replays in Chapter 47) and dangerous (accidental aliasing). The CUDA runtime equivalent is that graph node parameters hold device pointers; changing the data at those addresses between replays changes the effective inputs.

## 真机对照

PyTorch's `torch.cuda.graphs` module exposes the same amortization benefit:

```python
import torch

# Warmup pass (required before capture on real hardware)
with torch.cuda.stream(torch.cuda.Stream()):
    y = model(x)

# Capture into a CUDA graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y = model(x)  # captured — no actual execution

# Replay 100 times — CPU overhead is constant, not proportional to replay count
for i in range(100):
    graph.replay()
```

Key differences from the simulator:

| Behavior | Simulator | PyTorch / CUDA runtime |
|---|---|---|
| **Capture warmup** | Not required | Required (to allocate memory, JIT-compile) |
| **Per-replay CPU cost** | Zero (no driver model) | ~few µs (one `cudaGraphLaunch`) |
| **Amortization factor** | ~1x (no driver overhead) | 5–20x for short kernels |
| **Output determinism** | Always identical cycles | Near-deterministic (< 1% jitter) |
| **Memory aliasing** | NumPy references | CUDA device pointers |

The real hardware benefit is largest for workloads where the per-kernel CPU overhead (driver validation, scheduling) is a significant fraction of GPU execution time. Transformer attention layers with small sequence lengths and CNN layers with small spatial dimensions are prime candidates.

## Why This Chapter Is ⭐

Graph replay amortization is the primary reason CUDA Graphs exist. Chapters 44 and 45 showed how to build graphs; this chapter shows *why* you would bother. The amortization factor quantifies the benefit in a single number, making it easy to decide whether graphing a particular workload is worthwhile.

Chapter 47 will apply graph replay to the most important recurring workload in deep learning: the SGD training step.
