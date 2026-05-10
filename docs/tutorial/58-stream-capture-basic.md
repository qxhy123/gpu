# 58 · Stream Capture Basic — Convert Imperative Stream Code Into a Graph

Phase 11 added `Stream.begin_capture()` / `Stream.end_capture()` as a minimal
capture entry. Phase 15 hardens it: `mode="global"` (the only supported mode)
must be passed correctly, double-`begin_capture` raises `RuntimeError`,
`end_capture` without `begin_capture` raises, and captured graphs carry
`is_captured=True` so analysis can distinguish them from hand-built ones.

This chapter walks through the simplest pattern: capture three kernel launches
into one Graph, then replay the Graph five times.

## What the example does

```python
from gpusim.api import Stream
from gpusim.config.loader import load_default
import numpy as np, pathlib

OUT = np.zeros(32, dtype=np.uint32)
cfg = load_default()
ptx = pathlib.Path("kernel.ptx").read_text()

s = Stream()
s.begin_capture()                                     # mode="global" by default
for _ in range(3):
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": OUT}, kernel_name="inc", config=cfg)
g = s.end_capture()
print(g.is_captured, len(g.nodes), len(g.edges))     # True 3 2

exec = g.instantiate(cfg)
for _ in range(5):
    exec.launch()
print(OUT.sum())                                      # 480 = 5 replays * 3 kernels * 32 threads
```

## 看模拟器

Each `Stream.launch` during capture appends a kernel node and chains it to the
previous capture node (intra-stream ordering becomes a graph edge). The captured
Graph is fully reusable — `g.instantiate(cfg)` builds a `GraphExec` whose
`launch()` walks the topo order and dispatches each node, just as if you had
called `Graph.add_kernel_node` by hand.

The trace records `StreamCaptureBegin(stream_id, cycle=0)` at `begin_capture`
and `StreamCaptureEnd(stream_id, cycle=0, captured_node_count=N)` at
`end_capture` (when a recorder is attached to the stream via `s._recorder = ...`).
Analytics can then count captures and total captured nodes via the Phase 15
metrics `stream_capture_count(recorder)` and `captured_node_count(recorder)`.

## 改一改

- Add `s.record(ev)` between launches: a third event-type node appears in the
  captured graph, with chaining edges before and after it.
- Try `s.begin_capture(mode="thread")` — the call raises `ValueError` because
  Phase 15 supports only `"global"`.
- Try calling `s.begin_capture()` twice without an `end_capture` between —
  raises `RuntimeError("already capturing")`.

## 真机对照

Real CUDA: `cudaStreamBeginCapture(stream, mode)` with three modes (global,
thread-local, relaxed). `cudaStreamEndCapture(stream, &graph)` returns the
recorded graph. PyTorch and JAX wrap this for AOT-compiled training step graphs
and execute them via `cudaGraphLaunch` on every iteration. Phase 15 replicates
the single-stream API; multi-stream capture (chapter 59) requires
`CaptureSession` because pure CUDA's notion of cross-stream capture relies on
event-driven edges that are auto-discovered, while gpusim makes the session
explicit for clarity.
