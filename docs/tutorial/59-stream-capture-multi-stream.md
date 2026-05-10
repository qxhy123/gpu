# 59 · Stream Capture Multi-Stream — CaptureSession + Cross-Stream Edges

Phase 15's `CaptureSession` lets two or more streams record into the same
captured `Graph`. Cross-stream `record(ev)` / `wait(ev)` pairs become Graph
dependency edges automatically.

## What the example does

```python
from gpusim.graph.capture_session import CaptureSession
from gpusim.api import Stream, Event

sess = CaptureSession()
sA, sB = Stream(), Stream()
sess.attach(sA); sess.attach(sB)
ev = Event()

sA.launch(...)               # k1 on stream A
sA.record(ev)                # event node, registered as source in session
sB.wait(ev)                  # event node + edge from k1's record node to here
sB.launch(...)               # k2 on stream B (chained from wait)
sA.launch(...)               # k3 on stream A (chained from record)

g = sess.end()               # detaches both streams, returns shared Graph
# g has 5 nodes; 3+ edges including the cross-stream record→wait edge.
```

## 看模拟器

`CaptureSession.attach(stream)` sets the stream's `_captured_graph` to the
session's shared Graph and registers the stream in `sess.streams`. During
capture, when a stream calls `record(ev)`, it appends an event node AND
registers `(id(ev) -> node_id)` in the session's `_event_source_node` table.
When a stream calls `wait(ev)`, it appends an event node, then looks up the
event's source node in the session table — if found, it adds a dependency edge.

`sess.end()` detaches every attached stream (clears their `_captured_graph` and
`_capture_session`) and returns the shared `Graph`.

## 改一改

- Wait on an event that was never recorded inside the session — the wait node
  is appended but no cross-edge is created (the wait becomes a no-op edge).
- Attach the same stream twice — `RuntimeError("already attached")`.
- Replay the captured graph multiple times: `g.instantiate(cfg).launch()`
  re-executes the entire DAG including the cross-stream edges.

## 真机对照

Real CUDA: cross-stream capture relies on the runtime detecting event sync
between streams during capture mode. PyTorch's `torch.cuda.graph()` context
manager wraps multiple streams under one captured graph the same way — the
explicit `CaptureSession` in Phase 15 makes the merge boundary visible instead
of relying on global stream-capture state.
