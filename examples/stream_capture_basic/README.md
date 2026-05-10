# stream_capture_basic — Phase 15

Capture a 3-kernel sequence on a stream into a Graph, then replay it 5 times.

Demonstrates:
- `Stream.begin_capture()` / `Stream.end_capture()` — Phase 11 entry; Phase 15 adds mode validation, error handling, `is_captured` flag, and trace events.
- The captured Graph is reusable: `g.instantiate(cfg)` yields a `GraphExec` that can `.launch()` repeatedly.
- Each captured `Stream.launch` becomes a kernel node; consecutive ops within one stream chain via dependency edges.

## Run
```bash
python run.py
```

Expected output:
```
Captured: 3 nodes, 2 edges, is_captured=True
After 5 replays * 3 kernels * 1 inc per thread: OUT.sum() = 480 (expected 480)
```
