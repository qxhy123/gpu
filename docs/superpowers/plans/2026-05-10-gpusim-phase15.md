# gpusim Phase 15 Implementation Plan — Stream Capture API + Conditional Graph Nodes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Phase 11's basic `Stream.begin_capture/end_capture` into a complete capture API (mode validation, error handling, full op coverage, multi-stream sessions) and add conditional + while graph node types.

**Architecture:** Phase 11 already has minimal capture for `Stream.launch` only. Phase 15 (a) adds mode parameter + validation + double-begin/end errors + `StreamCaptureBegin/End` trace events + capture for `Stream.record`/`wait`, (b) introduces `CaptureSession` so multiple streams capture into a single shared `Graph`, and (c) adds two new `GraphNode` types — `conditional` (host-evaluated if/else dispatching to true/false sub-graphs) and `while` (host-evaluated loop body with `max_iterations` cap). Backward-compatible: `Stream.begin_capture()` with no args + no session still produces a single-stream Graph identical to Phase 11.

**Tech Stack:** Python 3, numpy, pytest. Existing modules: `gpusim/api.py` (Stream), `gpusim/graph/{graph,node,exec}.py`, `gpusim/trace/{events,recorder}.py`, `gpusim/analysis/metrics.py`.

---

## File structure

### New files
- `gpusim/graph/capture_session.py` — `CaptureSession` class (M2)
- `examples/stream_capture_basic/` (M1) — 5 files: `__init__.py`, `kernel.ptx`, `reference.py`, `run.py`, `README.md`
- `examples/stream_capture_multi_stream/` (M2) — 5 files
- `examples/graph_conditional_branch/` (M3) — 5 files
- `examples/graph_while_loop/` (M4) — 5 files
- `docs/tutorial/58-stream-capture-basic.md` ... `61-graph-while-loop.md` (M5)
- `tests/unit/core/test_stream_capture_phase15.py` (M1)
- `tests/unit/graph/test_capture_session.py` (M2)
- `tests/unit/graph/test_conditional_node.py` (M3)
- `tests/unit/graph/test_while_node.py` (M4)
- `tests/unit/analysis/test_phase15_metrics.py` (M4)
- `tests/parity/test_stream_capture_basic.py` (M1)
- `tests/parity/test_stream_capture_multi_stream.py` (M2)
- `tests/parity/test_graph_conditional_branch.py` (M3)
- `tests/parity/test_graph_while_loop.py` (M4)
- `tests/microbench/test_phase15_facts.py` (M5)
- `tests/microbench/test_phase15_runtime.py` (M5)
- `tests/parity/test_phase1_14_examples_unchanged.py` (M5, replaces `test_phase1_13_examples_unchanged.py`)

### Modified files
- `gpusim/api.py` (Stream) — `begin_capture(mode="global", session=None)` validation, capture for `record`/`wait`, hook into CaptureSession
- `gpusim/graph/graph.py` — `is_captured` flag, `add_conditional_node`, `add_while_node`
- `gpusim/graph/node.py` — `ConditionalNodeArgs`, `WhileNodeArgs`, `GraphNode` field additions
- `gpusim/graph/exec.py` — `conditional` and `while` branches in `launch()`
- `gpusim/trace/events.py` — `StreamCaptureBegin`, `StreamCaptureEnd`, `ConditionalBranch`, `LoopIteration`
- `gpusim/trace/recorder.py` — recorder methods for the 4 new events
- `gpusim/analysis/metrics.py` — `stream_capture_count`, `captured_node_count`, `conditional_branch_taken_count`, `avg_loop_iterations`
- `README.md` — v15 capabilities

---

# M1 — Stream Capture Core Extension + Basic Example

## Task 1: `Graph.is_captured` flag

**Files:**
- Modify: `gpusim/graph/graph.py`
- Test: `tests/unit/graph/test_capture_session.py` (new file, test belongs here since `is_captured` is set by capture path)

- [ ] **Step 1: Write failing test**

Create `tests/unit/graph/test_capture_session.py`:
```python
def test_graph_is_captured_default_false():
    from gpusim.graph.graph import Graph
    g = Graph()
    assert g.is_captured is False


def test_graph_is_captured_can_be_set():
    from gpusim.graph.graph import Graph
    g = Graph()
    g.is_captured = True
    assert g.is_captured is True
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/unit/graph/test_capture_session.py -v`
Expected: FAIL with `AttributeError: 'Graph' object has no attribute 'is_captured'`

- [ ] **Step 3: Add field to Graph**

Edit `gpusim/graph/graph.py` — add `is_captured: bool = False` to the dataclass:
```python
@dataclass
class Graph:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    _next_id: int = 0
    is_captured: bool = False    # NEW Phase 15
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/unit/graph/test_capture_session.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/graph/graph.py tests/unit/graph/test_capture_session.py
git commit -m "feat(graph): is_captured flag on Graph (Phase 15 prep)"
```

---

## Task 2: `Stream.begin_capture` mode + double-begin/end validation

**Files:**
- Modify: `gpusim/api.py:443-454` (existing `begin_capture` / `end_capture`)
- Test: `tests/unit/core/test_stream_capture_phase15.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/core/test_stream_capture_phase15.py`:
```python
import pytest


def test_begin_capture_default_mode_global_works():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()                    # default mode="global"
    g = s.end_capture()
    assert g is not None


def test_begin_capture_explicit_global_works():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture(mode="global")
    g = s.end_capture()
    assert g is not None


def test_begin_capture_unknown_mode_raises():
    from gpusim.api import Stream
    s = Stream()
    with pytest.raises(ValueError, match="only 'global' capture mode supported"):
        s.begin_capture(mode="thread")


def test_begin_capture_double_begin_raises():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()
    with pytest.raises(RuntimeError, match="already capturing"):
        s.begin_capture()


def test_end_capture_without_begin_raises():
    from gpusim.api import Stream
    s = Stream()
    with pytest.raises(RuntimeError, match="not capturing"):
        s.end_capture()


def test_end_capture_marks_graph_is_captured():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()
    g = s.end_capture()
    assert g.is_captured is True
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/unit/core/test_stream_capture_phase15.py -v`
Expected: 6 tests, mode/error tests fail with `TypeError: begin_capture() got an unexpected keyword argument 'mode'` or no error raised; `is_captured` test fails because Phase 11 path doesn't set it.

- [ ] **Step 3: Update `Stream.begin_capture` and `end_capture`**

Edit `gpusim/api.py` — replace lines 443-454:
```python
    def begin_capture(self, mode: str = "global") -> None:
        """Start recording subsequent .launch into a fresh Graph.
        Phase 11 capture, extended in Phase 15 with mode validation."""
        if mode != "global":
            raise ValueError(
                f"only 'global' capture mode supported in Phase 15, got {mode!r}"
            )
        if self._captured_graph is not None:
            raise RuntimeError(
                f"stream {self.stream_id} is already capturing"
            )
        from gpusim.graph.graph import Graph
        self._captured_graph = Graph()
        self._captured_graph.is_captured = True
        self._capture_last_node = None

    def end_capture(self) -> "Graph":
        """Stop capture; return the recorded Graph. Phase 11 + Phase 15 errors."""
        if self._captured_graph is None:
            raise RuntimeError(
                f"stream {self.stream_id} is not capturing"
            )
        g = self._captured_graph
        self._captured_graph = None
        self._capture_last_node = None
        return g
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/unit/core/test_stream_capture_phase15.py -v`
Expected: 6 passed.

- [ ] **Step 5: Confirm Phase 11 capture still works**

Run: `pytest tests/parity/test_phase1_13_examples_unchanged.py -k graph_capture_from_stream -v`
Expected: PASS (the existing Phase 11 capture example still works because `begin_capture()` defaults to `mode="global"`).

- [ ] **Step 6: Commit**

```bash
git add gpusim/api.py tests/unit/core/test_stream_capture_phase15.py
git commit -m "feat(stream): begin_capture mode validation + double-begin/end errors (Phase 15)"
```

---

## Task 3: `StreamCaptureBegin` / `StreamCaptureEnd` trace events

**Files:**
- Modify: `gpusim/trace/events.py` (add 2 dataclasses)
- Modify: `gpusim/trace/recorder.py` (add init slots + 2 methods)
- Test: `tests/unit/core/test_stream_capture_phase15.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/core/test_stream_capture_phase15.py`:
```python
def test_recorder_records_stream_capture_begin():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.stream_capture_begin(stream_id=3, cycle=0)
    assert len(rec.stream_capture_begin_events) == 1
    ev = rec.stream_capture_begin_events[0]
    assert ev.stream_id == 3
    assert ev.cycle == 0


def test_recorder_records_stream_capture_end():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.stream_capture_end(stream_id=3, cycle=10, captured_node_count=5)
    assert len(rec.stream_capture_end_events) == 1
    ev = rec.stream_capture_end_events[0]
    assert ev.stream_id == 3
    assert ev.cycle == 10
    assert ev.captured_node_count == 5


def test_stream_begin_capture_with_recorder_emits_begin_event():
    from gpusim.api import Stream
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    s = Stream()
    s._recorder = rec
    s.begin_capture()
    assert len(rec.stream_capture_begin_events) == 1
    assert rec.stream_capture_begin_events[0].stream_id == s.stream_id


def test_stream_end_capture_with_recorder_emits_end_event_with_node_count():
    from gpusim.api import Stream
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    s = Stream()
    s._recorder = rec
    s.begin_capture()
    s.end_capture()
    assert len(rec.stream_capture_end_events) == 1
    assert rec.stream_capture_end_events[0].captured_node_count == 0
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/unit/core/test_stream_capture_phase15.py -v -k stream_capture`
Expected: 4 new tests fail with `AttributeError: 'Recorder' object has no attribute 'stream_capture_begin'` or `Stream` has no `_recorder`.

- [ ] **Step 3a: Add event dataclasses**

Append to `gpusim/trace/events.py` (at end of file):
```python
@dataclass(frozen=True)
class StreamCaptureBegin:
    stream_id: int
    cycle: int


@dataclass(frozen=True)
class StreamCaptureEnd:
    stream_id: int
    cycle: int
    captured_node_count: int


@dataclass(frozen=True)
class ConditionalBranch:
    node_id: int
    taken: bool
    cycle: int


@dataclass(frozen=True)
class LoopIteration:
    node_id: int
    iteration: int
    cycle: int
```
(Define all 4 Phase 15 events here in one shot — used in M3/M4 too. We add the dataclasses now to avoid touching `events.py` four separate times.)

- [ ] **Step 3b: Add recorder slots + methods**

Edit `gpusim/trace/recorder.py`:

In `__init__`, append:
```python
        self.stream_capture_begin_events: list = []
        self.stream_capture_end_events: list = []
        self.conditional_branch_events: list = []
        self.loop_iteration_events: list = []
```

At the end of the `Recorder` class, add:
```python
    def stream_capture_begin(self, *, stream_id: int, cycle: int) -> None:
        from gpusim.trace.events import StreamCaptureBegin
        self.stream_capture_begin_events.append(
            StreamCaptureBegin(stream_id=stream_id, cycle=cycle)
        )

    def stream_capture_end(self, *, stream_id: int, cycle: int,
                              captured_node_count: int) -> None:
        from gpusim.trace.events import StreamCaptureEnd
        self.stream_capture_end_events.append(
            StreamCaptureEnd(stream_id=stream_id, cycle=cycle,
                              captured_node_count=captured_node_count)
        )

    def conditional_branch(self, *, node_id: int, taken: bool, cycle: int) -> None:
        from gpusim.trace.events import ConditionalBranch
        self.conditional_branch_events.append(
            ConditionalBranch(node_id=node_id, taken=taken, cycle=cycle)
        )

    def loop_iteration(self, *, node_id: int, iteration: int, cycle: int) -> None:
        from gpusim.trace.events import LoopIteration
        self.loop_iteration_events.append(
            LoopIteration(node_id=node_id, iteration=iteration, cycle=cycle)
        )
```

- [ ] **Step 3c: Add `_recorder` to Stream + emit on begin/end**

Edit `gpusim/api.py`:

In `Stream` dataclass, add field (after `_capture_last_node`):
```python
    _recorder: object | None = None    # NEW Phase 15 — for capture trace events
```

In `begin_capture`, after `self._capture_last_node = None`, append:
```python
        if self._recorder is not None:
            self._recorder.stream_capture_begin(stream_id=self.stream_id, cycle=0)
```

In `end_capture`, before `return g`, capture node count and emit:
```python
        captured_count = len(g.nodes)
        if self._recorder is not None:
            self._recorder.stream_capture_end(
                stream_id=self.stream_id, cycle=0,
                captured_node_count=captured_count,
            )
```

(Full updated `end_capture`:)
```python
    def end_capture(self) -> "Graph":
        if self._captured_graph is None:
            raise RuntimeError(
                f"stream {self.stream_id} is not capturing"
            )
        g = self._captured_graph
        captured_count = len(g.nodes)
        self._captured_graph = None
        self._capture_last_node = None
        if self._recorder is not None:
            self._recorder.stream_capture_end(
                stream_id=self.stream_id, cycle=0,
                captured_node_count=captured_count,
            )
        return g
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/unit/core/test_stream_capture_phase15.py -v`
Expected: 10 passed (6 from Task 2 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add gpusim/trace/events.py gpusim/trace/recorder.py gpusim/api.py tests/unit/core/test_stream_capture_phase15.py
git commit -m "feat(trace): StreamCaptureBegin/End + ConditionalBranch + LoopIteration events + Stream emit hooks (Phase 15)"
```

---

## Task 4: Capture `Stream.record` and `Stream.wait` ops as event nodes

**Files:**
- Modify: `gpusim/api.py` — `Stream.record`, `Stream.wait`
- Test: `tests/unit/core/test_stream_capture_phase15.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/core/test_stream_capture_phase15.py`:
```python
def test_capture_records_event_node_for_record():
    from gpusim.api import Stream, Event
    s = Stream()
    ev = Event()
    s.begin_capture()
    s.record(ev)
    g = s.end_capture()
    assert len(g.nodes) == 1
    assert g.nodes[0].type == "event"
    assert g.nodes[0].event_args.op == "record"
    assert g.nodes[0].event_args.event is ev


def test_capture_records_event_node_for_wait():
    from gpusim.api import Stream, Event
    s = Stream()
    ev = Event()
    s.begin_capture()
    s.wait(ev)
    g = s.end_capture()
    assert len(g.nodes) == 1
    assert g.nodes[0].type == "event"
    assert g.nodes[0].event_args.op == "wait"


def test_capture_chains_kernel_to_record_to_kernel_with_edges():
    """Within a single stream, ordering is captured as edges."""
    from gpusim.api import Stream, Event
    from gpusim.config.loader import load_default
    cfg = load_default()
    ptx = """
.visible .entry k(.param .u64 OUT) {
    .reg .u64 %rd<3>; .reg .u32 %r<3>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 1; st.global.u32 [%rd2], %r2;
    ret;
}
"""
    import numpy as np
    OUT = np.zeros(32, dtype=np.uint32)
    s = Stream()
    ev = Event()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": OUT}, kernel_name="k1", config=cfg)
    s.record(ev)
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": OUT}, kernel_name="k2", config=cfg)
    g = s.end_capture()
    assert len(g.nodes) == 3                    # k1, record, k2
    assert len(g.edges) == 2                    # k1→record, record→k2
    nids = [n.node_id for n in g.nodes]
    assert (nids[0], nids[1]) in g.edges
    assert (nids[1], nids[2]) in g.edges
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/unit/core/test_stream_capture_phase15.py::test_capture_records_event_node_for_record -v`
Expected: FAIL — record/wait don't yet route to capture path.

- [ ] **Step 3: Update `Stream.record` and `Stream.wait`**

Edit `gpusim/api.py`:

`record`:
```python
    def record(self, ev: "Event") -> None:
        """Append a record-marker. In capture mode (Phase 15), records as event node."""
        if self._captured_graph is not None:
            nid = self._captured_graph.add_event_node(event=ev, op="record")
            if self._capture_last_node is not None:
                self._captured_graph.add_dependency(self._capture_last_node, nid)
            self._capture_last_node = nid
            return
        self.pending.append(_RecordMarker(event=ev))
```

`wait`:
```python
    def wait(self, ev: "Event") -> None:
        """Block this stream's future launches. In capture mode (Phase 15), records as event node."""
        if self._captured_graph is not None:
            nid = self._captured_graph.add_event_node(event=ev, op="wait")
            if self._capture_last_node is not None:
                self._captured_graph.add_dependency(self._capture_last_node, nid)
            self._capture_last_node = nid
            return
        self.event_waits.append(ev)
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/unit/core/test_stream_capture_phase15.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/api.py tests/unit/core/test_stream_capture_phase15.py
git commit -m "feat(stream): capture Stream.record + Stream.wait as event nodes (Phase 15)"
```

---

## Task 5: `examples/stream_capture_basic/` + parity test

**Files:**
- Create: `examples/stream_capture_basic/{__init__.py, kernel.ptx, reference.py, run.py, README.md}`
- Create: `tests/parity/test_stream_capture_basic.py`

- [ ] **Step 1: Write failing parity test**

Create `tests/parity/test_stream_capture_basic.py`:
```python
import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "stream_capture_basic"


def test_stream_capture_basic_correctness():
    """Capture 3-kernel sequence + replay 5 times produces 5x output."""
    from gpusim.api import Stream
    from gpusim.config.loader import load_default

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    OUT = np.zeros(32, dtype=np.uint32)
    s = Stream()
    s.begin_capture()
    for _ in range(3):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"OUT": OUT}, kernel_name="inc", config=cfg)
    g = s.end_capture()

    assert g.is_captured is True
    assert len(g.nodes) == 3
    assert len(g.edges) == 2

    exec = g.instantiate(cfg)
    for _ in range(5):
        exec.launch()
    # 5 replays * 3 kernels per replay * 1 increment per thread per kernel = 15
    assert OUT.sum() == 32 * 15
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/parity/test_stream_capture_basic.py -v`
Expected: FAIL — `kernel.ptx` doesn't exist.

- [ ] **Step 3: Create example files**

`examples/stream_capture_basic/__init__.py` — empty file.

`examples/stream_capture_basic/kernel.ptx`:
```
.visible .entry inc(.param .u64 OUT)
{
    .reg .u64 %rd<5>;
    .reg .u32 %r<5>;

    ld.param.u64 %rd0, [OUT];

    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;

    ld.global.u32 %r2, [%rd2];
    add.u32 %r3, %r2, 1;
    st.global.u32 [%rd2], %r3;

    ret;
}
```

`examples/stream_capture_basic/reference.py`:
```python
import numpy as np


def reference(n: int = 32, replays: int = 5, kernels: int = 3):
    return np.full(n, replays * kernels, dtype=np.uint32)
```

`examples/stream_capture_basic/run.py`:
```python
import numpy as np, pathlib
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()

    s = Stream()
    s.begin_capture()
    for _ in range(3):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"OUT": OUT}, kernel_name="inc", config=cfg)
    g = s.end_capture()

    print(f"Captured: {len(g.nodes)} nodes, {len(g.edges)} edges, is_captured={g.is_captured}")

    exec = g.instantiate(cfg)
    for _ in range(5):
        exec.launch()
    print(f"After 5 replays * 3 kernels * 1 inc per thread: OUT.sum() = {OUT.sum()} (expected 480)")


if __name__ == "__main__":
    main()
```

`examples/stream_capture_basic/README.md`:
```markdown
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
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/parity/test_stream_capture_basic.py -v`
Expected: PASS.

Then sanity-run the example:
```bash
python examples/stream_capture_basic/run.py
```
Expected: prints node/edge counts and `OUT.sum() = 480`.

- [ ] **Step 5: Commit**

```bash
git add examples/stream_capture_basic/ tests/parity/test_stream_capture_basic.py
git commit -m "feat(examples): stream_capture_basic — capture 3 kernels, replay 5x (Phase 15 M1)"
```

---

## Task 6: Tag M1

- [ ] **Step 1: Run full non-slow suite**

Run: `pytest -m "not slow" -q`
Expected: ~728-735 passed (Phase 14 baseline 717 + ~13 new Phase 15 M1 unit tests + 1 parity).

- [ ] **Step 2: Tag**

```bash
git tag M1-phase15-complete
```

- [ ] **Step 3: Verify**

```bash
git tag -l 'M1-phase15-complete'
```
Expected: `M1-phase15-complete`

---

# M2 — CaptureSession (Multi-Stream)

## Task 7: `CaptureSession` class

**Files:**
- Create: `gpusim/graph/capture_session.py`
- Test: `tests/unit/graph/test_capture_session.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/graph/test_capture_session.py`:
```python
def test_capture_session_creates_shared_graph():
    from gpusim.graph.capture_session import CaptureSession
    sess = CaptureSession()
    assert sess.graph is not None
    assert sess.graph.is_captured is True
    assert len(sess.streams) == 0


def test_capture_session_attach_stream_records_membership():
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream
    sess = CaptureSession()
    s = Stream()
    sess.attach(s)
    assert s in sess.streams
    assert s._captured_graph is sess.graph
    assert s._capture_session is sess


def test_capture_session_attach_double_raises():
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream
    sess = CaptureSession()
    s = Stream()
    sess.attach(s)
    import pytest
    with pytest.raises(RuntimeError, match="already attached"):
        sess.attach(s)


def test_capture_session_end_returns_graph_and_detaches_streams():
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream
    sess = CaptureSession()
    s1 = Stream()
    s2 = Stream()
    sess.attach(s1)
    sess.attach(s2)
    g = sess.end()
    assert g is sess.graph
    assert s1._captured_graph is None
    assert s2._captured_graph is None
    assert s1._capture_session is None
    assert s2._capture_session is None
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/unit/graph/test_capture_session.py -v`
Expected: FAIL with `ModuleNotFoundError: gpusim.graph.capture_session`.

- [ ] **Step 3: Create CaptureSession**

Create `gpusim/graph/capture_session.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from gpusim.graph.graph import Graph


@dataclass
class CaptureSession:
    """Shared capture state for multiple Streams capturing into one Graph (Phase 15)."""

    graph: Graph = field(default_factory=Graph)
    streams: list = field(default_factory=list)
    _event_source_node: dict = field(default_factory=dict)    # event_id -> node_id

    def __post_init__(self):
        self.graph.is_captured = True

    def attach(self, stream) -> None:
        """Attach a Stream to this session. The Stream now captures into self.graph."""
        if stream in self.streams:
            raise RuntimeError(
                f"stream {stream.stream_id} already attached to this CaptureSession"
            )
        if stream._captured_graph is not None:
            raise RuntimeError(
                f"stream {stream.stream_id} is already capturing standalone"
            )
        self.streams.append(stream)
        stream._captured_graph = self.graph
        stream._capture_last_node = None
        stream._capture_session = self

    def end(self) -> Graph:
        """End the session, detach all streams, return the shared graph."""
        for s in self.streams:
            s._captured_graph = None
            s._capture_last_node = None
            s._capture_session = None
        self.streams = []
        self._event_source_node = {}
        return self.graph

    def register_event_source(self, event_id: int, node_id: int) -> None:
        self._event_source_node[event_id] = node_id

    def lookup_event_source(self, event_id: int):
        return self._event_source_node.get(event_id)
```

- [ ] **Step 4: Add `_capture_session` field to Stream**

Edit `gpusim/api.py` — add field to `Stream` dataclass (after `_recorder`):
```python
    _capture_session: object | None = None    # NEW Phase 15
```

- [ ] **Step 5: Run to confirm pass**

Run: `pytest tests/unit/graph/test_capture_session.py -v`
Expected: 6 passed (2 from Task 1 + 4 new).

- [ ] **Step 6: Commit**

```bash
git add gpusim/graph/capture_session.py gpusim/api.py tests/unit/graph/test_capture_session.py
git commit -m "feat(graph): CaptureSession for multi-stream capture (Phase 15 M2)"
```

---

## Task 8: Cross-stream event-edge translation in CaptureSession

**Files:**
- Modify: `gpusim/api.py` — `Stream.record` + `Stream.wait` use session lookup table
- Test: `tests/unit/graph/test_capture_session.py` (extend)

- [ ] **Step 1: Write failing test**

Append to `tests/unit/graph/test_capture_session.py`:
```python
def test_session_cross_stream_record_then_wait_creates_edge():
    """sA.record(ev) → sB.wait(ev) should create a graph edge from sA's record node to sB's wait node."""
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream, Event
    sess = CaptureSession()
    sA = Stream()
    sB = Stream()
    sess.attach(sA)
    sess.attach(sB)
    ev = Event()
    sA.record(ev)        # adds event node, registers event_id -> node_id in session
    sB.wait(ev)          # adds event node + creates cross-stream edge
    g = sess.end()
    assert len(g.nodes) == 2
    record_node = next(n for n in g.nodes if n.event_args.op == "record")
    wait_node = next(n for n in g.nodes if n.event_args.op == "wait")
    assert (record_node.node_id, wait_node.node_id) in g.edges


def test_session_wait_for_unrecorded_event_no_edge():
    """If the event was never recorded inside the session, wait creates a node but no cross-edge."""
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream, Event
    sess = CaptureSession()
    sA = Stream()
    sess.attach(sA)
    ev = Event()
    sA.wait(ev)          # never recorded
    g = sess.end()
    assert len(g.nodes) == 1
    # no cross-edge (only intra-stream chaining edges, of which there are none here)
    assert g.edges == []
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/unit/graph/test_capture_session.py::test_session_cross_stream_record_then_wait_creates_edge -v`
Expected: FAIL — currently `record` and `wait` only chain within a single stream's `_capture_last_node`, no session-level cross-edge logic.

- [ ] **Step 3: Update `Stream.record` and `Stream.wait`**

Edit `gpusim/api.py`:

`record`:
```python
    def record(self, ev: "Event") -> None:
        """Append a record-marker. In capture mode (Phase 15), records as event node;
        in CaptureSession (M2), also registers as event source for cross-stream wait."""
        if self._captured_graph is not None:
            nid = self._captured_graph.add_event_node(event=ev, op="record")
            if self._capture_last_node is not None:
                self._captured_graph.add_dependency(self._capture_last_node, nid)
            self._capture_last_node = nid
            if self._capture_session is not None:
                self._capture_session.register_event_source(id(ev), nid)
            return
        self.pending.append(_RecordMarker(event=ev))
```

`wait`:
```python
    def wait(self, ev: "Event") -> None:
        """Block this stream's future launches. In capture mode (Phase 15), records as event node;
        in CaptureSession (M2), looks up the matching record node and creates cross-stream edge."""
        if self._captured_graph is not None:
            nid = self._captured_graph.add_event_node(event=ev, op="wait")
            if self._capture_last_node is not None:
                self._captured_graph.add_dependency(self._capture_last_node, nid)
            self._capture_last_node = nid
            if self._capture_session is not None:
                src = self._capture_session.lookup_event_source(id(ev))
                if src is not None and src != nid:
                    self._captured_graph.add_dependency(src, nid)
            return
        self.event_waits.append(ev)
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/unit/graph/test_capture_session.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/api.py tests/unit/graph/test_capture_session.py
git commit -m "feat(stream): cross-stream event-edge translation in CaptureSession (Phase 15 M2)"
```

---

## Task 9: `examples/stream_capture_multi_stream/` + parity test

**Files:**
- Create: `examples/stream_capture_multi_stream/{__init__.py, kernel.ptx, reference.py, run.py, README.md}`
- Create: `tests/parity/test_stream_capture_multi_stream.py`

- [ ] **Step 1: Write failing parity test**

Create `tests/parity/test_stream_capture_multi_stream.py`:
```python
import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "stream_capture_multi_stream"


def test_stream_capture_multi_stream_correctness():
    """sA: k1 → record(ev) → k3.  sB: wait(ev) → k2.  Captured into one graph."""
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream, Event
    from gpusim.config.loader import load_default

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    OUT = np.zeros(32, dtype=np.uint32)

    sess = CaptureSession()
    sA = Stream()
    sB = Stream()
    sess.attach(sA)
    sess.attach(sB)
    ev = Event()

    sA.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": OUT}, kernel_name="k1", config=cfg)
    sA.record(ev)
    sB.wait(ev)
    sB.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": OUT}, kernel_name="k2", config=cfg)
    sA.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": OUT}, kernel_name="k3", config=cfg)

    g = sess.end()
    # 5 nodes: k1, record, wait, k2, k3
    assert len(g.nodes) == 5
    # Edges: k1→record, record→wait (cross-stream), wait→k2, record→k3 (intra-stream A chain)
    assert len(g.edges) >= 3   # at minimum: k1→record, record→wait (cross-stream), wait→k2

    exec = g.instantiate(cfg)
    exec.launch()
    # 3 kernels * 32 threads * 1 increment = 96
    assert OUT.sum() == 96
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/parity/test_stream_capture_multi_stream.py -v`
Expected: FAIL — kernel.ptx missing.

- [ ] **Step 3: Create example files**

`examples/stream_capture_multi_stream/__init__.py` — empty.

`examples/stream_capture_multi_stream/kernel.ptx` (same single-thread incrementer as M1):
```
.visible .entry inc(.param .u64 OUT)
{
    .reg .u64 %rd<5>;
    .reg .u32 %r<5>;

    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;

    ld.global.u32 %r2, [%rd2];
    add.u32 %r3, %r2, 1;
    st.global.u32 [%rd2], %r3;

    ret;
}
```

`examples/stream_capture_multi_stream/reference.py`:
```python
import numpy as np


def reference(n: int = 32, kernels: int = 3):
    return np.full(n, kernels, dtype=np.uint32)
```

`examples/stream_capture_multi_stream/run.py`:
```python
import numpy as np, pathlib
from gpusim.graph.capture_session import CaptureSession
from gpusim.api import Stream, Event
from gpusim.config.loader import load_default


def main():
    n = 32
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()

    sess = CaptureSession()
    sA = Stream()
    sB = Stream()
    sess.attach(sA)
    sess.attach(sB)
    ev = Event()

    sA.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": OUT}, kernel_name="k1", config=cfg)
    sA.record(ev)
    sB.wait(ev)
    sB.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": OUT}, kernel_name="k2", config=cfg)
    sA.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": OUT}, kernel_name="k3", config=cfg)

    g = sess.end()
    print(f"Captured: {len(g.nodes)} nodes, {len(g.edges)} edges")
    print(f"Edges: {g.edges}")

    exec = g.instantiate(cfg)
    exec.launch()
    print(f"OUT.sum() = {OUT.sum()} (expected 96)")


if __name__ == "__main__":
    main()
```

`examples/stream_capture_multi_stream/README.md`:
```markdown
# stream_capture_multi_stream — Phase 15

Two streams capture into a shared `CaptureSession`. Cross-stream `record`/`wait`
becomes a Graph edge.

Stream A: `k1 → record(ev) → k3`
Stream B: `wait(ev) → k2`

Demonstrates:
- `CaptureSession` shared across streams (M2)
- Cross-stream event sync becomes a graph dependency edge automatically

## Run
```bash
python run.py
```
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/parity/test_stream_capture_multi_stream.py -v`
Expected: PASS.

Sanity:
```bash
python examples/stream_capture_multi_stream/run.py
```
Expected: prints node/edge counts and `OUT.sum() = 96`.

- [ ] **Step 5: Commit**

```bash
git add examples/stream_capture_multi_stream/ tests/parity/test_stream_capture_multi_stream.py
git commit -m "feat(examples): stream_capture_multi_stream — CaptureSession + cross-stream edge (Phase 15 M2)"
```

---

## Task 10: Tag M2

- [ ] **Step 1: Run full non-slow suite**

Run: `pytest -m "not slow" -q`
Expected: ~735-740 passed.

- [ ] **Step 2: Tag**

```bash
git tag M2-phase15-complete
```

---

# M3 — Conditional Graph Node

## Task 11: `ConditionalNodeArgs` + `Graph.add_conditional_node`

**Files:**
- Modify: `gpusim/graph/node.py` — add `ConditionalNodeArgs`, extend `GraphNode`
- Modify: `gpusim/graph/graph.py` — add `add_conditional_node`
- Test: `tests/unit/graph/test_conditional_node.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/graph/test_conditional_node.py`:
```python
def test_add_conditional_node_appends():
    from gpusim.graph.graph import Graph
    g_outer = Graph()
    g_true = Graph()
    g_false = Graph()
    nid = g_outer.add_conditional_node(
        cond_fn=lambda: True,
        true_graph=g_true,
        false_graph=g_false,
    )
    assert isinstance(nid, int)
    assert len(g_outer.nodes) == 1
    node = g_outer.nodes[0]
    assert node.type == "conditional"
    assert node.conditional_args is not None
    assert node.conditional_args.true_graph is g_true
    assert node.conditional_args.false_graph is g_false


def test_conditional_args_stores_callable():
    from gpusim.graph.graph import Graph
    from gpusim.graph.node import ConditionalNodeArgs
    g = Graph()
    f = lambda: True
    nid = g.add_conditional_node(cond_fn=f, true_graph=Graph(), false_graph=Graph())
    args = g.nodes[0].conditional_args
    assert isinstance(args, ConditionalNodeArgs)
    assert args.cond_fn is f
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/unit/graph/test_conditional_node.py -v`
Expected: FAIL with `AttributeError` (no `add_conditional_node`).

- [ ] **Step 3a: Add `ConditionalNodeArgs` and update `GraphNode`**

Edit `gpusim/graph/node.py` — append:
```python
@dataclass
class ConditionalNodeArgs:
    cond_fn: object       # Callable[[], bool], evaluated at exec time
    true_graph: object    # Graph
    false_graph: object   # Graph (may be empty)


@dataclass
class WhileNodeArgs:
    cond_fn: object       # Callable[[], bool], re-evaluated each iteration
    body_graph: object    # Graph
    max_iterations: int = 1000
```

Update `GraphNode` (replace existing dataclass):
```python
@dataclass
class GraphNode:
    node_id: int
    type: str             # "kernel" | "memcpy" | "event" | "memset" | "child_graph" | "conditional" | "while"
    kernel_args: KernelNodeArgs | None = None
    memcpy_args: MemcpyNodeArgs | None = None
    event_args: EventNodeArgs | None = None
    memset_args: MemsetNodeArgs | None = None
    child_graph_args: ChildGraphNodeArgs | None = None
    conditional_args: ConditionalNodeArgs | None = None    # NEW Phase 15
    while_args: WhileNodeArgs | None = None                # NEW Phase 15
```

(Defining `WhileNodeArgs` here too so M4 doesn't re-touch this file.)

- [ ] **Step 3b: Add `Graph.add_conditional_node`**

Edit `gpusim/graph/graph.py` — update import line + add method:
```python
from gpusim.graph.node import (
    GraphNode, KernelNodeArgs, MemcpyNodeArgs, EventNodeArgs,
    ConditionalNodeArgs, WhileNodeArgs,
)
```

Add method to `Graph` class (after `add_child_graph_node`):
```python
    def add_conditional_node(self, *, cond_fn, true_graph: "Graph",
                                false_graph: "Graph") -> int:
        nid = self._next_id; self._next_id += 1
        args = ConditionalNodeArgs(cond_fn=cond_fn, true_graph=true_graph,
                                      false_graph=false_graph)
        self.nodes.append(GraphNode(node_id=nid, type="conditional",
                                       conditional_args=args))
        return nid

    def add_while_node(self, *, cond_fn, body_graph: "Graph",
                          max_iterations: int = 1000) -> int:
        nid = self._next_id; self._next_id += 1
        args = WhileNodeArgs(cond_fn=cond_fn, body_graph=body_graph,
                                max_iterations=max_iterations)
        self.nodes.append(GraphNode(node_id=nid, type="while",
                                       while_args=args))
        return nid
```

(Both builder methods added together; `add_while_node` exercised in M4.)

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/unit/graph/test_conditional_node.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/graph/node.py gpusim/graph/graph.py tests/unit/graph/test_conditional_node.py
git commit -m "feat(graph): add_conditional_node + add_while_node builders + ConditionalNodeArgs/WhileNodeArgs (Phase 15 M3 prep)"
```

---

## Task 12: GraphExec branch dispatch + `ConditionalBranch` trace event

**Files:**
- Modify: `gpusim/graph/exec.py` — add conditional branch in `launch()`
- Test: `tests/unit/graph/test_conditional_node.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/graph/test_conditional_node.py`:
```python
def test_conditional_takes_true_branch():
    """When cond_fn returns True, only true_graph executes."""
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    src = """
.visible .entry inc(.param .u64 OUT) {
    .reg .u64 %rd<5>; .reg .u32 %r<5>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r2, [%rd2]; add.u32 %r3, %r2, 1; st.global.u32 [%rd2], %r3;
    ret;
}
"""
    OUT_T = np.zeros(32, dtype=np.uint32)
    OUT_F = np.zeros(32, dtype=np.uint32)
    g_true = Graph()
    g_true.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                              params={"OUT": OUT_T}, kernel_name="t")
    g_false = Graph()
    g_false.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                               params={"OUT": OUT_F}, kernel_name="f")

    g_outer = Graph()
    g_outer.add_conditional_node(cond_fn=lambda: True,
                                    true_graph=g_true, false_graph=g_false)
    g_outer.instantiate(cfg).launch()
    assert OUT_T.sum() == 32
    assert OUT_F.sum() == 0


def test_conditional_takes_false_branch():
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    src = """
.visible .entry inc(.param .u64 OUT) {
    .reg .u64 %rd<5>; .reg .u32 %r<5>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r2, [%rd2]; add.u32 %r3, %r2, 1; st.global.u32 [%rd2], %r3;
    ret;
}
"""
    OUT_T = np.zeros(32, dtype=np.uint32)
    OUT_F = np.zeros(32, dtype=np.uint32)
    g_true = Graph()
    g_true.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                              params={"OUT": OUT_T}, kernel_name="t")
    g_false = Graph()
    g_false.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                               params={"OUT": OUT_F}, kernel_name="f")

    g_outer = Graph()
    g_outer.add_conditional_node(cond_fn=lambda: False,
                                    true_graph=g_true, false_graph=g_false)
    g_outer.instantiate(cfg).launch()
    assert OUT_T.sum() == 0
    assert OUT_F.sum() == 32


def test_conditional_with_empty_false_branch():
    """If false_graph has no nodes, false branch executes 0 nodes (no error)."""
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    g_outer = Graph()
    g_outer.add_conditional_node(cond_fn=lambda: False,
                                    true_graph=Graph(), false_graph=Graph())
    cycles = g_outer.instantiate(cfg).launch()
    assert cycles >= 5    # at least the 5-cycle conditional eval overhead


def test_conditional_emits_trace_event():
    from gpusim.graph.graph import Graph
    from gpusim.graph.exec import GraphExec
    from gpusim.trace.recorder import Recorder
    from gpusim.config.loader import load_default

    cfg = load_default()
    rec = Recorder()
    g = Graph()
    g.add_conditional_node(cond_fn=lambda: True,
                              true_graph=Graph(), false_graph=Graph())
    exec = GraphExec.from_graph(g, cfg)
    exec._recorder = rec
    exec.launch()
    assert len(rec.conditional_branch_events) == 1
    ev = rec.conditional_branch_events[0]
    assert ev.taken is True
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/unit/graph/test_conditional_node.py -v -k branch`
Expected: FAIL — `launch()` doesn't yet have a `conditional` branch.

- [ ] **Step 3: Update `GraphExec.launch()`**

Edit `gpusim/graph/exec.py` — extend the `for node_id` loop with new branches (after the `child_graph` branch, before the `if self._recorder is not None:` block):

```python
            elif node.type == "conditional":
                a = node.conditional_args
                taken = bool(a.cond_fn())
                if self._recorder is not None:
                    self._recorder.conditional_branch(
                        node_id=node.node_id, taken=taken, cycle=total_cycles,
                    )
                chosen = a.true_graph if taken else a.false_graph
                if len(chosen.nodes) > 0:
                    child_exec = chosen.instantiate(self.config)
                    total_cycles += child_exec.launch()
                total_cycles += 5    # conditional eval overhead
            elif node.type == "while":
                a = node.while_args
                iteration = 0
                while a.cond_fn():
                    if iteration >= a.max_iterations:
                        raise RuntimeError(
                            f"while node {node.node_id} exceeded "
                            f"max_iterations={a.max_iterations}"
                        )
                    if self._recorder is not None:
                        self._recorder.loop_iteration(
                            node_id=node.node_id, iteration=iteration,
                            cycle=total_cycles,
                        )
                    if len(a.body_graph.nodes) > 0:
                        child_exec = a.body_graph.instantiate(self.config)
                        total_cycles += child_exec.launch()
                    iteration += 1
                total_cycles += 5    # final cond_fn eval overhead
```

(Both branches added together — saves a second touch of `exec.py` in M4.)

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/unit/graph/test_conditional_node.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/graph/exec.py tests/unit/graph/test_conditional_node.py
git commit -m "feat(graph): conditional + while branches in GraphExec.launch + trace events (Phase 15 M3 core)"
```

---

## Task 13: `examples/graph_conditional_branch/` + parity test

**Files:**
- Create: `examples/graph_conditional_branch/{__init__.py, kernel.ptx, reference.py, run.py, README.md}`
- Create: `tests/parity/test_graph_conditional_branch.py`

- [ ] **Step 1: Write failing parity test**

Create `tests/parity/test_graph_conditional_branch.py`:
```python
import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_conditional_branch"


def test_graph_conditional_branch_correctness():
    """Probe buffer determines which branch runs."""
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    OUT_A = np.zeros(32, dtype=np.uint32)
    OUT_B = np.zeros(32, dtype=np.uint32)
    probe = np.array([10], dtype=np.int32)    # > 5 → take true branch (A)

    g_A = Graph()
    g_A.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                            params={"OUT": OUT_A}, kernel_name="A")
    g_B = Graph()
    g_B.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                            params={"OUT": OUT_B}, kernel_name="B")

    g = Graph()
    g.add_conditional_node(cond_fn=lambda: probe[0] > 5,
                              true_graph=g_A, false_graph=g_B)
    g.instantiate(cfg).launch()
    assert OUT_A.sum() == 32
    assert OUT_B.sum() == 0

    # flip probe, run again on fresh buffers
    OUT_A2 = np.zeros(32, dtype=np.uint32)
    OUT_B2 = np.zeros(32, dtype=np.uint32)
    probe2 = np.array([2], dtype=np.int32)    # ≤ 5 → take false branch (B)

    g_A2 = Graph()
    g_A2.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                              params={"OUT": OUT_A2}, kernel_name="A")
    g_B2 = Graph()
    g_B2.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                              params={"OUT": OUT_B2}, kernel_name="B")

    g2 = Graph()
    g2.add_conditional_node(cond_fn=lambda: probe2[0] > 5,
                               true_graph=g_A2, false_graph=g_B2)
    g2.instantiate(cfg).launch()
    assert OUT_A2.sum() == 0
    assert OUT_B2.sum() == 32
```

- [ ] **Step 2: Run to confirm fail**

Expected: kernel.ptx missing.

- [ ] **Step 3: Create example files**

`examples/graph_conditional_branch/__init__.py` — empty.

`examples/graph_conditional_branch/kernel.ptx`:
```
.visible .entry inc(.param .u64 OUT)
{
    .reg .u64 %rd<5>;
    .reg .u32 %r<5>;

    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;

    mov.u32 %r2, 1;
    st.global.u32 [%rd2], %r2;

    ret;
}
```

`examples/graph_conditional_branch/reference.py`:
```python
import numpy as np


def reference(branch: str = "A", n: int = 32):
    return np.ones(n, dtype=np.uint32) if branch == "A" else np.zeros(n, dtype=np.uint32)
```

`examples/graph_conditional_branch/run.py`:
```python
import numpy as np, pathlib
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default


def main():
    n = 32
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()

    for probe_value in (10, 2):
        OUT_A = np.zeros(n, dtype=np.uint32)
        OUT_B = np.zeros(n, dtype=np.uint32)
        probe = np.array([probe_value], dtype=np.int32)

        g_A = Graph()
        g_A.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                params={"OUT": OUT_A}, kernel_name="A")
        g_B = Graph()
        g_B.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                params={"OUT": OUT_B}, kernel_name="B")

        g = Graph()
        g.add_conditional_node(cond_fn=lambda p=probe: p[0] > 5,
                                  true_graph=g_A, false_graph=g_B)
        g.instantiate(cfg).launch()
        taken = "A (true)" if probe_value > 5 else "B (false)"
        print(f"probe={probe_value}: branch {taken}, OUT_A.sum()={OUT_A.sum()}, OUT_B.sum()={OUT_B.sum()}")


if __name__ == "__main__":
    main()
```

`examples/graph_conditional_branch/README.md`:
```markdown
# graph_conditional_branch — Phase 15

Conditional graph node selects between two sub-graphs based on host-side `cond_fn`.

## Run
```bash
python run.py
```
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/parity/test_graph_conditional_branch.py -v`
Expected: PASS.

Sanity:
```bash
python examples/graph_conditional_branch/run.py
```
Expected: prints both branch outcomes.

- [ ] **Step 5: Commit**

```bash
git add examples/graph_conditional_branch/ tests/parity/test_graph_conditional_branch.py
git commit -m "feat(examples): graph_conditional_branch — host-eval if/else (Phase 15 M3)"
```

---

## Task 14: Tag M3

- [ ] **Step 1: Run full non-slow suite**

Run: `pytest -m "not slow" -q`
Expected: ~745-748 passed.

- [ ] **Step 2: Tag**

```bash
git tag M3-phase15-complete
```

---

# M4 — While Node + 4 Metrics

## Task 15: While node — exec + max_iterations cap test

**Files:**
- Test: `tests/unit/graph/test_while_node.py` (new)

(Builder method already added in Task 11; exec branch already added in Task 12. M4's first task focuses on dedicated while-node tests including the cap.)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/graph/test_while_node.py`:
```python
def test_while_node_runs_until_cond_false():
    """Body runs until cond_fn returns False."""
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    counter = np.array([3], dtype=np.int32)
    src = """
.visible .entry dec(.param .u64 OUT) {
    .reg .u64 %rd<3>; .reg .u32 %r<3>;
    ld.param.u64 %rd0, [OUT];
    ld.global.u32 %r0, [%rd0]; sub.u32 %r1, %r0, 1; st.global.u32 [%rd0], %r1;
    ret;
}
"""
    body = Graph()
    body.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(1,1,1),
                            params={"OUT": counter}, kernel_name="dec")

    # Use a single-iteration body counter held in Python so the kernel doesn't actually
    # have to mutate counter via PTX — for this unit test we use a host-side counter.
    iter_box = [3]
    def cond():
        return iter_box[0] > 0
    def body_python():
        iter_box[0] -= 1

    # Build a graph with a while node whose body is empty; we count iterations via cond_fn side-effect.
    g = Graph()
    g.add_while_node(cond_fn=lambda: (iter_box[0] > 0 and (body_python() or True)),
                        body_graph=Graph(),
                        max_iterations=10)
    g.instantiate(cfg).launch()
    assert iter_box[0] == 0


def test_while_node_max_iterations_raises():
    """When cond_fn never goes False, max_iterations cap raises."""
    import pytest
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    g = Graph()
    g.add_while_node(cond_fn=lambda: True, body_graph=Graph(), max_iterations=5)
    with pytest.raises(RuntimeError, match="exceeded max_iterations"):
        g.instantiate(cfg).launch()


def test_while_node_emits_loop_iteration_events():
    """Each iteration produces a LoopIteration trace event."""
    from gpusim.graph.graph import Graph
    from gpusim.graph.exec import GraphExec
    from gpusim.trace.recorder import Recorder
    from gpusim.config.loader import load_default

    cfg = load_default()
    rec = Recorder()
    iter_box = [4]
    g = Graph()
    g.add_while_node(cond_fn=lambda: (iter_box.__setitem__(0, iter_box[0] - 1) or iter_box[0] >= 0),
                        body_graph=Graph(), max_iterations=10)
    exec = GraphExec.from_graph(g, cfg)
    exec._recorder = rec
    exec.launch()
    assert len(rec.loop_iteration_events) == 4
    assert [e.iteration for e in rec.loop_iteration_events] == [0, 1, 2, 3]


def test_while_node_zero_iterations_when_cond_initially_false():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    g = Graph()
    g.add_while_node(cond_fn=lambda: False, body_graph=Graph(), max_iterations=10)
    cycles = g.instantiate(cfg).launch()
    assert cycles >= 5    # at least the cond_fn eval overhead
```

- [ ] **Step 2: Run to confirm pass**

Run: `pytest tests/unit/graph/test_while_node.py -v`
Expected: 4 passed (builder + exec already wired up in Tasks 11+12).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/graph/test_while_node.py
git commit -m "test(graph): while node iteration semantics + max_iterations cap (Phase 15 M4)"
```

---

## Task 16: `examples/graph_while_loop/` + parity test

**Files:**
- Create: `examples/graph_while_loop/{__init__.py, kernel.ptx, reference.py, run.py, README.md}`
- Create: `tests/parity/test_graph_while_loop.py`

- [ ] **Step 1: Write failing parity test**

Create `tests/parity/test_graph_while_loop.py`:
```python
import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_while_loop"


def test_graph_while_loop_correctness():
    """Loop until counter reaches 0; body runs N iterations."""
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    OUT = np.zeros(32, dtype=np.uint32)

    counter = [4]
    body = Graph()
    body.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                            params={"OUT": OUT}, kernel_name="inc")

    def cond():
        if counter[0] > 0:
            counter[0] -= 1
            return True
        return False

    g = Graph()
    g.add_while_node(cond_fn=cond, body_graph=body, max_iterations=10)
    g.instantiate(cfg).launch()
    # 4 iterations × 32 threads × 1 increment = 128
    assert OUT.sum() == 128
```

- [ ] **Step 2: Run to confirm fail**

Expected: kernel.ptx missing.

- [ ] **Step 3: Create example files**

`examples/graph_while_loop/__init__.py` — empty.

`examples/graph_while_loop/kernel.ptx`:
```
.visible .entry inc(.param .u64 OUT)
{
    .reg .u64 %rd<5>;
    .reg .u32 %r<5>;

    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;

    ld.global.u32 %r2, [%rd2];
    add.u32 %r3, %r2, 1;
    st.global.u32 [%rd2], %r3;

    ret;
}
```

`examples/graph_while_loop/reference.py`:
```python
import numpy as np


def reference(n: int = 32, iterations: int = 4):
    return np.full(n, iterations, dtype=np.uint32)
```

`examples/graph_while_loop/run.py`:
```python
import numpy as np, pathlib
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default


def main():
    n = 32
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    OUT = np.zeros(n, dtype=np.uint32)

    counter = [4]
    body = Graph()
    body.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                            params={"OUT": OUT}, kernel_name="inc")

    def cond():
        if counter[0] > 0:
            counter[0] -= 1
            return True
        return False

    g = Graph()
    g.add_while_node(cond_fn=cond, body_graph=body, max_iterations=10)
    g.instantiate(cfg).launch()
    print(f"After while loop (4 iters): OUT.sum() = {OUT.sum()} (expected 128)")


if __name__ == "__main__":
    main()
```

`examples/graph_while_loop/README.md`:
```markdown
# graph_while_loop — Phase 15

Graph with a while-node whose body runs until host-side `cond_fn` returns False.
`max_iterations` enforces a safety cap.

## Run
```bash
python run.py
```
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/parity/test_graph_while_loop.py -v`
Expected: PASS.

Sanity:
```bash
python examples/graph_while_loop/run.py
```
Expected: `OUT.sum() = 128`.

- [ ] **Step 5: Commit**

```bash
git add examples/graph_while_loop/ tests/parity/test_graph_while_loop.py
git commit -m "feat(examples): graph_while_loop — host-eval while body + max_iterations (Phase 15 M4)"
```

---

## Task 17: 4 Phase 15 metrics + unit tests

**Files:**
- Modify: `gpusim/analysis/metrics.py` — append 4 metrics
- Test: `tests/unit/analysis/test_phase15_metrics.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/analysis/test_phase15_metrics.py`:
```python
def test_stream_capture_count_zero_for_empty_trace():
    from gpusim.analysis.metrics import stream_capture_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    assert stream_capture_count(rec) == 0


def test_stream_capture_count_counts_end_events():
    from gpusim.analysis.metrics import stream_capture_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.stream_capture_end(stream_id=0, cycle=10, captured_node_count=3)
    rec.stream_capture_end(stream_id=1, cycle=20, captured_node_count=5)
    assert stream_capture_count(rec) == 2


def test_captured_node_count_sums_across_captures():
    from gpusim.analysis.metrics import captured_node_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.stream_capture_end(stream_id=0, cycle=10, captured_node_count=3)
    rec.stream_capture_end(stream_id=1, cycle=20, captured_node_count=5)
    assert captured_node_count(rec) == 8


def test_captured_node_count_zero_when_no_captures():
    from gpusim.analysis.metrics import captured_node_count
    from gpusim.trace.recorder import Recorder
    assert captured_node_count(Recorder()) == 0


def test_conditional_branch_taken_count_only_counts_true():
    from gpusim.analysis.metrics import conditional_branch_taken_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.conditional_branch(node_id=0, taken=True, cycle=0)
    rec.conditional_branch(node_id=1, taken=False, cycle=10)
    rec.conditional_branch(node_id=2, taken=True, cycle=20)
    assert conditional_branch_taken_count(rec) == 2


def test_conditional_branch_taken_count_zero_when_none_taken():
    from gpusim.analysis.metrics import conditional_branch_taken_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.conditional_branch(node_id=0, taken=False, cycle=0)
    assert conditional_branch_taken_count(rec) == 0


def test_avg_loop_iterations_returns_zero_for_no_while_nodes():
    from gpusim.analysis.metrics import avg_loop_iterations
    from gpusim.trace.recorder import Recorder
    assert avg_loop_iterations(Recorder()) == 0.0


def test_avg_loop_iterations_per_node():
    """node_id 0 ran 3 iterations, node_id 1 ran 5 — average is 4.0."""
    from gpusim.analysis.metrics import avg_loop_iterations
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    for i in range(3):
        rec.loop_iteration(node_id=0, iteration=i, cycle=i*10)
    for i in range(5):
        rec.loop_iteration(node_id=1, iteration=i, cycle=100+i*10)
    assert avg_loop_iterations(rec) == 4.0
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest tests/unit/analysis/test_phase15_metrics.py -v`
Expected: FAIL — metrics not yet defined.

- [ ] **Step 3: Add metrics**

Append to `gpusim/analysis/metrics.py`:
```python
def stream_capture_count(recorder) -> int:
    """Phase 15: Number of distinct captured graphs in this trace
    (counts StreamCaptureEnd events)."""
    return len(getattr(recorder, "stream_capture_end_events", []))


def captured_node_count(recorder) -> int:
    """Phase 15: Total nodes added across all stream-captured graphs in this trace."""
    return sum(
        ev.captured_node_count
        for ev in getattr(recorder, "stream_capture_end_events", [])
    )


def conditional_branch_taken_count(recorder) -> int:
    """Phase 15: Number of conditional-node evaluations that took the true branch."""
    return sum(
        1 for ev in getattr(recorder, "conditional_branch_events", [])
        if ev.taken
    )


def avg_loop_iterations(recorder) -> float:
    """Phase 15: Mean iteration count across all while-nodes in the trace.
    Returns 0.0 if no while-nodes were executed."""
    events = getattr(recorder, "loop_iteration_events", [])
    if not events:
        return 0.0
    per_node = {}
    for ev in events:
        per_node[ev.node_id] = per_node.get(ev.node_id, 0) + 1
    return sum(per_node.values()) / len(per_node)
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/unit/analysis/test_phase15_metrics.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/analysis/metrics.py tests/unit/analysis/test_phase15_metrics.py
git commit -m "feat(analysis): 4 Phase 15 metrics — stream_capture_count, captured_node_count, conditional_branch_taken_count, avg_loop_iterations"
```

---

## Task 18: Tag M4

- [ ] **Step 1: Run full non-slow suite**

Run: `pytest -m "not slow" -q`
Expected: ~755-760 passed.

- [ ] **Step 2: Tag**

```bash
git tag M4-phase15-complete
```

---

# M5 — Tutorials + Microbench + Regression Rename + README + Ship

## Task 19: 4 tutorial chapters

**Files:**
- Create: `docs/tutorial/58-stream-capture-basic.md`
- Create: `docs/tutorial/59-stream-capture-multi-stream.md`
- Create: `docs/tutorial/60-graph-conditional-branch.md`
- Create: `docs/tutorial/61-graph-while-loop.md`

Use Phase 14 chapters (54-57) as the structural template: English body, Chinese subheadings `看模拟器` / `改一改` / `真机对照`, ~500-700 words each.

- [ ] **Step 1: Read Phase 14 reference template**

Run: `cat docs/tutorial/54-persistent-kernel-server.md`

Use this as the format reference (English prose body + Chinese subheadings).

- [ ] **Step 2: Create chapter 58**

Create `docs/tutorial/58-stream-capture-basic.md`:
```markdown
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
```

- [ ] **Step 3: Create chapter 59**

Create `docs/tutorial/59-stream-capture-multi-stream.md`:
```markdown
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
```

- [ ] **Step 4: Create chapter 60**

Create `docs/tutorial/60-graph-conditional-branch.md`:
```markdown
# 60 · Graph Conditional Branch — Host-Side If/Else In a Graph

`Graph.add_conditional_node(cond_fn, true_graph, false_graph)` adds a node that
evaluates `cond_fn()` host-side at execution time and dispatches into the chosen
sub-graph.

## What the example does

```python
from gpusim.graph.graph import Graph

probe = np.array([10], dtype=np.int32)

g_A = Graph(); g_A.add_kernel_node(...)   # branch A
g_B = Graph(); g_B.add_kernel_node(...)   # branch B

g = Graph()
g.add_conditional_node(cond_fn=lambda: probe[0] > 5,
                          true_graph=g_A, false_graph=g_B)
g.instantiate(cfg).launch()
# probe[0] == 10 > 5 → A executed, B skipped
```

## 看模拟器

At launch, GraphExec encounters the `conditional` node, calls `cond_fn()`,
emits `ConditionalBranch(node_id, taken, cycle)` to the recorder, and
instantiates+launches the chosen sub-graph. The eval itself adds 5 cycles of
overhead. An empty `false_graph` (or `true_graph`) is allowed — that branch
becomes a no-op.

The metric `conditional_branch_taken_count(recorder)` counts how many evaluations
chose the true branch over the lifetime of the trace.

## 改一改

- Pass `false_graph=Graph()` (empty) to express "if without else".
- Make `cond_fn` close over a buffer that earlier kernel nodes wrote to —
  capture-then-decide patterns can be expressed entirely within one Graph.

## 真机对照

CUDA 12.4+ adds `cudaGraphAddNode` with `cudaGraphNodeTypeConditional` for true
device-side conditional branches inside graphs. Phase 15 implements the
host-evaluation variant — the simulator equivalent of running `cond_fn` in a
host callback between graph segments. Future Phase: device-side conditional
nodes that read predicates from device memory without round-tripping host.
```

- [ ] **Step 5: Create chapter 61**

Create `docs/tutorial/61-graph-while-loop.md`:
```markdown
# 61 · Graph While Loop — Bounded Iteration In a Graph

`Graph.add_while_node(cond_fn, body_graph, max_iterations=1000)` adds a node
that re-runs `body_graph` until `cond_fn()` returns False. `max_iterations`
caps the loop to prevent runaway simulation; exceeding it raises
`RuntimeError`.

## What the example does

```python
counter = [4]
body = Graph(); body.add_kernel_node(...)

def cond():
    if counter[0] > 0:
        counter[0] -= 1
        return True
    return False

g = Graph()
g.add_while_node(cond_fn=cond, body_graph=body, max_iterations=10)
g.instantiate(cfg).launch()
# body ran 4 times, OUT.sum() reflects 4 increments per thread
```

## 看模拟器

GraphExec's `while` branch evaluates `cond_fn()` at the top of each iteration.
On `True`, it emits `LoopIteration(node_id, iteration, cycle)` and
instantiates+launches `body_graph`. On `False`, the loop terminates and 5
overhead cycles are added for the final eval.

`max_iterations` is a safety net — when an unbounded `cond_fn` (e.g.,
`lambda: True`) would otherwise spin forever, the simulator raises after the
cap. Tune it conservatively for tests; keep production loops well below the
cap.

The metric `avg_loop_iterations(recorder)` returns the mean iteration count
across all while-nodes whose iterations were recorded.

## 改一改

- Pass `body_graph=Graph()` (empty) to count iterations from `cond_fn`'s side
  effects without doing any kernel work — useful for instrumentation.
- Set `max_iterations=5` and supply a `cond_fn=lambda: True` — the launch
  raises `RuntimeError("exceeded max_iterations")`.

## 真机对照

CUDA 12.4+ adds `cudaGraphNodeTypeWhileLoop` for device-side loops in graphs.
Phase 15's host-evaluated variant mirrors a host-callback driven loop — the
common pattern in PyTorch/JAX before device-side conditional graph nodes
became available.
```

- [ ] **Step 6: Commit**

```bash
git add docs/tutorial/58-stream-capture-basic.md docs/tutorial/59-stream-capture-multi-stream.md docs/tutorial/60-graph-conditional-branch.md docs/tutorial/61-graph-while-loop.md
git commit -m "docs(tutorial): chapters 58-61 — Phase 15 stream capture + conditional/while nodes"
```

---

## Task 20: Microbench facts + runtime

**Files:**
- Create: `tests/microbench/test_phase15_facts.py`
- Create: `tests/microbench/test_phase15_runtime.py`

- [ ] **Step 1: Create facts microbench**

Create `tests/microbench/test_phase15_facts.py`:
```python
"""Phase 15 microbench — stream capture + conditional/while node facts."""


def test_capture_appends_nodes_without_executing():
    """begin_capture → launch puts node in graph but does not execute kernel."""
    import numpy as np
    from gpusim.api import Stream
    from gpusim.config.loader import load_default
    cfg = load_default()
    ptx = """
.visible .entry inc(.param .u64 OUT) {
    .reg .u64 %rd<5>; .reg .u32 %r<5>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 1; st.global.u32 [%rd2], %r2;
    ret;
}
"""
    OUT = np.zeros(32, dtype=np.uint32)
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": OUT}, kernel_name="t", config=cfg)
    s.end_capture()
    assert OUT.sum() == 0    # not executed during capture


def test_captured_graph_replay_equivalent_to_imperative():
    """Captured graph + replay produces same OUT as direct stream launch."""
    import numpy as np
    from gpusim.api import Stream, synchronize
    from gpusim.config.loader import load_default
    cfg = load_default()
    ptx = """
.visible .entry inc(.param .u64 OUT) {
    .reg .u64 %rd<5>; .reg .u32 %r<5>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r2, [%rd2]; add.u32 %r3, %r2, 1; st.global.u32 [%rd2], %r3;
    ret;
}
"""
    # Imperative
    OUT_imp = np.zeros(32, dtype=np.uint32)
    s_imp = Stream()
    s_imp.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"OUT": OUT_imp}, kernel_name="t", config=cfg)
    synchronize(streams=[s_imp], config=cfg)
    # Captured + replay
    OUT_cap = np.zeros(32, dtype=np.uint32)
    s_cap = Stream()
    s_cap.begin_capture()
    s_cap.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"OUT": OUT_cap}, kernel_name="t", config=cfg)
    g = s_cap.end_capture()
    g.instantiate(cfg).launch()
    assert OUT_imp.sum() == OUT_cap.sum() == 32


def test_conditional_branch_event_records_taken():
    """ConditionalBranch.taken matches cond_fn return value."""
    from gpusim.graph.graph import Graph
    from gpusim.graph.exec import GraphExec
    from gpusim.trace.recorder import Recorder
    from gpusim.config.loader import load_default
    cfg = load_default()
    rec = Recorder()
    g = Graph()
    g.add_conditional_node(cond_fn=lambda: False,
                              true_graph=Graph(), false_graph=Graph())
    exec = GraphExec.from_graph(g, cfg)
    exec._recorder = rec
    exec.launch()
    assert rec.conditional_branch_events[0].taken is False


def test_while_max_iterations_enforced():
    """Loop with always-True cond_fn raises after max_iterations."""
    import pytest
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = Graph()
    g.add_while_node(cond_fn=lambda: True, body_graph=Graph(), max_iterations=3)
    with pytest.raises(RuntimeError, match="exceeded max_iterations"):
        g.instantiate(cfg).launch()


def test_captured_graph_is_captured_flag_true():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()
    g = s.end_capture()
    assert g.is_captured is True


def test_handbuilt_graph_is_captured_flag_false():
    from gpusim.graph.graph import Graph
    g = Graph()
    assert g.is_captured is False
```

- [ ] **Step 2: Run facts**

Run: `pytest tests/microbench/test_phase15_facts.py -v`
Expected: 6 passed.

- [ ] **Step 3: Create runtime microbench**

Create `tests/microbench/test_phase15_runtime.py`:
```python
import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_stream_capture_basic_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "stream_capture_basic"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_stream_capture_multi_stream_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "stream_capture_multi_stream"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_graph_conditional_branch_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_conditional_branch"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_graph_while_loop_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_while_loop"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
```

- [ ] **Step 4: Run runtime (slow)**

Run: `pytest tests/microbench/test_phase15_runtime.py -v -m slow`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/microbench/test_phase15_facts.py tests/microbench/test_phase15_runtime.py
git commit -m "test(microbench): Phase 15 facts (6) + runtime (4 slow)"
```

---

## Task 21: Regression rename phase1_13 → phase1_14 + add 4 Phase 14 examples

**Files:**
- Rename: `tests/parity/test_phase1_13_examples_unchanged.py` → `tests/parity/test_phase1_14_examples_unchanged.py`
- Modify the renamed file: list name, docstring, comments, and append the 4 Phase 14 examples to `PHASE_1_13_EXAMPLES` (which becomes `PHASE_1_14_EXAMPLES`).

- [ ] **Step 1: Rename file**

```bash
git mv tests/parity/test_phase1_13_examples_unchanged.py tests/parity/test_phase1_14_examples_unchanged.py
```

- [ ] **Step 2: Update list name + add Phase 14 examples**

Edit `tests/parity/test_phase1_14_examples_unchanged.py`:

Change line 1:
```python
"""Smoke-test: each Phase 1-14 example runs without crashing on Phase 14 Device path."""
```

Change `PHASE_1_13_EXAMPLES` → `PHASE_1_14_EXAMPLES` (rename the constant globally in the file).

After the `# Phase 13` block (`graph_memset_zero`, `graph_with_child`, `graph_update_replay`), append:
```python
    # Phase 14
    "persistent_kernel_server",
    "dynamic_parallelism_recursive",
    "persistent_work_queue",
    "persistent_pipeline",
```

Update both `@pytest.mark.parametrize` decorators and both function names from `_1_13_` → `_1_14_`:
```python
@pytest.mark.parametrize("ex", [e for e in PHASE_1_14_EXAMPLES if e not in SLOW_EXAMPLES])
def test_phase_1_14_example_smoke(ex):
    ...

@pytest.mark.slow
@pytest.mark.parametrize("ex", sorted(SLOW_EXAMPLES))
def test_phase_1_14_example_smoke_slow(ex):
    ...
```

- [ ] **Step 3: Run renamed regression**

Run: `pytest tests/parity/test_phase1_14_examples_unchanged.py -v -m "not slow"`
Expected: ~52-53 passed (49 from before + 4 Phase 14 examples; minus any examples without `run.py` which are skipped).

- [ ] **Step 4: Commit**

```bash
git add tests/parity/test_phase1_14_examples_unchanged.py
git commit -m "test(regression): rename phase1_13 → phase1_14 + add 4 Phase 14 examples"
```

---

## Task 22: README v15

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Find phase status table**

Run: `grep -n 'Phase status\|^### Phase 14\|Phase 15' README.md | head -10`

- [ ] **Step 2: Add Phase 15 row to status table**

Edit `README.md` — locate the "Phase status" table (around line 143). After the `Phase 14` row, add:
```markdown
| 15 | Stream Capture API + Conditional Graph Nodes | ✅ |
```
(Match the exact column structure of the existing table — read a few lines around line 143 first to confirm column count.)

- [ ] **Step 3: Add Phase 15 detailed section**

After the existing Phase 14 detailed section (around line 201), append:
```markdown
### Phase 15 ✅ — Stream Capture API + Conditional Graph Nodes

Phase 11 introduced minimal `Stream.begin_capture/end_capture` for single-stream
capture of `Stream.launch` only. Phase 15 completes the picture:
**`mode` parameter validation** (only `"global"` accepted), **double-`begin_capture`
and `end_capture`-without-begin both raise**, **`Graph.is_captured` flag** so
analytics distinguish captured graphs from hand-built ones, and **`record`/`wait`
also captured** as event nodes with intra-stream chaining edges. **`CaptureSession`**
lets multiple streams capture into one shared `Graph` — cross-stream
`record(ev)` / `wait(ev)` translate automatically into Graph dependency edges.
**Conditional graph nodes** (`add_conditional_node(cond_fn, true_graph, false_graph)`)
and **while graph nodes** (`add_while_node(cond_fn, body_graph, max_iterations=1000)`)
add host-evaluated control flow inside graphs, with a safety cap on while loops.
**4 new metrics**: `stream_capture_count`, `captured_node_count`,
`conditional_branch_taken_count`, `avg_loop_iterations`. **4 new trace events**:
`StreamCaptureBegin`, `StreamCaptureEnd`, `ConditionalBranch`, `LoopIteration`.
**4 new examples** (stream_capture_basic, stream_capture_multi_stream,
graph_conditional_branch, graph_while_loop). **4 new tutorial chapters** (58-61).
**100% backward compatible:** Phase 1-14 APIs unchanged; existing Phase 11
`graph_capture_from_stream` example still works because `begin_capture()` defaults
to `mode="global"`.
```

- [ ] **Step 4: Verify README renders**

Run: `head -210 README.md | tail -30`
Expected: see Phase 14 + Phase 15 sections in order.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(readme): v15 — Phase 15 capabilities (stream capture + conditional/while graph nodes)"
```

---

## Task 23: Final acceptance verification + tag `phase15-complete`

- [ ] **Step 1: Run full non-slow suite**

Run: `pytest -m "not slow" -q`
Expected: ~755-760 passed (Phase 14 baseline 717 + ~26 Phase 15 unit/parity + 6 microbench facts + 4 Phase 14 regression).

If count differs significantly, investigate before tagging.

- [ ] **Step 2: Run slow suite at least once**

Run: `pytest -m slow -q`
Expected: Phase 15 runtime tests (4) + Phase 14 runtime tests (2) all pass; the `l1_thrash_demo` slow regression passes.

- [ ] **Step 3: Verify all 5 milestone tags**

Run: `git tag -l 'M*-phase15-complete' 'phase15-complete' | sort`
Expected output (before this step's tag):
```
M1-phase15-complete
M2-phase15-complete
M3-phase15-complete
M4-phase15-complete
```

- [ ] **Step 4: Tag `phase15-complete`**

```bash
git tag phase15-complete
```

- [ ] **Step 5: Verify final tag list**

Run: `git tag -l 'M*-phase15-complete' 'phase15-complete' | sort -V`
Expected:
```
M1-phase15-complete
M2-phase15-complete
M3-phase15-complete
M4-phase15-complete
phase15-complete
```

- [ ] **Step 6: Final summary commit (no-op tag verification)**

Run: `git log --oneline -25 | head -30`
Expected: see the chronological sequence of Phase 15 commits ending with the README v15 commit.

---

## Acceptance Criteria

Phase 15 ships when:

- [ ] All 5 milestone tags present (`M1-phase15-complete` ... `M4-phase15-complete`, `phase15-complete`)
- [ ] All 4 examples run cleanly via `python run.py`
- [ ] All 4 parity tests pass
- [ ] Microbench facts: capture appends only (no exec), capture-replay equivalence, conditional taken recorded, while max_iterations enforced, is_captured flag toggles correctly
- [ ] Phase 1-14 regression test (renamed) passes with 4 Phase 14 examples added
- [ ] Test count: 717 → ~755-760 (+38-43 — slightly above original ~28 estimate due to bundling all event types into recorder upfront)
- [ ] README v15 documents Phase 15 capabilities
