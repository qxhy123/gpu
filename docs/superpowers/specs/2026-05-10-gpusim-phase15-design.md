# gpusim Phase 15 — Stream Capture API + Conditional Graph Nodes

> **Status:** Brainstormed 2026-05-10. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- Add **stream capture API** (`Stream.begin_capture(mode="global")` / `Stream.end_capture() -> Graph`) — convert an imperative stream sequence into a Graph automatically.
- Add **conditional graph nodes** (`Graph.add_conditional_node(cond_fn, true_graph, false_graph)`) — host-evaluated if/else branching during graph execution.
- Add **while graph nodes** (`Graph.add_while_node(cond_fn, body_graph, max_iterations=1000)`) — host-evaluated loop nodes with safety cap.
- 4 examples + 4 tutorial chapters (58-61).
- 4 new metrics: `stream_capture_count`, `captured_node_count`, `conditional_branch_taken_count`, `avg_loop_iterations`.
- Reuse Phase 11 HTML §35 + Perfetto Graph swimlane; add capture-vs-handbuilt distinction.
- 100% backward compatible: Phase 1-14 unchanged.

### Non-goals (deferred to Phase 16+)
- `cudaStreamBeginCaptureToGraph` — capture appended to an existing graph.
- Cross-process graphs (CUDA IPC).
- `cudaGraphExecChildGraphNodeSetParams` — swap child-graph at runtime.
- Nested stream capture (capturing within a captured kernel).
- `thread` / `relaxed` capture modes (CUDA exposes 3; we ship only `global`).
- Device-side condition evaluation (we evaluate `cond_fn` on host only).

---

## 2. Architecture

```
gpusim.core.stream (Phase 7-9) — NEW capture state:
├── Stream._capturing: bool
├── Stream._capture_graph: Graph | None
├── Stream._capture_last_node_id: int | None     # for chaining edges
├── Stream.begin_capture(mode: str = "global") -> None
├── Stream.end_capture() -> Graph
└── Stream.{launch_kernel, memcpy_h2d, memcpy_d2h, record, wait_event} —
       branch on _capturing → append to _capture_graph instead of executing

gpusim.graph (Phase 11+13) — NEW node types + APIs:
├── GraphNode.type adds "conditional" | "while"
├── ConditionalNodeArgs (NEW)
├── WhileNodeArgs (NEW)
├── Graph.is_captured: bool                       # True iff produced by Stream.end_capture
├── Graph.add_conditional_node(cond_fn, true_graph, false_graph) -> nid
├── Graph.add_while_node(cond_fn, body_graph, max_iterations=1000) -> nid
└── GraphExec.launch — handles conditional / while branches

gpusim.trace (extended):
├── StreamCaptureBegin(stream_id, cycle)
├── StreamCaptureEnd(stream_id, cycle, captured_node_count)
├── ConditionalBranch(node_id, taken: bool, cycle)
└── LoopIteration(node_id, iteration: int, cycle)
```

### Key invariants
- Capture is per-stream, single-active: calling `begin_capture` on a stream that is already capturing raises `RuntimeError`.
- Capture mode is `"global"` only; other values raise `ValueError` (forward-compat slot).
- Cross-stream event dependencies are translated to graph edges only when **both** the recording and waiting stream are in capture mode at the time of the call. Otherwise the wait/record is a runtime no-op as today.
- Capture overhead: 5 cycles per appended node (cheap bookkeeping); the captured Graph itself reports zero overhead at replay relative to a hand-built equivalent.
- `cond_fn(buf_or_state) -> bool` is evaluated host-side at the cycle of the conditional node; result determines which sub-graph executes.
- `while_node` halts when `cond_fn` returns False OR `max_iterations` reached; reaching cap raises `RuntimeError("while node max_iterations exceeded")`.
- Captured graphs respond to all Phase 11/13 APIs (`instantiate`, `launch`, `update_kernel_node_params`) identically to hand-built graphs.

---

## 3. Data model

### 3.1 New node arg dataclasses (`gpusim/graph/node.py`)

```python
@dataclass
class ConditionalNodeArgs:
    cond_fn: Callable          # () -> bool, evaluated at exec
    true_graph: object         # Graph
    false_graph: object        # Graph (may be empty Graph for "if without else")


@dataclass
class WhileNodeArgs:
    cond_fn: Callable          # () -> bool, re-evaluated each iteration
    body_graph: object         # Graph
    max_iterations: int = 1000
```

### 3.2 GraphNode extension

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

### 3.3 Graph builder methods

```python
class Graph:
    is_captured: bool = False  # default False; Stream.end_capture sets True

    def add_conditional_node(self, *, cond_fn, true_graph, false_graph) -> int:
        nid = self._next_id; self._next_id += 1
        args = ConditionalNodeArgs(cond_fn=cond_fn, true_graph=true_graph, false_graph=false_graph)
        self.nodes.append(GraphNode(node_id=nid, type="conditional", conditional_args=args))
        return nid

    def add_while_node(self, *, cond_fn, body_graph, max_iterations: int = 1000) -> int:
        nid = self._next_id; self._next_id += 1
        args = WhileNodeArgs(cond_fn=cond_fn, body_graph=body_graph, max_iterations=max_iterations)
        self.nodes.append(GraphNode(node_id=nid, type="while", while_args=args))
        return nid
```

### 3.4 Stream capture (in `gpusim/core/stream.py`)

```python
class Stream:
    _capturing: bool = False
    _capture_graph: object | None = None
    _capture_last_node_id: int | None = None

    def begin_capture(self, mode: str = "global") -> None:
        if mode != "global":
            raise ValueError(f"only 'global' capture mode supported in Phase 15, got {mode!r}")
        if self._capturing:
            raise RuntimeError(f"stream {self.stream_id} is already capturing")
        from gpusim.graph import Graph
        self._capturing = True
        self._capture_graph = Graph()
        self._capture_graph.is_captured = True
        self._capture_last_node_id = None
        # emit StreamCaptureBegin trace event if recorder attached

    def end_capture(self) -> "Graph":
        if not self._capturing:
            raise RuntimeError(f"stream {self.stream_id} is not capturing")
        g = self._capture_graph
        captured_count = len(g.nodes)
        self._capturing = False
        self._capture_graph = None
        self._capture_last_node_id = None
        # emit StreamCaptureEnd trace event with captured_count
        return g
```

### 3.5 Stream op interception during capture

For each existing Stream op (`launch_kernel`, `memcpy_h2d`, `memcpy_d2h`, `record`, `wait_event`), wrap entry with:

```python
def launch_kernel(self, ptx_src, grid, block, params, *, kernel_name="<k>"):
    if self._capturing:
        nid = self._capture_graph.add_kernel_node(
            ptx_src=ptx_src, grid=grid, block=block, params=params, kernel_name=kernel_name
        )
        if self._capture_last_node_id is not None:
            self._capture_graph.add_dependency(self._capture_last_node_id, nid)
        self._capture_last_node_id = nid
        return  # do NOT execute
    # ... existing execute path ...
```

`record(event)` during capture: appends an event node and stores `(event_id → node_id)` in the capture graph's local event-source table.

`wait_event(event)` during capture: looks up the event in the same capture graph's event-source table; if found, calls `add_dependency(src_node_id, current_node_id)` for the next op. If the event is not in the same graph (i.e. recorded outside the current capture, or in a sibling stream that did not share the capture session), the wait is captured as a no-op edge — Phase 15 does not silently merge across distinct captures.

**Single-stream capture** (M1, the common case) uses `s.begin_capture()` and gets its own `Graph` back from `end_capture()`.

**Multi-stream capture** (M2) uses an explicit `CaptureSession` so participating streams share the same captured `Graph`:

```python
session = CaptureSession()
sA.begin_capture(session=session)
sB.begin_capture(session=session)
# ... ops + cross-stream record/wait ...
g = session.end()  # returns the merged Graph; both streams exit capture mode
```

Inside a session, `record`/`wait_event` translate cross-stream event sync into Graph dependency edges directly (because both streams write into the same `Graph` instance).

### 3.6 GraphExec.launch — handle new node types

```python
def launch(self) -> int:
    # ... existing kernel/memcpy/event/memset/child_graph branches ...
    elif node.type == "conditional":
        a = node.conditional_args
        taken = bool(a.cond_fn())
        # emit ConditionalBranch(node_id=node.node_id, taken=taken)
        chosen = a.true_graph if taken else a.false_graph
        if len(chosen.nodes) > 0:
            child_exec = chosen.instantiate(self.config)
            total_cycles += child_exec.launch()
        total_cycles += 5  # conditional eval overhead
    elif node.type == "while":
        a = node.while_args
        iteration = 0
        while a.cond_fn():
            if iteration >= a.max_iterations:
                raise RuntimeError(
                    f"while node {node.node_id} exceeded max_iterations={a.max_iterations}"
                )
            # emit LoopIteration(node_id=node.node_id, iteration=iteration)
            child_exec = a.body_graph.instantiate(self.config)
            total_cycles += child_exec.launch()
            iteration += 1
        total_cycles += 5  # final cond_fn eval overhead
```

---

## 4. Trace + Analysis

### 4.1 Trace events (in `gpusim/trace/events.py`)

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

### 4.2 4 new metrics (in `gpusim/analysis/metrics.py`)

```python
def stream_capture_count(trace) -> int:
    """Number of distinct captured graphs in this trace (counts StreamCaptureEnd events)."""

def captured_node_count(trace) -> int:
    """Total nodes added across all stream-captured graphs in this trace
    (sums StreamCaptureEnd.captured_node_count)."""

def conditional_branch_taken_count(trace) -> int:
    """Number of conditional-node evaluations that took the true branch."""

def avg_loop_iterations(trace) -> float:
    """Mean iteration count across all while-nodes in the trace.
    Returns 0.0 if no while-nodes were executed."""
```

### 4.3 Result API

```python
class GraphExecResult:
    def stream_capture_count(self) -> int: ...
    def captured_node_count(self) -> int: ...
    def conditional_branch_taken_count(self) -> int: ...
    def avg_loop_iterations(self) -> float: ...
```

---

## 5. Viz

Reuse Phase 11 HTML §35 + Perfetto Graph swimlane. Phase 15 adds:
- §35 entries gain a `[captured]` tag for graphs with `is_captured=True`.
- Conditional / while node entries include the branch decision (`true` / `false`) or iteration count for quick scan.
- No new HTML sections.

---

## 6. Examples (4)

### 6.1 `stream_capture_basic/`
- 3-kernel sequence on 1 stream.
- Run imperatively: measure cycles. Then `begin_capture` → same 3 kernels → `end_capture` → replay 100×.
- **Verifies:** captured graph produces same outputs; replay total cycles ≈ 100× one execution + overhead.

### 6.2 `stream_capture_multi_stream/`
- 2 streams in fork-join via shared `CaptureSession`.
- Stream A: kernel1 → record(eventX) → kernel3.
- Stream B: wait_event(eventX) → kernel2.
- **Verifies:** captured Graph has correct cross-stream edge; replayed cycles match imperative.

### 6.3 `graph_conditional_branch/`
- Build graph: kernel that writes to a probe buffer → conditional node (`cond_fn = lambda: probe.sum() > threshold`) → branch into either kernel_A or kernel_B.
- **Verifies:** branch taken matches expected based on data; trace records `ConditionalBranch(taken=...)`.

### 6.4 `graph_while_loop/`
- Build graph: while-node body = kernel that decrements a counter; `cond_fn = lambda: counter[0] > 0`.
- **Verifies:** loop runs N iterations until counter zeroes; `LoopIteration` events recorded; `avg_loop_iterations == N`.

---

## 7. Tutorials

`docs/tutorial/` chapters 58-61:
- **58-stream-capture-basic.md** — example 1
- **59-stream-capture-multi-stream.md** — example 2
- **60-graph-conditional-branch.md** — example 3
- **61-graph-while-loop.md** — example 4

---

## 8. Testing strategy

### Unit tests (~14 new)
- `tests/unit/core/test_stream_capture.py` — begin/end semantics, double-begin error, end-without-begin error, captured graph contents
- `tests/unit/core/test_capture_session.py` — multi-stream merged session
- `tests/unit/graph/test_conditional_node.py` — Graph.add_conditional_node + true / false dispatch + empty branch
- `tests/unit/graph/test_while_node.py` — Graph.add_while_node + iteration count + max_iterations cap raises
- `tests/unit/analysis/test_phase15_metrics.py` — 4 new metrics

### Parity tests (~4)
One per example.

### Microbench
- `test_phase15_facts.py` (fast):
  - Capture appends nodes without executing them
  - Captured graph replay equivalence to imperative
  - Conditional node trace event recorded with correct `taken`
  - While node respects max_iterations
- `test_phase15_runtime.py` (slow): 4 examples each under 60s

### Regression
- Rename `tests/parity/test_phase1_13_examples_unchanged.py` → `test_phase1_14_examples_unchanged.py`
- Add the 4 Phase 14 examples to the regression list (`persistent_kernel_server`, `dynamic_parallelism_recursive`, `persistent_work_queue`, `persistent_pipeline`)

### Test count target
717 (Phase 14 baseline) → ~745 (+28).

---

## 9. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1** Stream capture core + basic example | Stream.begin/end_capture + capture-translation of kernel/memcpy/record/wait + Graph.is_captured + stream_capture_basic | `M1-phase15-complete` |
| **M2** Multi-stream capture session | CaptureSession + cross-stream event-edge translation + stream_capture_multi_stream example | `M2-phase15-complete` |
| **M3** Conditional node + example | ConditionalNodeArgs + Graph.add_conditional_node + GraphExec branch + ConditionalBranch trace event + graph_conditional_branch | `M3-phase15-complete` |
| **M4** While node + example + 4 metrics | WhileNodeArgs + Graph.add_while_node + GraphExec loop + LoopIteration trace event + graph_while_loop + 4 analysis metrics | `M4-phase15-complete` |
| **M5** Tutorials + microbench + regression rename + README v15 + ship | 4 chapters + microbench + Phase 1-14 regression rename + README | `phase15-complete` |

Estimated 22 tasks total.

---

## 10. File list

### New files
```
gpusim/core/capture_session.py         # CaptureSession (M2)
examples/stream_capture_basic/         # 5 files (M1)
examples/stream_capture_multi_stream/  # 5 files (M2)
examples/graph_conditional_branch/     # 5 files (M3)
examples/graph_while_loop/             # 5 files (M4)
docs/tutorial/58-stream-capture-basic.md
docs/tutorial/59-stream-capture-multi-stream.md
docs/tutorial/60-graph-conditional-branch.md
docs/tutorial/61-graph-while-loop.md
tests/unit/core/test_stream_capture.py
tests/unit/core/test_capture_session.py
tests/unit/graph/test_conditional_node.py
tests/unit/graph/test_while_node.py
tests/unit/analysis/test_phase15_metrics.py
tests/parity/test_stream_capture_basic.py
tests/parity/test_stream_capture_multi_stream.py
tests/parity/test_graph_conditional_branch.py
tests/parity/test_graph_while_loop.py
tests/microbench/test_phase15_facts.py
tests/microbench/test_phase15_runtime.py
tests/reference/data/{4 example names}.ref.json
```

### Modified files
```
gpusim/core/stream.py                  # +capture state +begin/end_capture +op interception
gpusim/graph/node.py                   # +ConditionalNodeArgs +WhileNodeArgs +GraphNode fields
gpusim/graph/graph.py                  # +is_captured +add_conditional_node +add_while_node
gpusim/graph/exec.py                   # +conditional/while branches in launch
gpusim/graph/__init__.py               # export new node types
gpusim/trace/events.py                 # +StreamCaptureBegin +StreamCaptureEnd +ConditionalBranch +LoopIteration
gpusim/trace/recorder.py               # +recorder methods for the 4 new events
gpusim/analysis/metrics.py             # +4 metrics
tests/parity/test_phase1_13_examples_unchanged.py → test_phase1_14_examples_unchanged.py
tests/reference/gen_reference.py       # +4 kernel names
README.md                              # v15 — Phase 15 capabilities
```

---

## 11. Backward compatibility

- All Phase 1-14 examples + tests pass unchanged.
- Stream capture is opt-in: existing imperative stream code is unaffected (`_capturing=False` default).
- Conditional + while nodes are additive; existing graph node paths unchanged.
- `Graph.is_captured` defaults to False; existing manually-built graphs retain `False`.

---

## 12. Acceptance criteria

Phase 15 ships when:

- [ ] All 5 milestone tags present (`M1-phase15-complete` ... `M4-phase15-complete`, `phase15-complete`)
- [ ] All 4 examples run cleanly
- [ ] All 4 parity tests pass
- [ ] Microbench: capture appends only (no exec), capture-replay equivalence, conditional taken recorded, while max_iterations enforced
- [ ] Phase 1-14 regression test (renamed) passes
- [ ] Test count: 717 → ~745 (+28)
- [ ] README v15 documents Phase 15 capabilities
