# gpusim Phase 11 — CUDA Graphs (DAG of kernels, capture/instantiate/replay)

> **Status:** Brainstormed 2026-05-10. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- Add **CUDA Graphs equivalent**: a DAG of kernel/memcpy/event nodes that can be captured once and replayed many times to amortize launch overhead.
- Implement **explicit construction API**: `gpusim.Graph()` + `add_kernel_node()` + `add_memcpy_node()` + `add_event_node()` + `add_dependency()` + `instantiate()` + `launch()`.
- Implement **stream capture mode**: `Stream.begin_capture()` / `Stream.end_capture()` intercepts subsequent `launch()` calls and records them as kernel nodes into a graph (no execution at capture time).
- Support **3 node types**: kernel (PTX launch), memcpy (host↔device or device↔device), event (record/wait — reuses Phase 8 Event).
- Support **dual dependency model**: explicit edges via `add_dependency(parent, child)` (builder) + implicit edges from stream capture order + cross-stream events (capture mode).
- Replay semantics: each `Graph.launch()` re-executes the full graph against current parameter buffers; cycles measured per replay.
- 4 examples + 4 tutorial chapters demonstrating explicit build, capture, replay-perf-amortization, iterative training step.
- 3 new metrics + 1 new trace event + HTML §35 + Perfetto Graph swimlane.
- 100% backward compatible: Phase 1-10 examples unchanged.

### Non-goals (deferred to Phase 12+)
- **Child graphs** (graph nesting / `cudaGraphAddChildGraphNode`).
- **Host callback nodes** (`cudaGraphAddHostNode`) — ours has no host callback model.
- **Graph update API** (`cudaGraphExecUpdate` — modify executable graph between replays).
- **Memset nodes** (covered by user-provided kernels in Phase 11).
- **Multi-GPU graphs** (graph spans multiple GPUs / NCCL collectives in graph).
- **Conditional graph nodes** (CUDA 12+ if/while nodes).

---

## 2. Architecture

```
gpusim.Graph (NEW gpusim/graph/graph.py)
├── nodes: list[GraphNode]
├── edges: list[(parent_id, child_id)]
├── add_kernel_node(ptx, grid, block, params, kernel_name) -> node_id
├── add_memcpy_node(src, dst, n_bytes) -> node_id
├── add_event_node(event, op="record"|"wait") -> node_id
├── add_dependency(parent_id, child_id) -> None
└── instantiate(config) -> GraphExec

gpusim.GraphExec (NEW)
├── graph: Graph (frozen at instantiate time)
├── topo_order: list[node_id] (computed at instantiate)
└── launch(params_override=None) -> int (cycles)

gpusim.Stream extensions (gpusim/api.py)
├── begin_capture() -> None  (start recording subsequent .launch into a graph)
├── end_capture() -> Graph   (return captured graph)
└── _captured_graph: Graph | None  (state during capture)

GraphNode (dataclass, polymorphic via type field)
├── node_id: int
├── type: str  ("kernel" | "memcpy" | "event")
├── kernel_args: KernelNodeArgs | None
├── memcpy_args: MemcpyNodeArgs | None
└── event_args: EventNodeArgs | None
```

### Key invariants
- During `Stream.begin_capture()` ... `end_capture()`, `Stream.launch()` does NOT execute the kernel — it records a kernel node into the in-progress capture graph.
- Implicit dependency: each capture-mode kernel node depends on the previous capture-mode node in the same stream.
- Cross-stream sync via Phase 8 events ALSO captured: `Stream.record(ev)` adds an event node; `Stream.wait(ev)` adds an event-wait node + creates implicit dependency from the matching `record(ev)` node.
- `Graph.instantiate()` validates DAG (no cycles), computes topological order.
- `GraphExec.launch()` walks topological order, executes each node, accumulates cycles.
- Each `launch()` replay re-executes from current params (replay reads buffer contents at launch time, NOT at capture time).
- Multiple `launch()` calls amortize "graph instantiation overhead" — measured as cycles savings vs N separate Stream.launch calls.
- Phase 1-10 unchanged: capturing must be explicit; default Stream.launch behavior is identical.

---

## 3. Data model

### 3.1 GraphNode types (`gpusim/graph/node.py`)

```python
@dataclass
class KernelNodeArgs:
    ptx_src: str
    grid: tuple
    block: tuple
    params: dict          # ndarray references; replay reads at launch time
    kernel_name: str = "<unnamed>"


@dataclass
class MemcpyNodeArgs:
    src: object           # ndarray
    dst: object           # ndarray
    n_bytes: int


@dataclass
class EventNodeArgs:
    event: object         # gpusim.api.Event (Phase 8)
    op: str               # "record" | "wait"


@dataclass
class GraphNode:
    node_id: int
    type: str             # "kernel" | "memcpy" | "event"
    kernel_args: KernelNodeArgs | None = None
    memcpy_args: MemcpyNodeArgs | None = None
    event_args: EventNodeArgs | None = None
```

### 3.2 Graph class (`gpusim/graph/graph.py`)

```python
@dataclass
class Graph:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)   # [(parent_id, child_id), ...]
    _next_id: int = 0
    
    def add_kernel_node(self, *, ptx_src, grid, block, params,
                          kernel_name="<unnamed>") -> int:
        node_id = self._next_id; self._next_id += 1
        args = KernelNodeArgs(ptx_src=ptx_src, grid=grid, block=block,
                                params=params, kernel_name=kernel_name)
        self.nodes.append(GraphNode(node_id=node_id, type="kernel", kernel_args=args))
        return node_id
    
    def add_memcpy_node(self, *, src, dst, n_bytes) -> int: ...
    def add_event_node(self, *, event, op) -> int: ...
    
    def add_dependency(self, parent_id: int, child_id: int) -> None:
        if parent_id == child_id:
            raise ValueError("self-dependency not allowed")
        self.edges.append((parent_id, child_id))
    
    def instantiate(self, config) -> "GraphExec":
        from gpusim.graph.exec import GraphExec
        return GraphExec.from_graph(self, config)
```

### 3.3 GraphExec class (`gpusim/graph/exec.py`)

```python
@dataclass
class GraphExec:
    graph: Graph
    topo_order: list                # list of node_ids in valid execution order
    config: object
    
    @classmethod
    def from_graph(cls, graph: Graph, config) -> "GraphExec":
        topo = _topological_sort(graph.nodes, graph.edges)
        return cls(graph=graph, topo_order=topo, config=config)
    
    def launch(self) -> int:
        """Execute all nodes in topological order. Returns total cycles."""
        from gpusim.frontend.parser import parse
        from gpusim.api import Stream, synchronize
        # Group consecutive kernel nodes into a single stream for execution
        # Memcpy nodes execute on host (numpy slicing — instantaneous + small fixed cost)
        # Event nodes use Phase 8 Stream.record/wait
        total_cycles = 0
        for node_id in self.topo_order:
            node = next(n for n in self.graph.nodes if n.node_id == node_id)
            if node.type == "kernel":
                # Execute kernel via Stream.launch + synchronize
                s = Stream()
                args = node.kernel_args
                s.launch(ptx_src=args.ptx_src, grid=args.grid, block=args.block,
                          params=args.params, kernel_name=args.kernel_name,
                          config=self.config)
                res = synchronize(streams=[s], config=self.config)
                total_cycles += res.streams[s.stream_id][0].metrics.get("cycles", 0)
            elif node.type == "memcpy":
                args = node.memcpy_args
                args.dst[:] = args.src    # functional memcpy
                total_cycles += 100        # fixed overhead per memcpy
            elif node.type == "event":
                # Event nodes are functional in graph mode (signal/wait modeled via topo order)
                pass
        return total_cycles


def _topological_sort(nodes, edges) -> list:
    """Kahn's algorithm. Raises if graph has cycles."""
    from collections import defaultdict, deque
    in_degree = defaultdict(int)
    children = defaultdict(list)
    node_ids = {n.node_id for n in nodes}
    for parent, child in edges:
        in_degree[child] += 1
        children[parent].append(child)
    queue = deque([nid for nid in node_ids if in_degree[nid] == 0])
    out = []
    while queue:
        nid = queue.popleft()
        out.append(nid)
        for c in children[nid]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    if len(out) != len(node_ids):
        raise ValueError("graph has a cycle")
    return out
```

### 3.4 Stream capture mode (`gpusim/api.py`)

```python
class Stream:
    # ... existing fields ...
    _captured_graph: object | None = None    # NEW Phase 11
    _capture_last_node: int | None = None    # NEW Phase 11
    
    def begin_capture(self) -> None:
        """Start recording subsequent .launch() into a fresh Graph."""
        from gpusim.graph.graph import Graph
        self._captured_graph = Graph()
        self._capture_last_node = None
    
    def end_capture(self) -> "Graph":
        """Stop capture; return the recorded Graph."""
        g = self._captured_graph
        self._captured_graph = None
        self._capture_last_node = None
        return g
    
    def launch(self, ptx_src, grid, block, params, *,
                kernel_name="<unnamed>", config=None) -> None:
        """Append GridLaunch — OR record kernel node if capturing."""
        if self._captured_graph is not None:
            # Capture mode: record as kernel node + implicit dep
            node_id = self._captured_graph.add_kernel_node(
                ptx_src=ptx_src, grid=grid, block=block, params=params,
                kernel_name=kernel_name,
            )
            if self._capture_last_node is not None:
                self._captured_graph.add_dependency(self._capture_last_node, node_id)
            self._capture_last_node = node_id
            return
        # Normal mode (Phase 7-10 behavior)
        # ... existing pending.append(...) ...
```

### 3.5 New trace event `GraphLaunch` (`gpusim/trace/events.py`)

```python
@dataclass(frozen=True)
class GraphLaunch:
    graph_id: int
    n_nodes: int
    n_edges: int
    launch_index: int        # 0 = first replay, 1 = second, etc.
    start_cycle: int
    end_cycle: int
```

---

## 4. Capture mode details

### 4.1 Implicit dependencies
- Each kernel node added during capture depends on the **previous kernel node in the same stream**.
- Cross-stream dependencies via Phase 8 events: `Stream.record(ev)` adds an event node; subsequent `Stream.wait(ev)` on a different stream adds a wait node + dependency from the recording node.

### 4.2 Capture restrictions
- During capture, `synchronize()` is forbidden — raises `RuntimeError("cannot synchronize during capture")`.
- During capture, `gpusim.run()` (single-shot) is forbidden — same error.

### 4.3 Capture example

```python
s = gpusim.Stream()
s.begin_capture()
s.launch(ptx_a, ...)    # → kernel node 0
s.launch(ptx_b, ...)    # → kernel node 1, depends on node 0
g = s.end_capture()     # returns Graph with 2 nodes + 1 edge
exec = g.instantiate(cfg)
for i in range(100):    # replay 100 times
    cycles = exec.launch()
```

---

## 5. Replay semantics

- Each `GraphExec.launch()` walks `topo_order` and executes nodes sequentially.
- Kernel nodes: PTX is parsed once per launch (or cached at instantiate); params are read at launch time (so callers can mutate buffers between launches).
- Memcpy nodes: numpy slicing copy + fixed 100-cycle overhead.
- Event nodes: ordering is enforced via topological order; cycle accounting follows.
- Phase 11 minimal: kernel execution still uses `Stream.launch + synchronize` per node; future iteration could batch nodes for better simulator throughput.
- Cycle counting: total cycles = sum over nodes of (kernel cycles + memcpy overhead + event ordering overhead).

---

## 6. Trace + Analysis

### 6.1 Trace
- `GraphExec.launch()` records a `GraphLaunch` event with start/end cycles + launch_index counter (incremented per replay on this exec).

### 6.2 3 new metrics

```python
def graph_replay_amortization(graph_launch_df, single_kernel_baseline_cycles: int) -> dict:
    """Cycles per replay vs equivalent N independent kernel launches.
    Higher amortization = graph saves more overhead per replay."""

def graph_dag_depth(graph) -> int:
    """Length of longest dependency chain in the graph."""

def graph_node_type_breakdown(graph) -> dict:
    """Count of nodes by type: {kernel: N, memcpy: M, event: K}."""
```

### 6.3 Result API extensions

```python
class GraphExecResult:
    graph_id: int
    launch_cycles: list[int]            # cycles per replay
    
    def replay_amortization(self, baseline_cycles: int) -> dict: ...
    def dag_depth(self) -> int: ...
    def node_type_breakdown(self) -> dict: ...
```

---

## 7. Viz

### 7.1 HTML §35 — Graph DAG visualization
`gpusim/viz/html_report.py` adds `_render_graph_dag(rec)` that produces an HTML/SVG node-edge representation of the captured graph: nodes labeled by type+kernel_name, edges drawn as arrows.

### 7.2 Perfetto Graph swimlane
`gpusim/viz/perfetto.py` adds `pid="Graph"` swimlane: each graph launch is a duration event spanning launch_index. Inside, sub-events show per-node cycles.

---

## 8. Examples (4)

### 8.1 `graph_explicit_build/`
- Explicit build: 3-kernel chain (vec_add → vec_mul → vec_sub) using `add_kernel_node` + `add_dependency`.
- **Verifies:** topological order correct; outputs match sequential equivalent.
- 5 files (kernel.ptx + reference.py + run.py + README.md + __init__.py).

### 8.2 `graph_capture_from_stream/`
- Use `Stream.begin_capture` to record 3 launches into a graph.
- **Verifies:** captured graph has 3 nodes + 2 edges (linear chain).

### 8.3 `graph_replay_perf/` ⭐
- Capture a graph; launch 10 times; compare total cycles vs 10 independent Stream.launch.
- **Verifies:** graph replay shows lower per-replay overhead.

### 8.4 `graph_iterative_train_step/`
- Build a graph for one training step (forward + grad + update); replay 5 times to simulate epochs.
- **Verifies:** correctness across replays; demonstrates production graph use.

---

## 9. Tutorials

`docs/tutorial/` chapters 44-47:
- **44-cuda-graphs-explicit-build.md** — example 1
- **45-stream-capture-to-graph.md** — example 2
- **46-graph-replay-amortization.md** — example 3 ⭐
- **47-graph-iterative-training.md** — example 4

---

## 10. Testing strategy

### Unit tests (~12 new)
- `tests/unit/graph/test_graph_construction.py` — Graph.add_kernel_node, add_dependency, validation
- `tests/unit/graph/test_graph_exec.py` — GraphExec topological sort, launch
- `tests/unit/graph/test_graph_capture.py` — Stream.begin_capture/end_capture
- `tests/unit/graph/test_graph_node_types.py` — kernel + memcpy + event node correctness
- `tests/unit/graph/test_graph_dag_validation.py` — cycle detection
- `tests/unit/trace/test_graph_launch_event.py` — GraphLaunch recorder
- `tests/unit/analysis/test_phase11_metrics.py` — 3 new metrics

### Parity tests (~4 — one per example)

### Microbench
- `test_phase11_facts.py` (fast):
  - Topological sort correctness for known graphs
  - Capture order produces expected dependency chain
  - Replay 5x produces same cycles each time (replay determinism)
- `test_phase11_runtime.py` (slow): 4 examples each under 60s

### Regression
- Rename `tests/parity/test_phase1_9_examples_unchanged.py` → `test_phase1_10_examples_unchanged.py`
- Add 4 Phase 10 examples to the regression list

### Test count target
596 (Phase 10 baseline) → ~625 (+29).

---

## 11. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1** Graph + GraphNode + topological sort + node types | Graph class + 3 node types + add_dependency + cycle validation | `M1-phase11-complete` |
| **M2** GraphExec + execute kernel/memcpy/event nodes + 1 example | GraphExec.launch executes nodes; graph_explicit_build example | `M2-phase11-complete` |
| **M3** Stream capture mode + capture example | Stream.begin_capture/end_capture + implicit deps + graph_capture_from_stream example | `M3-phase11-complete` |
| **M4** GraphLaunch trace + 3 metrics + 2 examples | GraphLaunch event + amortization metric + dag_depth + node_breakdown + graph_replay_perf + graph_iterative_train_step | `M4-phase11-complete` |
| **M5** Viz + tutorials + microbench + regression + README v11 + ship | HTML §35 + Perfetto Graph swimlane + 4 chapters + microbench + Phase 1-10 regression rename + README v11 | `phase11-complete` |

Estimated 28 tasks total.

---

## 12. File list

### New files
```
gpusim/graph/__init__.py
gpusim/graph/node.py             # GraphNode + 3 args dataclasses
gpusim/graph/graph.py            # Graph class
gpusim/graph/exec.py             # GraphExec + topological sort
examples/graph_explicit_build/   # 5 files (M2)
examples/graph_capture_from_stream/  # 5 files (M3)
examples/graph_replay_perf/      # 5 files (M4)
examples/graph_iterative_train_step/ # 5 files (M4)
docs/tutorial/44-cuda-graphs-explicit-build.md
docs/tutorial/45-stream-capture-to-graph.md
docs/tutorial/46-graph-replay-amortization.md
docs/tutorial/47-graph-iterative-training.md
tests/unit/graph/__init__.py
tests/unit/graph/test_graph_construction.py
tests/unit/graph/test_graph_exec.py
tests/unit/graph/test_graph_capture.py
tests/unit/graph/test_graph_node_types.py
tests/unit/graph/test_graph_dag_validation.py
tests/unit/trace/test_graph_launch_event.py
tests/unit/analysis/test_phase11_metrics.py
tests/parity/test_graph_explicit_build.py
tests/parity/test_graph_capture_from_stream.py
tests/parity/test_graph_replay_perf.py
tests/parity/test_graph_iterative_train_step.py
tests/microbench/test_phase11_facts.py
tests/microbench/test_phase11_runtime.py
tests/reference/data/{4 example names}.ref.json
```

### Modified files
```
gpusim/api.py                    # Stream.begin_capture/end_capture + Stream.launch capture branch
gpusim/__init__.py               # Export Graph, GraphExec
gpusim/trace/events.py           # +GraphLaunch
gpusim/trace/recorder.py         # +graph_launch method
gpusim/trace/writer.py           # +graph_launch.parquet
gpusim/analysis/metrics.py       # +3 metrics
gpusim/viz/notebook.py           # +graph_launch_events_dataframe
gpusim/viz/html_report.py        # +§35 graph DAG render
gpusim/viz/_template.html.j2     # +§35 block
gpusim/viz/perfetto.py           # +Graph swimlane
tests/parity/test_phase1_9_examples_unchanged.py → test_phase1_10_examples_unchanged.py
tests/reference/gen_reference.py # +4 kernel names
README.md                        # v11 — Phase 11 capabilities
```

---

## 13. Backward compatibility

- All Phase 1-10 examples + tests pass unchanged.
- `Stream.launch` behavior identical when not in capture mode.
- `gpusim.run`, `gpusim.synchronize` — unchanged.
- New imports `Graph`, `GraphExec` are additive.
- `Stream.begin_capture()` / `end_capture()` are new methods (no impact unless called).

---

## 14. Open questions / future work

- **Child graphs** — Phase 12: `add_child_graph_node(graph)` for nested graphs.
- **Graph update API** — Phase 12: `GraphExec.update_node_params(node_id, ...)` between replays.
- **Conditional graph nodes** — Phase 13: if/while inside graph (CUDA 12.4+).
- **Memset nodes** — currently absorbed into user kernels.
- **Multi-GPU graphs** — graph spans GPUs / contains NCCL collective nodes.
- **PyTorch torch.cuda.graphs wrapper** — adapter on top of Phase 11 API.

---

## 15. Acceptance criteria

Phase 11 ships when:

- [ ] All 5 milestone tags present (`M1-phase11-complete` ... `M4-phase11-complete`, `phase11-complete`)
- [ ] All 4 examples run cleanly (`python examples/<name>/run.py`)
- [ ] All 4 parity tests pass
- [ ] Microbench: graph DAG depth = expected for known graphs; replay 5x deterministic
- [ ] HTML report shows §35 when graph_launch_events present
- [ ] Perfetto JSON has Graph swimlane
- [ ] Phase 1-10 regression test (renamed) passes
- [ ] Test count: 596 → ~625 (+29)
- [ ] README v11 documents Phase 11 capabilities
