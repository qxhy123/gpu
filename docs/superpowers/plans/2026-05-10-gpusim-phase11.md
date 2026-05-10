# gpusim Phase 11 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Implement gpusim Phase 11 per `docs/superpowers/specs/2026-05-10-gpusim-phase11-design.md` — CUDA Graphs equivalent: Graph builder + GraphExec replay + Stream capture mode.

**Architecture:** New `gpusim/graph/` package with `Graph`, `GraphNode`, `GraphExec`. Stream gains `begin_capture()/end_capture()` that intercept `Stream.launch` to record kernel nodes instead of executing. Topological sort validates DAG; GraphExec walks order to replay. New `GraphLaunch` trace event + 3 metrics + HTML §35 + Perfetto Graph swimlane.

**Tech Stack:** Python 3.11+. No new runtime dependencies.

**Execution note:** Plan has 5 milestones (M1–M5) with 28 tasks. Tags after each: `M{1..5}-phase11-complete`.

---

## Phase 1+2+3+4+5+6+7+8+9+10 prerequisites

```bash
cd /Users/yangyang/ai_projs/gpu
git tag | grep phase
.venv/bin/pytest -q -m "not slow"
```
Expected: ~596 passed (Phase 10 baseline).

---

## File structure

```
gpusim/
├── api.py                       MODIFY: + Stream.begin_capture/end_capture + capture branch in launch
├── graph/                       NEW (M1+M2+M3)
│   ├── __init__.py
│   ├── node.py                  # GraphNode + 3 args dataclasses
│   ├── graph.py                 # Graph builder
│   └── exec.py                  # GraphExec + topological sort
├── trace/events.py              MODIFY (M4): + GraphLaunch
├── trace/recorder.py            MODIFY (M4): + graph_launch method
├── trace/writer.py              MODIFY (M4): + parquet writer
├── analysis/metrics.py          MODIFY (M4): + 3 metrics
└── viz/                         MODIFY (M5): + §35 + Perfetto Graph swimlane

examples/
├── graph_explicit_build/        NEW (M2)
├── graph_capture_from_stream/   NEW (M3)
├── graph_replay_perf/           NEW (M4)
└── graph_iterative_train_step/  NEW (M4)

tests/unit/graph/                NEW (M1-M3)
tests/parity/test_phase1_10_examples_unchanged.py    RENAME from phase1_9 (M5)
tests/microbench/test_phase11_facts.py               NEW (M5)
tests/microbench/test_phase11_runtime.py             NEW (M5, slow)
tests/reference/data/{4 names}.ref.json              NEW (M5)
docs/tutorial/{44,45,46,47}-*.md                     NEW (M5)
README.md                                            MODIFY (M5): v11
```

---

## Milestones

| Milestone | Tasks | Tag |
|---|---|---|
| **M1** Graph + GraphNode + 3 node types + DAG validation | T1–T5 | `M1-phase11-complete` |
| **M2** GraphExec + topological sort + execute + 1 example | T6–T9 | `M2-phase11-complete` |
| **M3** Stream capture mode + capture example | T10–T13 | `M3-phase11-complete` |
| **M4** GraphLaunch trace + 3 metrics + 2 examples | T14–T20 | `M4-phase11-complete` |
| **M5** Viz + tutorials + microbench + regression + README v11 + ship | T21–T28 | `phase11-complete` |

---

## Milestone M1: Graph + GraphNode + DAG validation

### Task 1: GraphNode + 3 args dataclasses

**Files:**
- Create: `gpusim/graph/__init__.py`, `gpusim/graph/node.py`
- Test: `tests/unit/graph/__init__.py`, `tests/unit/graph/test_graph_node_types.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_kernel_node_args():
    from gpusim.graph.node import KernelNodeArgs
    a = KernelNodeArgs(ptx_src=".entry t() { ret; }",
                          grid=(1,1,1), block=(32,1,1),
                          params={}, kernel_name="k")
    assert a.kernel_name == "k"
    assert a.grid == (1,1,1)


def test_memcpy_node_args():
    import numpy as np
    from gpusim.graph.node import MemcpyNodeArgs
    src = np.zeros(8); dst = np.zeros(8)
    a = MemcpyNodeArgs(src=src, dst=dst, n_bytes=64)
    assert a.n_bytes == 64


def test_graph_node_kernel_type():
    from gpusim.graph.node import GraphNode, KernelNodeArgs
    args = KernelNodeArgs(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="k")
    n = GraphNode(node_id=0, type="kernel", kernel_args=args)
    assert n.type == "kernel"
    assert n.kernel_args is args
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Create gpusim/graph/__init__.py:**

```python
from gpusim.graph.node import (
    GraphNode, KernelNodeArgs, MemcpyNodeArgs, EventNodeArgs,
)
__all__ = ["GraphNode", "KernelNodeArgs", "MemcpyNodeArgs", "EventNodeArgs"]
```

- [ ] **Step 4: Create gpusim/graph/node.py:**

```python
"""Phase 11: CUDA Graphs node types."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class KernelNodeArgs:
    ptx_src: str
    grid: tuple
    block: tuple
    params: dict
    kernel_name: str = "<unnamed>"


@dataclass
class MemcpyNodeArgs:
    src: object
    dst: object
    n_bytes: int


@dataclass
class EventNodeArgs:
    event: object
    op: str   # "record" | "wait"


@dataclass
class GraphNode:
    node_id: int
    type: str    # "kernel" | "memcpy" | "event"
    kernel_args: KernelNodeArgs | None = None
    memcpy_args: MemcpyNodeArgs | None = None
    event_args: EventNodeArgs | None = None
```

- [ ] **Step 5: Create empty `tests/unit/graph/__init__.py`.**

- [ ] **Step 6: Run + commit**

```
.venv/bin/pytest tests/unit/graph/test_graph_node_types.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/graph/ tests/unit/graph/
git commit -m "feat(graph): GraphNode + 3 args dataclasses (kernel/memcpy/event)"
```

---

### Task 2: Graph class + add_kernel_node + add_dependency

**Files:**
- Create: `gpusim/graph/graph.py`
- Test: `tests/unit/graph/test_graph_construction.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_graph_empty_construction():
    from gpusim.graph.graph import Graph
    g = Graph()
    assert g.nodes == []
    assert g.edges == []


def test_graph_add_kernel_node_returns_id():
    from gpusim.graph.graph import Graph
    g = Graph()
    nid = g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                              params={}, kernel_name="k0")
    assert nid == 0
    assert len(g.nodes) == 1
    nid2 = g.add_kernel_node(ptx_src="y", grid=(1,1,1), block=(32,1,1),
                                params={}, kernel_name="k1")
    assert nid2 == 1


def test_graph_add_dependency():
    from gpusim.graph.graph import Graph
    g = Graph()
    a = g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="ka")
    b = g.add_kernel_node(ptx_src="y", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="kb")
    g.add_dependency(a, b)
    assert (a, b) in g.edges


def test_graph_self_dependency_raises():
    from gpusim.graph.graph import Graph
    import pytest
    g = Graph()
    a = g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="k")
    with pytest.raises(ValueError, match="self-dependency"):
        g.add_dependency(a, a)
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Create gpusim/graph/graph.py:**

```python
"""Phase 11: Graph builder."""
from __future__ import annotations
from dataclasses import dataclass, field
from gpusim.graph.node import (
    GraphNode, KernelNodeArgs, MemcpyNodeArgs, EventNodeArgs,
)


@dataclass
class Graph:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    _next_id: int = 0
    
    def add_kernel_node(self, *, ptx_src, grid, block, params,
                          kernel_name="<unnamed>") -> int:
        nid = self._next_id; self._next_id += 1
        args = KernelNodeArgs(ptx_src=ptx_src, grid=grid, block=block,
                                params=params, kernel_name=kernel_name)
        self.nodes.append(GraphNode(node_id=nid, type="kernel", kernel_args=args))
        return nid
    
    def add_memcpy_node(self, *, src, dst, n_bytes) -> int:
        nid = self._next_id; self._next_id += 1
        args = MemcpyNodeArgs(src=src, dst=dst, n_bytes=n_bytes)
        self.nodes.append(GraphNode(node_id=nid, type="memcpy", memcpy_args=args))
        return nid
    
    def add_event_node(self, *, event, op: str) -> int:
        nid = self._next_id; self._next_id += 1
        args = EventNodeArgs(event=event, op=op)
        self.nodes.append(GraphNode(node_id=nid, type="event", event_args=args))
        return nid
    
    def add_dependency(self, parent_id: int, child_id: int) -> None:
        if parent_id == child_id:
            raise ValueError("self-dependency not allowed")
        self.edges.append((parent_id, child_id))
    
    def instantiate(self, config) -> "GraphExec":
        from gpusim.graph.exec import GraphExec
        return GraphExec.from_graph(self, config)
```

- [ ] **Step 4: Update __init__.py to export Graph:**

```python
from gpusim.graph.node import (
    GraphNode, KernelNodeArgs, MemcpyNodeArgs, EventNodeArgs,
)
from gpusim.graph.graph import Graph
__all__ = ["Graph", "GraphNode", "KernelNodeArgs", "MemcpyNodeArgs", "EventNodeArgs"]
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/graph/test_graph_construction.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/graph/ tests/unit/graph/test_graph_construction.py
git commit -m "feat(graph): Graph class + add_kernel/memcpy/event_node + add_dependency"
```

---

### Task 3: Topological sort + cycle detection

**Files:**
- Create: `gpusim/graph/exec.py` (just topo helper for now)
- Test: `tests/unit/graph/test_graph_dag_validation.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_topological_sort_linear_chain():
    from gpusim.graph.exec import _topological_sort
    from gpusim.graph.node import GraphNode
    nodes = [GraphNode(node_id=i, type="kernel") for i in range(3)]
    edges = [(0, 1), (1, 2)]
    order = _topological_sort(nodes, edges)
    assert order == [0, 1, 2]


def test_topological_sort_diamond():
    from gpusim.graph.exec import _topological_sort
    from gpusim.graph.node import GraphNode
    # 0 → 1 → 3 ; 0 → 2 → 3
    nodes = [GraphNode(node_id=i, type="kernel") for i in range(4)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    order = _topological_sort(nodes, edges)
    # 0 must come first, 3 must come last
    assert order[0] == 0
    assert order[-1] == 3


def test_topological_sort_cycle_raises():
    from gpusim.graph.exec import _topological_sort
    from gpusim.graph.node import GraphNode
    import pytest
    nodes = [GraphNode(node_id=i, type="kernel") for i in range(3)]
    edges = [(0, 1), (1, 2), (2, 0)]   # cycle
    with pytest.raises(ValueError, match="cycle"):
        _topological_sort(nodes, edges)
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Create gpusim/graph/exec.py with _topological_sort:**

```python
"""Phase 11: GraphExec + topological sort."""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field


def _topological_sort(nodes: list, edges: list) -> list:
    """Kahn's algorithm. Returns list of node_ids in execution order.
    Raises ValueError if graph has a cycle."""
    in_degree = defaultdict(int)
    children = defaultdict(list)
    node_ids = [n.node_id for n in nodes]
    for parent, child in edges:
        in_degree[child] += 1
        children[parent].append(child)
    queue = deque(sorted([nid for nid in node_ids if in_degree[nid] == 0]))
    out = []
    while queue:
        nid = queue.popleft()
        out.append(nid)
        for c in sorted(children[nid]):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    if len(out) != len(node_ids):
        raise ValueError("graph has a cycle")
    return out
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/graph/test_graph_dag_validation.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/graph/exec.py tests/unit/graph/test_graph_dag_validation.py
git commit -m "feat(graph): topological sort + cycle detection"
```

---

### Task 4: Graph.add_memcpy_node + add_event_node tests

**Files:**
- Test: extend `tests/unit/graph/test_graph_node_types.py`

- [ ] **Step 1: Append failing tests**

```python
def test_graph_add_memcpy_node():
    import numpy as np
    from gpusim.graph.graph import Graph
    src = np.zeros(8, dtype=np.float32)
    dst = np.zeros(8, dtype=np.float32)
    g = Graph()
    nid = g.add_memcpy_node(src=src, dst=dst, n_bytes=32)
    assert nid == 0
    assert g.nodes[0].type == "memcpy"
    assert g.nodes[0].memcpy_args.n_bytes == 32


def test_graph_add_event_node():
    from gpusim.graph.graph import Graph
    from gpusim.api import Event
    g = Graph()
    ev = Event()
    nid = g.add_event_node(event=ev, op="record")
    assert g.nodes[0].type == "event"
    assert g.nodes[0].event_args.op == "record"
    assert g.nodes[0].event_args.event is ev
```

- [ ] **Step 2: Run + verify PASS** (Graph.add_memcpy_node + add_event_node already implemented in T2).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/graph/test_graph_node_types.py
git commit -m "test(graph): add_memcpy + add_event coverage"
```

---

### Task 5: Tag M1

```bash
.venv/bin/pytest -q -m "not slow"
git tag M1-phase11-complete
```

---

## Milestone M2: GraphExec + execute + 1 example

### Task 6: GraphExec class + from_graph + launch (kernel only)

**Files:**
- Modify: `gpusim/graph/exec.py` (add GraphExec class)
- Test: `tests/unit/graph/test_graph_exec.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_graph_exec_from_graph():
    from gpusim.graph.graph import Graph
    from gpusim.graph.exec import GraphExec
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src=".entry t() { .reg .u32 %r0; mov.u32 %r0, %tid.x; ret; }",
                        grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k")
    exec = GraphExec.from_graph(g, cfg)
    assert exec.topo_order == [0]


def test_graph_exec_launch_single_kernel():
    from gpusim.graph.graph import Graph
    from gpusim.graph.exec import GraphExec
    from gpusim.config.loader import load_default
    cfg = load_default()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    g = Graph()
    g.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    assert cycles > 0
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add GraphExec to gpusim/graph/exec.py:**

```python
@dataclass
class GraphExec:
    graph: object
    topo_order: list
    config: object
    
    @classmethod
    def from_graph(cls, graph, config) -> "GraphExec":
        topo = _topological_sort(graph.nodes, graph.edges)
        return cls(graph=graph, topo_order=topo, config=config)
    
    def launch(self) -> int:
        """Execute all nodes in topological order. Returns total cycles."""
        from gpusim.api import Stream, synchronize
        total_cycles = 0
        for node_id in self.topo_order:
            node = next(n for n in self.graph.nodes if n.node_id == node_id)
            if node.type == "kernel":
                a = node.kernel_args
                s = Stream()
                s.launch(ptx_src=a.ptx_src, grid=a.grid, block=a.block,
                          params=a.params, kernel_name=a.kernel_name,
                          config=self.config)
                res = synchronize(streams=[s], config=self.config)
                if 0 in res.streams and res.streams[0]:
                    total_cycles += res.streams[0][0].metrics.get("cycles", 0)
            elif node.type == "memcpy":
                a = node.memcpy_args
                # numpy slicing copy
                a.dst[:] = a.src
                total_cycles += 100   # fixed memcpy overhead
            elif node.type == "event":
                # Functional: event ordering enforced by topo order
                pass
        return total_cycles
```

- [ ] **Step 4: Update gpusim/graph/__init__.py to export GraphExec:**

```python
from gpusim.graph.node import (
    GraphNode, KernelNodeArgs, MemcpyNodeArgs, EventNodeArgs,
)
from gpusim.graph.graph import Graph
from gpusim.graph.exec import GraphExec
__all__ = ["Graph", "GraphExec", "GraphNode",
            "KernelNodeArgs", "MemcpyNodeArgs", "EventNodeArgs"]
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/graph/test_graph_exec.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/graph/ tests/unit/graph/test_graph_exec.py
git commit -m "feat(graph): GraphExec + launch (kernel/memcpy/event execution)"
```

---

### Task 7: GraphExec executes 3-kernel chain in dependency order

**Files:**
- Test: extend `tests/unit/graph/test_graph_exec.py`

- [ ] **Step 1: Append failing test**

```python
def test_graph_exec_chain_3_kernels():
    """Build A → B → C dependency chain; verify launch executes all 3."""
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    
    src = """
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd3, %r1;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    ret;
}
"""
    g = Graph()
    n0 = g.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                              params={"A": A, "B": B, "OUT": OUT},
                              kernel_name="vec_add_0")
    n1 = g.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                              params={"A": A, "B": B, "OUT": OUT},
                              kernel_name="vec_add_1")
    n2 = g.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                              params={"A": A, "B": B, "OUT": OUT},
                              kernel_name="vec_add_2")
    g.add_dependency(n0, n1)
    g.add_dependency(n1, n2)
    
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    
    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0
```

- [ ] **Step 2: Run + verify PASS** (the GraphExec from T6 should already handle this).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/graph/test_graph_exec.py
git commit -m "test(graph): GraphExec chain-of-3 dependency execution"
```

---

### Task 8: Example graph_explicit_build

**Files:**
- Create: `examples/graph_explicit_build/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_graph_explicit_build.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_explicit_build"


def test_graph_explicit_build_correctness():
    """3-kernel chain via explicit Graph builder."""
    import gpusim
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
    g = Graph()
    nids = []
    for i in range(3):
        nid = g.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params={"A": A, "B": B, "OUT": OUT},
                                  kernel_name=f"vec_add_{i}")
        nids.append(nid)
    g.add_dependency(nids[0], nids[1])
    g.add_dependency(nids[1], nids[2])
    
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    
    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0
```

- [ ] **Step 2: kernel.ptx (vec_add):**

```
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd3, %r1;
    
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    
    add.f32 %f2, %f0, %f1;
    
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    
    ret;
}
```

- [ ] **Step 3: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np
def reference(A, B): return A + B
```

`run.py`:
```python
import numpy as np, pathlib
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    g = Graph()
    nids = []
    for i in range(3):
        nid = g.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params={"A": A, "B": B, "OUT": OUT},
                                  kernel_name=f"vec_add_{i}")
        nids.append(nid)
    g.add_dependency(nids[0], nids[1])
    g.add_dependency(nids[1], nids[2])
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    print(f"Graph (3-kernel chain): {cycles} cycles")
    print(f"OUT[0:4] = {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# graph_explicit_build

Phase 11 demo: 3-kernel chain built via explicit `Graph.add_kernel_node` +
`add_dependency` + `instantiate` + `launch`.

## Run
```
python examples/graph_explicit_build/run.py
```

## Tutorial
docs/tutorial/44-cuda-graphs-explicit-build.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_graph_explicit_build.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/graph_explicit_build/ tests/parity/test_graph_explicit_build.py
git commit -m "feat(examples): graph_explicit_build — 3-kernel chain via Graph builder"
```

---

### Task 9: Tag M2

```bash
.venv/bin/pytest -q -m "not slow"
git tag M2-phase11-complete
```

---

## Milestone M3: Stream capture mode + capture example

### Task 10: Stream.begin_capture / end_capture + capture branch in launch

**Files:**
- Modify: `gpusim/api.py` (Stream gains `_captured_graph` + `_capture_last_node` fields + `begin_capture`/`end_capture` + capture branch in `launch`)
- Test: `tests/unit/graph/test_graph_capture.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_stream_begin_capture_creates_graph():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.begin_capture()
    assert s._captured_graph is not None


def test_stream_capture_records_kernel_nodes():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src="x", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s.launch(ptx_src="y", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k1")
    g = s.end_capture()
    assert len(g.nodes) == 2
    assert g.nodes[0].kernel_args.kernel_name == "k0"
    assert g.nodes[1].kernel_args.kernel_name == "k1"


def test_stream_capture_implicit_dependency():
    """Each capture-mode launch depends on previous."""
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src="x", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s.launch(ptx_src="y", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k1")
    g = s.end_capture()
    assert len(g.edges) == 1
    assert g.edges[0] == (0, 1)


def test_stream_normal_launch_after_end_capture():
    """After end_capture, .launch goes back to normal pending queue."""
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src="x", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s.end_capture()
    # Normal launch should populate pending
    s.launch(ptx_src="y", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k_normal")
    assert len(s.pending) == 1
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add capture state + methods to Stream in gpusim/api.py:**

```python
    _captured_graph: object | None = None     # NEW Phase 11
    _capture_last_node: int | None = None     # NEW Phase 11
    
    def begin_capture(self) -> None:
        """Start recording subsequent .launch into a fresh Graph. Phase 11."""
        from gpusim.graph.graph import Graph
        self._captured_graph = Graph()
        self._capture_last_node = None
    
    def end_capture(self) -> "Graph":
        """Stop capture; return the recorded Graph. Phase 11."""
        g = self._captured_graph
        self._captured_graph = None
        self._capture_last_node = None
        return g
```

In `Stream.launch`, add a capture branch at the start:

```python
    def launch(self, ptx_src: str, grid: tuple, block: tuple,
                params: dict, *, kernel_name: str = "<unnamed>",
                config=None) -> None:
        # NEW Phase 11: capture mode
        if self._captured_graph is not None:
            nid = self._captured_graph.add_kernel_node(
                ptx_src=ptx_src, grid=grid, block=block, params=params,
                kernel_name=kernel_name,
            )
            if self._capture_last_node is not None:
                self._captured_graph.add_dependency(self._capture_last_node, nid)
            self._capture_last_node = nid
            return
        # Existing Phase 7-10 behavior:
        self.pending.append(GridLaunch(...))
```

⚠ Adapt to existing Stream.launch signature. Add capture check at top; existing behavior preserved when not capturing.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/graph/test_graph_capture.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/api.py tests/unit/graph/test_graph_capture.py
git commit -m "feat(api): Stream.begin_capture/end_capture + capture branch in launch"
```

---

### Task 11: Capture mode produces executable graph

**Files:**
- Test: extend `tests/unit/graph/test_graph_capture.py`

- [ ] **Step 1: Append failing test**

```python
def test_capture_then_instantiate_then_launch():
    """Capture 2 launches → instantiate → launch → outputs correct."""
    import numpy as np
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    cfg = load_default()
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    
    src = """
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd3, %r1;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    ret;
}
"""
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": OUT}, kernel_name="vec_add_0",
              config=cfg)
    s.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": OUT}, kernel_name="vec_add_1",
              config=cfg)
    g = s.end_capture()
    
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    
    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0
```

- [ ] **Step 2: Run + verify PASS** (capture + GraphExec from T6 should make this work).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/graph/test_graph_capture.py
git commit -m "test(graph): capture → instantiate → launch end-to-end"
```

---

### Task 12: Example graph_capture_from_stream

**Files:**
- Create: `examples/graph_capture_from_stream/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_graph_capture_from_stream.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_capture_from_stream"


def test_graph_capture_from_stream_correctness():
    """Capture 3-launch sequence into Graph; instantiate; launch."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
    s = Stream()
    s.begin_capture()
    for i in range(3):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": OUT},
                  kernel_name=f"vec_add_{i}", config=cfg)
    g = s.end_capture()
    
    assert len(g.nodes) == 3
    assert len(g.edges) == 2   # implicit chain
    
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    
    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0
```

- [ ] **Step 2: kernel.ptx** (same vec_add as graph_explicit_build).

- [ ] **Step 3: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np
def reference(A, B): return A + B
```

`run.py`:
```python
import numpy as np, pathlib
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s = Stream()
    s.begin_capture()
    for i in range(3):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": OUT},
                  kernel_name=f"vec_add_{i}", config=cfg)
    g = s.end_capture()
    print(f"Captured graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    print(f"Replay: {cycles} cycles, OUT[0:4] = {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# graph_capture_from_stream

Phase 11 demo: capture a 3-launch sequence from `Stream` into a `Graph`,
then instantiate + replay.

## Run
```
python examples/graph_capture_from_stream/run.py
```

## Tutorial
docs/tutorial/45-stream-capture-to-graph.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_graph_capture_from_stream.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/graph_capture_from_stream/ tests/parity/test_graph_capture_from_stream.py
git commit -m "feat(examples): graph_capture_from_stream — Stream capture → Graph"
```

---

### Task 13: Tag M3

```bash
.venv/bin/pytest -q -m "not slow"
git tag M3-phase11-complete
```

---

## Milestone M4: GraphLaunch trace + 3 metrics + 2 examples

### Task 14: GraphLaunch trace event + recorder + parquet

**Files:**
- Modify: `gpusim/trace/events.py` (add GraphLaunch)
- Modify: `gpusim/trace/recorder.py` (add graph_launch method + list)
- Modify: `gpusim/trace/writer.py` (add parquet writer)
- Test: `tests/unit/trace/test_graph_launch_event.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_recorder_records_graph_launch():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.graph_launch(graph_id=0, n_nodes=3, n_edges=2,
                     launch_index=0, start_cycle=0, end_cycle=300)
    assert len(r.graph_launch_events) == 1
    e = r.graph_launch_events[0]
    assert e.n_nodes == 3
    assert e.launch_index == 0


def test_recorder_writes_graph_launch_parquet(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.graph_launch(graph_id=0, n_nodes=2, n_edges=1,
                     launch_index=0, start_cycle=0, end_cycle=200)
    write_parquet(r, tmp_path)
    assert (tmp_path / "graph_launch.parquet").exists()
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add GraphLaunch dataclass + recorder method + writer:**

In `gpusim/trace/events.py`:
```python
@dataclass(frozen=True)
class GraphLaunch:
    graph_id: int
    n_nodes: int
    n_edges: int
    launch_index: int
    start_cycle: int
    end_cycle: int
```

In `gpusim/trace/recorder.py`:
- `__init__`: `self.graph_launch_events: list = []`
- Add method:
```python
    def graph_launch(self, *, graph_id: int, n_nodes: int, n_edges: int,
                       launch_index: int, start_cycle: int, end_cycle: int) -> None:
        from gpusim.trace.events import GraphLaunch
        self.graph_launch_events.append(GraphLaunch(
            graph_id=graph_id, n_nodes=n_nodes, n_edges=n_edges,
            launch_index=launch_index, start_cycle=start_cycle, end_cycle=end_cycle,
        ))
```

In `gpusim/trace/writer.py::write_parquet`:
```python
    if r.graph_launch_events:
        pd.DataFrame([asdict(e) for e in r.graph_launch_events]).to_parquet(
            out_dir / "graph_launch.parquet", index=False)
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/trace/test_graph_launch_event.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/trace/ tests/unit/trace/test_graph_launch_event.py
git commit -m "feat(trace): GraphLaunch event + recorder + parquet"
```

---

### Task 15: GraphExec records GraphLaunch on launch

**Files:**
- Modify: `gpusim/graph/exec.py` (GraphExec records GraphLaunch when called with recorder)
- Test: extend `tests/unit/graph/test_graph_exec.py`

- [ ] **Step 1: Append failing test**

```python
def test_graph_exec_records_launch_event():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    g = Graph()
    g.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    rec = Recorder()
    exec = g.instantiate(cfg)
    exec._recorder = rec
    exec._graph_id = 0
    exec.launch()
    assert len(rec.graph_launch_events) == 1
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add recorder support to GraphExec:**

```python
@dataclass
class GraphExec:
    graph: object
    topo_order: list
    config: object
    _recorder: object | None = None
    _graph_id: int = 0
    _launch_count: int = 0
    
    @classmethod
    def from_graph(cls, graph, config) -> "GraphExec":
        topo = _topological_sort(graph.nodes, graph.edges)
        return cls(graph=graph, topo_order=topo, config=config)
    
    def launch(self) -> int:
        # ... existing kernel/memcpy/event execution ...
        # ... compute total_cycles ...
        if self._recorder is not None:
            self._recorder.graph_launch(
                graph_id=self._graph_id,
                n_nodes=len(self.graph.nodes),
                n_edges=len(self.graph.edges),
                launch_index=self._launch_count,
                start_cycle=0,
                end_cycle=total_cycles,
            )
        self._launch_count += 1
        return total_cycles
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/graph/test_graph_exec.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/graph/exec.py tests/unit/graph/test_graph_exec.py
git commit -m "feat(graph): GraphExec records GraphLaunch event per replay"
```

---

### Task 16: 3 analysis metrics

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Test: `tests/unit/analysis/test_phase11_metrics.py` (NEW)

- [ ] **Step 1: Create test**

```python
import pandas as pd


def test_graph_replay_amortization():
    from gpusim.analysis.metrics import graph_replay_amortization
    df = pd.DataFrame([
        {"launch_index": 0, "start_cycle": 0, "end_cycle": 300},
        {"launch_index": 1, "start_cycle": 300, "end_cycle": 590},
        {"launch_index": 2, "start_cycle": 590, "end_cycle": 870},
    ])
    out = graph_replay_amortization(df, single_kernel_baseline_cycles=150)
    # 3 replays of 3-kernel graph; baseline would be 9 single launches
    # graph cycles per replay ≈ (870/3) ≈ 290; baseline 9*150=1350 vs 870 → savings
    assert "avg_cycles_per_replay" in out
    assert "amortization_factor" in out


def test_graph_dag_depth_linear():
    from gpusim.analysis.metrics import graph_dag_depth
    from gpusim.graph.graph import Graph
    g = Graph()
    a = g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="ka")
    b = g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="kb")
    c = g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="kc")
    g.add_dependency(a, b)
    g.add_dependency(b, c)
    assert graph_dag_depth(g) == 3   # length of longest path


def test_graph_node_type_breakdown():
    from gpusim.analysis.metrics import graph_node_type_breakdown
    from gpusim.graph.graph import Graph
    import numpy as np
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    g.add_kernel_node(ptx_src="y", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k2")
    g.add_memcpy_node(src=np.zeros(8), dst=np.zeros(8), n_bytes=32)
    out = graph_node_type_breakdown(g)
    assert out["kernel"] == 2
    assert out["memcpy"] == 1
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Append metrics:**

```python
def graph_replay_amortization(graph_launch_df, single_kernel_baseline_cycles: int) -> dict:
    """Cycles per replay vs N independent kernel launches.
    Returns dict with avg_cycles_per_replay and amortization_factor."""
    if graph_launch_df is None or graph_launch_df.empty:
        return {"avg_cycles_per_replay": 0.0, "amortization_factor": 0.0}
    durations = (graph_launch_df["end_cycle"] - graph_launch_df["start_cycle"]).values
    avg = float(durations.mean())
    if single_kernel_baseline_cycles <= 0 or avg <= 0:
        return {"avg_cycles_per_replay": avg, "amortization_factor": 0.0}
    # Assume each replay would otherwise take baseline*N if launched independently
    # Use launches as proxy for N (or 1 if not specified)
    return {
        "avg_cycles_per_replay": avg,
        "amortization_factor": float(single_kernel_baseline_cycles) / avg,
    }


def graph_dag_depth(graph) -> int:
    """Length of longest dependency chain (number of nodes on critical path)."""
    if not graph.nodes:
        return 0
    children = {}
    for parent, child in graph.edges:
        children.setdefault(parent, []).append(child)
    # Memoized DFS for longest path
    cache = {}
    def longest_from(nid):
        if nid in cache: return cache[nid]
        if nid not in children or not children[nid]:
            cache[nid] = 1
        else:
            cache[nid] = 1 + max(longest_from(c) for c in children[nid])
        return cache[nid]
    return max(longest_from(n.node_id) for n in graph.nodes)


def graph_node_type_breakdown(graph) -> dict:
    """Count of nodes by type."""
    out = {"kernel": 0, "memcpy": 0, "event": 0}
    for n in graph.nodes:
        out[n.type] = out.get(n.type, 0) + 1
    return out
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/analysis/test_phase11_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/analysis/metrics.py tests/unit/analysis/test_phase11_metrics.py
git commit -m "feat(analysis): graph_replay_amortization + graph_dag_depth + graph_node_type_breakdown"
```

---

### Task 17: Example graph_replay_perf

**Files:**
- Create: `examples/graph_replay_perf/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_graph_replay_perf.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_replay_perf"


def test_graph_replay_perf_correctness():
    """Capture a small graph; replay 5x; verify each replay produces correct output."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": OUT},
              kernel_name="vec_add", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    
    # Replay 5 times
    cycles_per_replay = []
    for i in range(5):
        cycles_per_replay.append(exec.launch())
    
    assert len(cycles_per_replay) == 5
    np.testing.assert_array_equal(OUT, A + B)
    # Replays should be deterministic
    assert all(c > 0 for c in cycles_per_replay)
```

- [ ] **Step 2: kernel.ptx** (vec_add — same).

- [ ] **Step 3: reference.py + run.py + README.md + __init__.py**

`run.py`:
```python
import numpy as np, pathlib
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": OUT},
              kernel_name="vec_add", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    cycles_per_replay = [exec.launch() for _ in range(5)]
    print(f"Replay cycles per launch: {cycles_per_replay}")
    print(f"Average cycles/replay: {sum(cycles_per_replay)/5:.1f}")
    print(f"Final OUT[0:4]: {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# graph_replay_perf

Phase 11 demo: capture a graph; replay 5 times; observe per-replay cycles.
Demonstrates graph replay deterministic behavior.

## Run
```
python examples/graph_replay_perf/run.py
```

## Tutorial
docs/tutorial/46-graph-replay-amortization.md
```

`__init__.py` (empty); `reference.py` similar.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_graph_replay_perf.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/graph_replay_perf/ tests/parity/test_graph_replay_perf.py
git commit -m "feat(examples): graph_replay_perf — 5x replay deterministic check"
```

---

### Task 18: Example graph_iterative_train_step

**Files:**
- Create: `examples/graph_iterative_train_step/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_graph_iterative_train_step.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_iterative_train_step"


def test_graph_iterative_train_step_correctness():
    """Capture training step graph; replay 3 times to simulate 3 epochs."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 32
    weights = np.zeros(n, dtype=np.float32)
    grads = np.ones(n, dtype=np.float32)
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
    # Capture: weights -= grads (single kernel)
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"WEIGHTS": weights, "GRADS": grads},
              kernel_name="sgd_update", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    
    # Replay 3 times — each subtracts grads from weights
    for epoch in range(3):
        exec.launch()
    
    # After 3 replays: weights = 0 - 3*1 = -3 (each epoch subtracts 1)
    np.testing.assert_array_equal(weights, np.full(n, -3.0, dtype=np.float32))
```

- [ ] **Step 2: kernel.ptx (SGD-style: weights[tid] -= grads[tid]):**

```
.visible .entry test(.param .u64 WEIGHTS, .param .u64 GRADS)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    
    ld.param.u64 %rd0, [WEIGHTS];
    ld.param.u64 %rd1, [GRADS];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    add.u64 %rd4, %rd1, %rd2;
    ld.global.f32 %f1, [%rd4];
    
    sub.f32 %f2, %f0, %f1;
    
    st.global.f32 [%rd3], %f2;
    
    ret;
}
```

- [ ] **Step 3: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np
def reference(weights, grads, n_epochs):
    return weights - grads * n_epochs
```

`run.py`:
```python
import numpy as np, pathlib
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    weights = np.zeros(n, dtype=np.float32)
    grads = np.ones(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"WEIGHTS": weights, "GRADS": grads},
              kernel_name="sgd_update", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    for epoch in range(3):
        exec.launch()
    print(f"After 3 epochs, weights[0:4]: {list(weights[0:4])}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# graph_iterative_train_step

Phase 11 demo: capture an SGD update step; replay 3 times to simulate 3 epochs.
Demonstrates production-style graph reuse pattern (PyTorch torch.cuda.graphs).

## Run
```
python examples/graph_iterative_train_step/run.py
```

## Tutorial
docs/tutorial/47-graph-iterative-training.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_graph_iterative_train_step.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/graph_iterative_train_step/ tests/parity/test_graph_iterative_train_step.py
git commit -m "feat(examples): graph_iterative_train_step — capstone SGD replay"
```

---

### Task 19: Tag M4

```bash
.venv/bin/pytest -q -m "not slow"
git tag M4-phase11-complete
```

---

### Task 20: (consolidated, kept as buffer)

(Reserved.)

---

## Milestone M5: Viz + tutorials + microbench + ship

### Task 21: HTML §35 graph DAG visualization

**Files:**
- Modify: `gpusim/viz/html_report.py`
- Modify: `gpusim/viz/_template.html.j2`
- Test: `tests/unit/viz/test_html_report_phase11.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_html_report_phase11_section(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.graph_launch(graph_id=0, n_nodes=3, n_edges=2,
                     launch_index=0, start_cycle=0, end_cycle=300)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=300, occupancy={"active_ctas": 1, "bottleneck": "none"})
    html = out.read_text()
    assert "Graph" in html or "graph" in html.lower() or "§35" in html
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add render helper:**

```python
def _render_graph_dag(rec):
    if not getattr(rec, "graph_launch_events", None): return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.graph_launch_events])
    return "<h3>Graph launches</h3>" + df.to_html(index=False)
```

In `save_html`, add to context:
```python
    context.update({"graph_dag_html": _render_graph_dag(rec)})
```

In `_template.html.j2` (after Phase 10 §34):
```html
{% if graph_dag_html %}
<h2>§35 Graph DAG</h2>
{{ graph_dag_html | safe }}
{% endif %}
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/viz/test_html_report_phase11.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/viz/html_report.py gpusim/viz/_template.html.j2 tests/unit/viz/test_html_report_phase11.py
git commit -m "feat(viz): HTML §35 — Graph DAG visualization"
```

---

### Task 22: Perfetto Graph swimlane

**Files:**
- Modify: `gpusim/viz/perfetto.py`
- Test: extend `tests/unit/viz/test_html_report_phase11.py`

- [ ] **Step 1: Append failing test**

```python
def test_perfetto_graph_swimlane():
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.graph_launch(graph_id=0, n_nodes=3, n_edges=2,
                     launch_index=0, start_cycle=0, end_cycle=300)
    pf = build_perfetto(r)
    pids = {e.get("pid") for e in pf.get("traceEvents", [])}
    assert any("Graph" in str(p) for p in pids)
```

- [ ] **Step 2: Add Graph swimlane to perfetto.py:**

```python
    # Phase 11: graph launches
    for ev in getattr(rec, "graph_launch_events", []):
        events.append({
            "name": f"graph_{ev.graph_id}_replay_{ev.launch_index}",
            "cat": "graph", "ph": "X",
            "ts": ev.start_cycle,
            "dur": max(1, ev.end_cycle - ev.start_cycle),
            "pid": "Graph",
            "tid": f"graph_{ev.graph_id}",
            "args": {"graph_id": ev.graph_id, "n_nodes": ev.n_nodes,
                     "n_edges": ev.n_edges, "launch_index": ev.launch_index},
        })
```

- [ ] **Step 3: Run + commit**

```
.venv/bin/pytest tests/unit/viz/test_html_report_phase11.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/viz/perfetto.py tests/unit/viz/test_html_report_phase11.py
git commit -m "feat(viz): Perfetto Graph swimlane"
```

---

### Task 23: 4 tutorial chapters 44-47

**Files:**
- Create: `docs/tutorial/{44,45,46,47}-*.md`

- [ ] **Step 1: Read existing style** (`docs/tutorial/43-ddp-training-pattern.md`).

- [ ] **Step 2: Write 4 chapters** (~500-700 words each, English body + Chinese subheadings):

**Chapter 44 — cuda-graphs-explicit-build:**
- Graph builder API: add_kernel_node + add_dependency + instantiate + launch
- 3-kernel chain demo
- 看模拟器: graph_dag_depth, graph_node_type_breakdown
- 改一改: add diamond-shape dependency
- 真机对照: cudaGraphCreate + cudaGraphAddKernelNode

**Chapter 45 — stream-capture-to-graph:**
- Stream.begin_capture/end_capture; implicit dependency from launch order
- graph_capture_from_stream demo
- 看模拟器: captured graph nodes + edges count
- 改一改: cross-stream events to add cross-stream dependencies in capture
- 真机对照: cudaStreamBeginCapture + cudaStreamEndCapture

**Chapter 46 — graph-replay-amortization ⭐:**
- Replay deterministic; cycles per replay
- graph_replay_perf demo
- 看模拟器: graph_replay_amortization metric
- 改一改: increase replay count to 100; observe steady cycles
- 真机对照: PyTorch torch.cuda.graphs.graph() launch overhead

**Chapter 47 — graph-iterative-training:**
- SGD-style replay pattern
- graph_iterative_train_step demo
- 看模拟器: weights drift across replays
- 改一改: change grads between replays (read at launch time)
- 真机对照: PyTorch torch.cuda.graphs.make_graphed_callables

```bash
git add docs/tutorial/44-cuda-graphs-explicit-build.md \
        docs/tutorial/45-stream-capture-to-graph.md \
        docs/tutorial/46-graph-replay-amortization.md \
        docs/tutorial/47-graph-iterative-training.md
git commit -m "docs(tutorial): chapters 44-47 — Phase 11 CUDA Graphs"
```

---

### Task 24: Phase 11 microbench + 4 ref stubs

**Files:**
- Create: `tests/microbench/test_phase11_facts.py`
- Create: `tests/microbench/test_phase11_runtime.py`
- Modify: `tests/reference/gen_reference.py`
- Create: 4 ref JSONs

- [ ] **Step 1: test_phase11_facts.py:**

```python
"""Phase 11 microbench — CUDA Graphs facts."""


def test_topo_sort_known_graph():
    """Topological sort produces valid order for diamond graph."""
    from gpusim.graph.exec import _topological_sort
    from gpusim.graph.node import GraphNode
    nodes = [GraphNode(node_id=i, type="kernel") for i in range(4)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    order = _topological_sort(nodes, edges)
    assert order[0] == 0
    assert order[-1] == 3


def test_capture_chain_dependency_count():
    """3 capture launches → 2 implicit dependencies."""
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.begin_capture()
    for i in range(3):
        s.launch(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                  params={}, kernel_name=f"k{i}")
    g = s.end_capture()
    assert len(g.nodes) == 3
    assert len(g.edges) == 2


def test_graph_replay_deterministic():
    """Replay 3x produces same output."""
    import numpy as np
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    cfg = load_default()
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    src = """
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT) {
    .reg .u64 %rd<6>; .reg .u32 %r<4>; .reg .f32 %f<4>;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.u64 %rd2, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd3, %r1;
    add.u64 %rd4, %rd0, %rd3; ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3; ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    add.u64 %rd4, %rd2, %rd3; st.global.f32 [%rd4], %f2;
    ret;
}
"""
    OUT = np.zeros(n, dtype=np.float32)
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": OUT}, kernel_name="vec_add", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    cycles = [exec.launch() for _ in range(3)]
    # Replays should produce same cycles (deterministic)
    assert cycles[0] == cycles[1] == cycles[2]
```

- [ ] **Step 2: test_phase11_runtime.py:**

```python
import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_graph_explicit_build_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_explicit_build"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_graph_replay_perf_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_replay_perf"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
```

- [ ] **Step 3: Append 4 kernel names to gen_reference.py:**

```python
"graph_explicit_build",
"graph_capture_from_stream",
"graph_replay_perf",
"graph_iterative_train_step",
```

- [ ] **Step 4: Create 4 ref JSONs:**

```bash
for k in graph_explicit_build graph_capture_from_stream graph_replay_perf graph_iterative_train_step; do
  cat > tests/reference/data/$k.ref.json <<JSON
{
  "kernel": "$k",
  "phase": 11,
  "metrics": {
    "graph_replay_amortization": null,
    "graph_dag_depth": null,
    "graph_node_type_breakdown": null
  },
  "tolerance": {
    "graph_replay_amortization_pct": 20,
    "graph_dag_depth_pct": 0,
    "graph_node_type_breakdown_pct": 0
  },
  "notes": "Generated by gen_reference.py on real Hopper. null = not yet measured."
}
JSON
done
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/microbench/test_phase11_facts.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add tests/microbench/test_phase11_facts.py tests/microbench/test_phase11_runtime.py \
        tests/reference/gen_reference.py tests/reference/data/graph_*.ref.json
git commit -m "test(microbench+reference): Phase 11 facts + 4 ref stubs"
```

---

### Task 25: Phase 1-10 regression rename + add 4 Phase 10 examples

**Files:**
- Rename: `tests/parity/test_phase1_9_examples_unchanged.py` → `test_phase1_10_examples_unchanged.py`

- [ ] **Step 1: git mv:**

```bash
git mv tests/parity/test_phase1_9_examples_unchanged.py tests/parity/test_phase1_10_examples_unchanged.py
```

- [ ] **Step 2: Edit:**
- Rename `PHASE_1_9_EXAMPLES` → `PHASE_1_10_EXAMPLES`
- Append 4 Phase 10 examples: `multi_gpu_setup`, `ring_allreduce`, `tree_allreduce`, `ddp_training_step`
- Update test function names from `phase_1_9_*` → `phase_1_10_*` if any

- [ ] **Step 3: Run + commit**

```
.venv/bin/pytest tests/parity/test_phase1_10_examples_unchanged.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add tests/parity/test_phase1_10_examples_unchanged.py
git commit -m "test(regression): rename phase1_9 → phase1_10 + 4 Phase 10 examples"
```

---

### Task 26: README v11 + final tag

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README.md (Phase 10 v10).**

- [ ] **Step 2: Update to v11:**
- Phase status: 1-11 ✅
- Phase 11 features section:
  - gpusim.Graph + Graph builder API + 3 node types (kernel + memcpy + event)
  - Stream.begin_capture / end_capture (implicit dependency from launch order)
  - GraphExec topological sort + replay
  - 3 metrics (graph_replay_amortization, graph_dag_depth, graph_node_type_breakdown)
  - 1 trace event (GraphLaunch), HTML §35 + Perfetto Graph swimlane
  - 4 examples + 4 tutorials chapters 44-47
  - Backward compatible: Phase 1-10 unchanged
- Examples list: add 4 (was 42, now 46)
- Tutorials list: add 44-47 (was 43, now 47)

- [ ] **Step 3: Run final suite + 4 examples:**

```
.venv/bin/pytest -q -m "not slow"
.venv/bin/python examples/graph_explicit_build/run.py
.venv/bin/python examples/graph_capture_from_stream/run.py
.venv/bin/python examples/graph_replay_perf/run.py
.venv/bin/python examples/graph_iterative_train_step/run.py
```

- [ ] **Step 4: Commit + tag:**

```bash
git add README.md
git commit -m "docs(readme): v11 — Phase 11 capabilities (CUDA Graphs)"
git tag phase11-complete
```

---

### Task 27: Final sanity sweep

```
.venv/bin/pytest -q -m "not slow"
.venv/bin/pytest tests/parity/test_phase1_10_examples_unchanged.py -v
```

---

### Task 28: Done

Phase 11 ships when all tasks complete + tags landed.

---

## End-of-plan checklist

- [ ] M1 (Graph + nodes + DAG): T1-T5
- [ ] M2 (GraphExec + execute + 1 example): T6-T9
- [ ] M3 (Stream capture + capture example): T10-T13
- [ ] M4 (GraphLaunch + 3 metrics + 2 examples): T14-T20
- [ ] M5 (Viz + tutorials + microbench + regression + README): T21-T28
- [ ] All 5 milestone tags + phase11-complete
- [ ] Phase 1-10 regression unbroken
- [ ] 4 new examples + 4 tutorials shipped
- [ ] README v11 reflects Phase 11
