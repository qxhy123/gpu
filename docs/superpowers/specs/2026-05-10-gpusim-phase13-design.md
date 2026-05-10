# gpusim Phase 13 — Graphs Completion (child graphs + update API + memset node)

> **Status:** Brainstormed 2026-05-10. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- Add **child graph nodes** (`Graph.add_child_graph_node(graph)`) — nested DAG execution.
- Add **graph update API** (`GraphExec.update_kernel_node_params(node_id, **kwargs)`) — modify executable graph between replays without re-instantiating.
- Add **memset node** (`Graph.add_memset_node(buf, value, n_bytes)`) — fill memory with constant value, 50-cycle overhead.
- 3 examples + 3 tutorial chapters (51-53).
- 2 new metrics: `graph_child_depth` (max child-nesting depth) + `graph_update_count` (number of update_node_params calls per GraphExec).
- Reuse Phase 11 HTML §35 + Perfetto Graph swimlane.
- 100% backward compatible: Phase 1-12 unchanged.

### Non-goals (deferred to Phase 14+)
- Conditional graph nodes (if/while — CUDA 12.4+).
- Host callback nodes (`cudaGraphAddHostNode`).
- Graph cloning (`cudaGraphClone`).
- Multi-GPU graphs.
- Cross-graph dependency between separate Graph instances.

---

## 2. Architecture

```
gpusim.graph (Phase 11) — NEW node types + APIs:
├── GraphNode.type: "kernel" | "memcpy" | "event" | "memset" | "child_graph"
├── MemsetNodeArgs (NEW)
├── ChildGraphNodeArgs (NEW)
├── Graph.add_memset_node(buf, value, n_bytes) → node_id
├── Graph.add_child_graph_node(graph) → node_id
├── GraphExec.update_kernel_node_params(node_id, **kwargs) → None
└── GraphExec.launch() — handles new node types (memset functional fill; child_graph recursive execution)
```

### Key invariants
- Child graph executes at parent node position; child nodes are NOT flattened into parent's topo order — child has its own GraphExec.
- Update API requires `node_id` to refer to a kernel node; raises `ValueError` for non-kernel.
- Update API modifies the GraphExec's internal node copy; original Graph builder is unchanged.
- Memset is functional (numpy fill) + 50-cycle overhead; no kernel execution.
- Replay count tracking: each `GraphExec.launch()` increments `_launch_count`; each `update_kernel_node_params` increments `_update_count`.

---

## 3. Data model

### 3.1 New node arg dataclasses (`gpusim/graph/node.py`)

```python
@dataclass
class MemsetNodeArgs:
    buf: object           # ndarray
    value: int            # fill value (treated as the dtype-cast scalar)
    n_bytes: int


@dataclass
class ChildGraphNodeArgs:
    graph: object         # nested Graph instance
```

### 3.2 GraphNode extension

```python
@dataclass
class GraphNode:
    node_id: int
    type: str             # "kernel" | "memcpy" | "event" | "memset" | "child_graph"
    kernel_args: KernelNodeArgs | None = None
    memcpy_args: MemcpyNodeArgs | None = None
    event_args: EventNodeArgs | None = None
    memset_args: MemsetNodeArgs | None = None        # NEW Phase 13
    child_graph_args: ChildGraphNodeArgs | None = None    # NEW Phase 13
```

### 3.3 Graph builder methods

```python
class Graph:
    def add_memset_node(self, *, buf, value: int, n_bytes: int) -> int:
        nid = self._next_id; self._next_id += 1
        args = MemsetNodeArgs(buf=buf, value=value, n_bytes=n_bytes)
        self.nodes.append(GraphNode(node_id=nid, type="memset", memset_args=args))
        return nid
    
    def add_child_graph_node(self, *, graph: "Graph") -> int:
        nid = self._next_id; self._next_id += 1
        args = ChildGraphNodeArgs(graph=graph)
        self.nodes.append(GraphNode(node_id=nid, type="child_graph", child_graph_args=args))
        return nid
```

### 3.4 GraphExec update API

```python
class GraphExec:
    _update_count: int = 0
    
    def update_kernel_node_params(self, node_id: int, **kwargs) -> None:
        """Modify a kernel node's params in place. Phase 13.
        
        Allowed kwargs: ptx_src, grid, block, params, kernel_name.
        Raises ValueError if node_id not found or node is not a kernel.
        """
        node = next((n for n in self.graph.nodes if n.node_id == node_id), None)
        if node is None:
            raise ValueError(f"node_id {node_id} not found")
        if node.type != "kernel":
            raise ValueError(f"node_id {node_id} is type {node.type!r}, not kernel")
        for k, v in kwargs.items():
            if k not in ("ptx_src", "grid", "block", "params", "kernel_name"):
                raise ValueError(f"unknown update field: {k}")
            setattr(node.kernel_args, k, v)
        self._update_count += 1
```

### 3.5 GraphExec.launch — handle new node types

```python
def launch(self) -> int:
    # ... existing kernel/memcpy/event ...
    elif node.type == "memset":
        a = node.memset_args
        # Functional: fill buf with value
        a.buf[:] = a.value
        total_cycles += 50
    elif node.type == "child_graph":
        a = node.child_graph_args
        # Execute child graph — recursively instantiate + launch
        child_exec = a.graph.instantiate(self.config)
        total_cycles += child_exec.launch()
```

---

## 4. Trace + Analysis

### 4.1 Trace
Reuse Phase 11 `GraphLaunch` event for top-level launches. Child graph launches do NOT emit separate GraphLaunch events (their cycles roll up into parent's count); this avoids double-counting.

### 4.2 2 new metrics

```python
def graph_child_depth(graph) -> int:
    """Maximum nesting depth of child graphs in this graph.
    Linear graph (no children) = 0; one child level = 1; etc."""

def graph_update_count(graph_exec) -> int:
    """Number of update_kernel_node_params calls performed on this GraphExec."""
```

### 4.3 Result API

```python
class GraphExecResult:
    def graph_child_depth(self) -> int: ...
    def graph_update_count(self) -> int: ...
```

---

## 5. Viz

Reuse Phase 11 HTML §35 + Perfetto Graph swimlane. Phase 13 adds no new viz sections — child graph nodes appear as nested entries in §35 with indentation.

---

## 6. Examples (3)

### 6.1 `graph_with_child/`
- Build outer graph with 1 child graph node (containing 2 kernel nodes).
- **Verifies:** child execution produces correct outputs; cycles roll up into parent's GraphLaunch count.

### 6.2 `graph_update_replay/`
- Capture single-kernel graph; replay 3 times. Between replays, call `update_kernel_node_params` to swap input buffers.
- **Verifies:** updated params take effect on next replay; `update_count == 2` after 2 update calls.

### 6.3 `graph_memset_zero/`
- Build graph with memset node (zero a buffer) + kernel node (write to buffer) + memset node (zero again at end).
- **Verifies:** buffer transitions correctly across the 3 nodes.

---

## 7. Tutorials

`docs/tutorial/` chapters 51-53:
- **51-graph-child-nested-dag.md** — example 1
- **52-graph-update-api-replay.md** — example 2
- **53-graph-memset-node.md** — example 3

---

## 8. Testing strategy

### Unit tests (~10 new)
- `tests/unit/graph/test_memset_node.py` — Graph.add_memset_node + GraphExec executes memset
- `tests/unit/graph/test_child_graph_node.py` — Graph.add_child_graph_node + nested execution
- `tests/unit/graph/test_graph_update.py` — update_kernel_node_params + validation errors
- `tests/unit/analysis/test_phase13_metrics.py` — 2 new metrics

### Parity tests (~3)

### Microbench
- `test_phase13_facts.py` (fast):
  - Memset node fills buffer correctly; cycles == 50
  - Child graph nesting computes correct depth
  - Update count matches number of update calls
- `test_phase13_runtime.py` (slow): 3 examples each under 60s

### Regression
- Rename `tests/parity/test_phase1_11_examples_unchanged.py` → `test_phase1_12_examples_unchanged.py`
- Add 3 Phase 12 examples to the regression list

### Test count target
661 (Phase 12 baseline) → ~680 (+19).

---

## 9. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1** Memset node + memset_zero example | MemsetNodeArgs + Graph.add_memset_node + GraphExec memset path + graph_memset_zero | `M1-phase13-complete` |
| **M2** Child graph node + with_child example | ChildGraphNodeArgs + add_child_graph_node + GraphExec child path + graph_with_child | `M2-phase13-complete` |
| **M3** Update API + replay example | GraphExec.update_kernel_node_params + validation + graph_update_replay | `M3-phase13-complete` |
| **M4** 2 metrics + analysis | graph_child_depth + graph_update_count | `M4-phase13-complete` |
| **M5** Tutorials + microbench + regression rename + README v13 + ship | 3 chapters + microbench + Phase 1-12 regression rename + README | `phase13-complete` |

Estimated 16 tasks total.

---

## 10. File list

### New files
```
examples/graph_memset_zero/        # 5 files (M1)
examples/graph_with_child/         # 5 files (M2)
examples/graph_update_replay/      # 5 files (M3)
docs/tutorial/51-graph-child-nested-dag.md
docs/tutorial/52-graph-update-api-replay.md
docs/tutorial/53-graph-memset-node.md
tests/unit/graph/test_memset_node.py
tests/unit/graph/test_child_graph_node.py
tests/unit/graph/test_graph_update.py
tests/unit/analysis/test_phase13_metrics.py
tests/parity/test_graph_memset_zero.py
tests/parity/test_graph_with_child.py
tests/parity/test_graph_update_replay.py
tests/microbench/test_phase13_facts.py
tests/microbench/test_phase13_runtime.py
tests/reference/data/{3 example names}.ref.json
```

### Modified files
```
gpusim/graph/node.py             # +MemsetNodeArgs +ChildGraphNodeArgs +GraphNode fields
gpusim/graph/graph.py            # +add_memset_node +add_child_graph_node
gpusim/graph/exec.py             # +memset/child_graph branches in launch + update_kernel_node_params
gpusim/graph/__init__.py         # export new node types
gpusim/analysis/metrics.py       # +2 metrics
tests/parity/test_phase1_11_examples_unchanged.py → test_phase1_12_examples_unchanged.py
tests/reference/gen_reference.py # +3 kernel names
README.md                        # v13 — Phase 13 capabilities
```

---

## 11. Backward compatibility

- All Phase 1-12 examples + tests pass unchanged.
- New node types (memset, child_graph) are additive; existing kernel/memcpy/event paths unchanged.
- Update API is opt-in.

---

## 12. Acceptance criteria

Phase 13 ships when:

- [ ] All 5 milestone tags present (`M1-phase13-complete` ... `M4-phase13-complete`, `phase13-complete`)
- [ ] All 3 examples run cleanly
- [ ] All 3 parity tests pass
- [ ] Microbench: memset cycles = 50, child depth correct, update count tracks calls
- [ ] Phase 1-12 regression test (renamed) passes
- [ ] Test count: 661 → ~680 (+19)
- [ ] README v13 documents Phase 13 capabilities
