# gpusim Phase 13 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Implement Phase 13 per `docs/superpowers/specs/2026-05-10-gpusim-phase13-design.md` — Graphs completion (child graphs + update API + memset node).

**Architecture:** Extend `GraphNode` with 2 new node types (memset, child_graph). New `MemsetNodeArgs` + `ChildGraphNodeArgs` dataclasses. `GraphExec.update_kernel_node_params(node_id, **kwargs)` modifies kernel args in place. Child graph executes via recursive `instantiate + launch`.

**Tech Stack:** Python 3.11+. No new deps.

**Execution note:** Plan has 5 milestones (M1–M5) with 16 tasks. Tags: `M{1..5}-phase13-complete`.

---

## Phase 1+2+...+12 prerequisites

```bash
.venv/bin/pytest -q -m "not slow"
```
Expected: ~661 passed (Phase 12 baseline).

---

## File structure

```
gpusim/graph/
├── node.py        MODIFY: + MemsetNodeArgs + ChildGraphNodeArgs + GraphNode fields
├── graph.py       MODIFY: + add_memset_node + add_child_graph_node
├── exec.py        MODIFY: + memset/child_graph branches + update_kernel_node_params
└── __init__.py    MODIFY: export new types

gpusim/analysis/metrics.py    MODIFY (M4): + 2 metrics

examples/
├── graph_memset_zero/         NEW (M1)
├── graph_with_child/          NEW (M2)
└── graph_update_replay/       NEW (M3)

tests/unit/graph/
├── test_memset_node.py        NEW (M1)
├── test_child_graph_node.py   NEW (M2)
└── test_graph_update.py       NEW (M3)

tests/unit/analysis/test_phase13_metrics.py    NEW (M4)
tests/parity/test_phase1_12_examples_unchanged.py    RENAME (M5)
tests/microbench/test_phase13_facts.py    NEW (M5)
tests/microbench/test_phase13_runtime.py  NEW (M5, slow)
tests/reference/data/{3 names}.ref.json   NEW (M5)
docs/tutorial/{51,52,53}-*.md             NEW (M5)
README.md                                  MODIFY (M5): v13
```

---

## Milestones

| Milestone | Tasks | Tag |
|---|---|---|
| **M1** Memset node + memset_zero example | T1–T3 | `M1-phase13-complete` |
| **M2** Child graph + with_child example | T4–T6 | `M2-phase13-complete` |
| **M3** Update API + update_replay example | T7–T9 | `M3-phase13-complete` |
| **M4** 2 metrics | T10–T11 | `M4-phase13-complete` |
| **M5** Tutorials + microbench + regression rename + README v13 + ship | T12–T16 | `phase13-complete` |

---

## Milestone M1: Memset node + memset_zero example

### Task 1: MemsetNodeArgs + Graph.add_memset_node + GraphExec memset path

**Files:**
- Modify: `gpusim/graph/node.py` (add MemsetNodeArgs + GraphNode.memset_args field)
- Modify: `gpusim/graph/graph.py` (add_memset_node)
- Modify: `gpusim/graph/exec.py` (memset branch in launch)
- Test: `tests/unit/graph/test_memset_node.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_memset_node_args():
    import numpy as np
    from gpusim.graph.node import MemsetNodeArgs
    buf = np.zeros(8, dtype=np.uint8)
    a = MemsetNodeArgs(buf=buf, value=0xff, n_bytes=8)
    assert a.value == 0xff


def test_graph_add_memset_node():
    import numpy as np
    from gpusim.graph.graph import Graph
    g = Graph()
    buf = np.zeros(8, dtype=np.uint8)
    nid = g.add_memset_node(buf=buf, value=42, n_bytes=8)
    assert nid == 0
    assert g.nodes[0].type == "memset"
    assert g.nodes[0].memset_args.value == 42


def test_graph_exec_memset_fills_buffer():
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    buf = np.full(8, 1, dtype=np.uint8)
    g = Graph()
    g.add_memset_node(buf=buf, value=0, n_bytes=8)
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    np.testing.assert_array_equal(buf, np.zeros(8, dtype=np.uint8))
    assert cycles == 50    # fixed memset overhead
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add MemsetNodeArgs to gpusim/graph/node.py:**

```python
@dataclass
class MemsetNodeArgs:
    buf: object
    value: int
    n_bytes: int
```

Add to `GraphNode`:
```python
    memset_args: MemsetNodeArgs | None = None    # NEW Phase 13
```

- [ ] **Step 4: Add to Graph (gpusim/graph/graph.py):**

```python
    def add_memset_node(self, *, buf, value: int, n_bytes: int) -> int:
        from gpusim.graph.node import MemsetNodeArgs
        nid = self._next_id; self._next_id += 1
        args = MemsetNodeArgs(buf=buf, value=value, n_bytes=n_bytes)
        self.nodes.append(GraphNode(node_id=nid, type="memset", memset_args=args))
        return nid
```

- [ ] **Step 5: Add memset branch to GraphExec.launch (gpusim/graph/exec.py):**

In the existing for-node-id loop, add after event branch:

```python
            elif node.type == "memset":
                a = node.memset_args
                a.buf[:] = a.value
                total_cycles += 50
```

- [ ] **Step 6: Update gpusim/graph/__init__.py to export MemsetNodeArgs.**

- [ ] **Step 7: Run + commit**

```
.venv/bin/pytest tests/unit/graph/test_memset_node.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/graph/ tests/unit/graph/test_memset_node.py
git commit -m "feat(graph): MemsetNodeArgs + add_memset_node + GraphExec memset branch"
```

---

### Task 2: Example graph_memset_zero

**Files:**
- Create: `examples/graph_memset_zero/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_graph_memset_zero.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_memset_zero"


def test_graph_memset_zero_correctness():
    """Memset-zero → kernel write → memset-zero. Final buffer = zeros."""
    import gpusim
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    
    n = 32
    buf = np.full(n * 4, 99, dtype=np.uint8)   # n*4 bytes for n float32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)   # we'll repurpose buf for OUT
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
    g = Graph()
    # Pre-zero buffer
    n0 = g.add_memset_node(buf=buf, value=0, n_bytes=n*4)
    # Compute kernel writes to OUT (separate buffer)
    n1 = g.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                              params={"A": A, "B": B, "OUT": OUT},
                              kernel_name="vec_add")
    g.add_dependency(n0, n1)
    # Post-zero buf (different from OUT)
    n2 = g.add_memset_node(buf=buf, value=0, n_bytes=n*4)
    g.add_dependency(n1, n2)
    
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    
    np.testing.assert_array_equal(buf, np.zeros(n*4, dtype=np.uint8))
    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 100   # 2 memsets (50 each) + kernel
```

- [ ] **Step 2: kernel.ptx (vec_add — same as graph_explicit_build):**

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
    buf = np.full(n*4, 99, dtype=np.uint8)
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    g = Graph()
    n0 = g.add_memset_node(buf=buf, value=0, n_bytes=n*4)
    n1 = g.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                              params={"A": A, "B": B, "OUT": OUT},
                              kernel_name="vec_add")
    g.add_dependency(n0, n1)
    n2 = g.add_memset_node(buf=buf, value=0, n_bytes=n*4)
    g.add_dependency(n1, n2)
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    print(f"Graph (memset+kernel+memset): {cycles} cycles")
    print(f"OUT[0:4] = {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# graph_memset_zero

Phase 13 demo: graph with memset → kernel → memset, demonstrating memset
node use in DAG. 50-cycle overhead per memset.

## Run
```
python examples/graph_memset_zero/run.py
```

## Tutorial
docs/tutorial/53-graph-memset-node.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_graph_memset_zero.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/graph_memset_zero/ tests/parity/test_graph_memset_zero.py
git commit -m "feat(examples): graph_memset_zero — memset-kernel-memset chain"
```

---

### Task 3: Tag M1

```bash
.venv/bin/pytest -q -m "not slow"
git tag M1-phase13-complete
```

---

## Milestone M2: Child graph + with_child example

### Task 4: ChildGraphNodeArgs + add_child_graph_node + GraphExec child_graph branch

**Files:**
- Modify: `gpusim/graph/node.py` (add ChildGraphNodeArgs + GraphNode.child_graph_args field)
- Modify: `gpusim/graph/graph.py` (add_child_graph_node)
- Modify: `gpusim/graph/exec.py` (child_graph branch)
- Test: `tests/unit/graph/test_child_graph_node.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_child_graph_node_args():
    from gpusim.graph.node import ChildGraphNodeArgs
    from gpusim.graph.graph import Graph
    inner = Graph()
    a = ChildGraphNodeArgs(graph=inner)
    assert a.graph is inner


def test_graph_add_child_graph_node():
    from gpusim.graph.graph import Graph
    inner = Graph()
    inner.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="inner_k")
    outer = Graph()
    nid = outer.add_child_graph_node(graph=inner)
    assert outer.nodes[0].type == "child_graph"
    assert outer.nodes[0].child_graph_args.graph is inner


def test_graph_exec_child_graph_executes():
    """Child graph nested execution produces correct outputs."""
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
    inner = Graph()
    inner.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                            params={"A": A, "B": B, "OUT": OUT},
                            kernel_name="vec_add")
    
    outer = Graph()
    outer.add_child_graph_node(graph=inner)
    
    exec = outer.instantiate(cfg)
    cycles = exec.launch()
    
    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add ChildGraphNodeArgs to node.py:**

```python
@dataclass
class ChildGraphNodeArgs:
    graph: object
```

Add to GraphNode:
```python
    child_graph_args: ChildGraphNodeArgs | None = None    # NEW Phase 13
```

- [ ] **Step 4: Add to Graph (gpusim/graph/graph.py):**

```python
    def add_child_graph_node(self, *, graph: "Graph") -> int:
        from gpusim.graph.node import ChildGraphNodeArgs
        nid = self._next_id; self._next_id += 1
        args = ChildGraphNodeArgs(graph=graph)
        self.nodes.append(GraphNode(node_id=nid, type="child_graph",
                                       child_graph_args=args))
        return nid
```

- [ ] **Step 5: Add child_graph branch to GraphExec.launch:**

```python
            elif node.type == "child_graph":
                a = node.child_graph_args
                child_exec = a.graph.instantiate(self.config)
                total_cycles += child_exec.launch()
```

- [ ] **Step 6: Update __init__.py to export ChildGraphNodeArgs.**

- [ ] **Step 7: Run + commit**

```
.venv/bin/pytest tests/unit/graph/test_child_graph_node.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/graph/ tests/unit/graph/test_child_graph_node.py
git commit -m "feat(graph): ChildGraphNodeArgs + add_child_graph_node + nested execution"
```

---

### Task 5: Example graph_with_child

**Files:**
- Create: `examples/graph_with_child/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_graph_with_child.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_with_child"


def test_graph_with_child_correctness():
    """Outer graph contains a child graph (with 2 kernel nodes)."""
    import gpusim
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
    # Build child graph: 2 kernel nodes (chain)
    inner = Graph()
    n0 = inner.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params={"A": A, "B": B, "OUT": OUT},
                                  kernel_name="vec_add_inner_0")
    n1 = inner.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params={"A": A, "B": B, "OUT": OUT},
                                  kernel_name="vec_add_inner_1")
    inner.add_dependency(n0, n1)
    
    # Build outer graph with one child node
    outer = Graph()
    outer.add_child_graph_node(graph=inner)
    
    exec = outer.instantiate(cfg)
    cycles = exec.launch()
    
    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0
```

- [ ] **Step 2: kernel.ptx (vec_add — same).**

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
    inner = Graph()
    n0 = inner.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params={"A": A, "B": B, "OUT": OUT},
                                  kernel_name="vec_add_inner_0")
    n1 = inner.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params={"A": A, "B": B, "OUT": OUT},
                                  kernel_name="vec_add_inner_1")
    inner.add_dependency(n0, n1)
    outer = Graph()
    outer.add_child_graph_node(graph=inner)
    exec = outer.instantiate(cfg)
    cycles = exec.launch()
    print(f"Graph (outer with 1 child of 2 kernels): {cycles} cycles")
    print(f"OUT[0:4] = {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# graph_with_child

Phase 13 demo: outer graph contains a child graph node (with 2 nested
kernel nodes). Demonstrates child graph nesting.

## Run
```
python examples/graph_with_child/run.py
```

## Tutorial
docs/tutorial/51-graph-child-nested-dag.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_graph_with_child.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/graph_with_child/ tests/parity/test_graph_with_child.py
git commit -m "feat(examples): graph_with_child — nested DAG with child graph node"
```

---

### Task 6: Tag M2

```bash
.venv/bin/pytest -q -m "not slow"
git tag M2-phase13-complete
```

---

## Milestone M3: Update API + update_replay example

### Task 7: GraphExec.update_kernel_node_params

**Files:**
- Modify: `gpusim/graph/exec.py` (add update_kernel_node_params + _update_count field)
- Test: `tests/unit/graph/test_graph_update.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_update_kernel_node_params():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k0")
    exec = g.instantiate(cfg)
    exec.update_kernel_node_params(0, kernel_name="k0_renamed")
    assert g.nodes[0].kernel_args.kernel_name == "k0_renamed"
    assert exec._update_count == 1


def test_update_invalid_node_raises():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    import pytest
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    exec = g.instantiate(cfg)
    with pytest.raises(ValueError, match="not found"):
        exec.update_kernel_node_params(99, kernel_name="x")


def test_update_non_kernel_raises():
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    import pytest
    cfg = load_default()
    g = Graph()
    g.add_memset_node(buf=np.zeros(8, dtype=np.uint8), value=0, n_bytes=8)
    exec = g.instantiate(cfg)
    with pytest.raises(ValueError, match="not kernel"):
        exec.update_kernel_node_params(0, kernel_name="x")


def test_update_unknown_field_raises():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    import pytest
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    exec = g.instantiate(cfg)
    with pytest.raises(ValueError, match="unknown update field"):
        exec.update_kernel_node_params(0, bogus_field="x")
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add update_kernel_node_params to GraphExec:**

In `gpusim/graph/exec.py`, add field + method to GraphExec:

```python
@dataclass
class GraphExec:
    # ... existing fields ...
    _update_count: int = 0    # NEW Phase 13
    
    def update_kernel_node_params(self, node_id: int, **kwargs) -> None:
        """Modify a kernel node's params in place. Phase 13."""
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

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/graph/test_graph_update.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/graph/exec.py tests/unit/graph/test_graph_update.py
git commit -m "feat(graph): GraphExec.update_kernel_node_params + validation"
```

---

### Task 8: Example graph_update_replay

**Files:**
- Create: `examples/graph_update_replay/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_graph_update_replay.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_update_replay"


def test_graph_update_replay_correctness():
    """Capture single-kernel graph; replay 3 times. Between replays, swap input buffers."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 32
    A1 = np.full(n, 1.0, dtype=np.float32)
    B1 = np.full(n, 1.0, dtype=np.float32)
    A2 = np.full(n, 5.0, dtype=np.float32)
    B2 = np.full(n, 3.0, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A1, "B": B1, "OUT": OUT},
              kernel_name="vec_add", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    
    # Launch 1 with A1+B1
    exec.launch()
    np.testing.assert_array_equal(OUT, A1 + B1)
    
    # Update params to A2+B2
    exec.update_kernel_node_params(0, params={"A": A2, "B": B2, "OUT": OUT})
    exec.launch()
    np.testing.assert_array_equal(OUT, A2 + B2)
    
    assert exec._update_count == 1
```

- [ ] **Step 2: kernel.ptx (vec_add — same).**

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
    A1 = np.full(n, 1.0, dtype=np.float32)
    B1 = np.full(n, 1.0, dtype=np.float32)
    A2 = np.full(n, 5.0, dtype=np.float32)
    B2 = np.full(n, 3.0, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A1, "B": B1, "OUT": OUT},
              kernel_name="vec_add", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    exec.launch()
    print(f"Replay 1 (A=1, B=1): OUT[0:4] = {list(OUT[0:4])}")
    exec.update_kernel_node_params(0, params={"A": A2, "B": B2, "OUT": OUT})
    exec.launch()
    print(f"Replay 2 (A=5, B=3): OUT[0:4] = {list(OUT[0:4])}")
    print(f"Update count: {exec._update_count}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# graph_update_replay

Phase 13 demo: capture single-kernel graph; replay with updated input buffers
between replays via `update_kernel_node_params`.

## Run
```
python examples/graph_update_replay/run.py
```

## Tutorial
docs/tutorial/52-graph-update-api-replay.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_graph_update_replay.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/graph_update_replay/ tests/parity/test_graph_update_replay.py
git commit -m "feat(examples): graph_update_replay — update params between replays"
```

---

### Task 9: Tag M3

```bash
.venv/bin/pytest -q -m "not slow"
git tag M3-phase13-complete
```

---

## Milestone M4: 2 metrics

### Task 10: graph_child_depth + graph_update_count metrics

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Test: `tests/unit/analysis/test_phase13_metrics.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_graph_child_depth_no_children():
    from gpusim.analysis.metrics import graph_child_depth
    from gpusim.graph.graph import Graph
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    assert graph_child_depth(g) == 0


def test_graph_child_depth_one_level():
    from gpusim.analysis.metrics import graph_child_depth
    from gpusim.graph.graph import Graph
    inner = Graph()
    inner.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="k_inner")
    outer = Graph()
    outer.add_child_graph_node(graph=inner)
    assert graph_child_depth(outer) == 1


def test_graph_child_depth_two_levels():
    from gpusim.analysis.metrics import graph_child_depth
    from gpusim.graph.graph import Graph
    inner_inner = Graph()
    inner_inner.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                                  params={}, kernel_name="k")
    inner = Graph()
    inner.add_child_graph_node(graph=inner_inner)
    outer = Graph()
    outer.add_child_graph_node(graph=inner)
    assert graph_child_depth(outer) == 2


def test_graph_update_count():
    from gpusim.analysis.metrics import graph_update_count
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    exec = g.instantiate(cfg)
    assert graph_update_count(exec) == 0
    exec.update_kernel_node_params(0, kernel_name="k2")
    assert graph_update_count(exec) == 1
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Append metrics to gpusim/analysis/metrics.py:**

```python
def graph_child_depth(graph) -> int:
    """Maximum nesting depth of child graphs.
    Returns 0 if no child graph nodes; 1 if single level; etc."""
    if not graph.nodes:
        return 0
    max_depth = 0
    for n in graph.nodes:
        if n.type == "child_graph" and n.child_graph_args is not None:
            child_depth = 1 + graph_child_depth(n.child_graph_args.graph)
            max_depth = max(max_depth, child_depth)
    return max_depth


def graph_update_count(graph_exec) -> int:
    """Number of update_kernel_node_params calls performed on this GraphExec."""
    return getattr(graph_exec, "_update_count", 0)
```

- [ ] **Step 4: Run + commit + tag M4**

```
.venv/bin/pytest tests/unit/analysis/test_phase13_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/analysis/metrics.py tests/unit/analysis/test_phase13_metrics.py
git commit -m "feat(analysis): graph_child_depth + graph_update_count"
git tag M4-phase13-complete
```

---

### Task 11: (consolidated)

(Reserved.)

---

## Milestone M5: Tutorials + microbench + ship

### Task 12: 3 tutorial chapters 51-53

**Files:**
- Create: `docs/tutorial/{51,52,53}-*.md`

- [ ] **Step 1: Read** `docs/tutorial/50-pytorch-dist-wrapper.md` for style reference.

- [ ] **Step 2: Write 3 chapters** (~500-700 words each):

**Chapter 51 — graph-child-nested-dag:**
- ChildGraphNodeArgs + add_child_graph_node + nested execution
- graph_with_child demo
- 看模拟器: `graph_child_depth` metric
- 改一改: 2-level nesting (child within child)
- 真机对照: cudaGraphAddChildGraphNode

**Chapter 52 — graph-update-api-replay:**
- GraphExec.update_kernel_node_params; in-place update without re-instantiating
- graph_update_replay demo
- 看模拟器: `graph_update_count` metric
- 改一改: update grid/block size between replays
- 真机对照: cudaGraphExecUpdate (with restrictions)

**Chapter 53 — graph-memset-node:**
- MemsetNodeArgs + add_memset_node; functional fill + 50-cycle overhead
- graph_memset_zero demo
- 看模拟器: memset cycles in launch trace
- 改一改: replace memset with kernel fill — observe cycle difference
- 真机对照: cudaGraphAddMemsetNode (real GPU dispatches actual memset kernel)

```bash
git add docs/tutorial/51-graph-child-nested-dag.md \
        docs/tutorial/52-graph-update-api-replay.md \
        docs/tutorial/53-graph-memset-node.md
git commit -m "docs(tutorial): chapters 51-53 — Phase 13 graphs completion"
```

---

### Task 13: Phase 13 microbench + 3 ref stubs

**Files:**
- Create: `tests/microbench/test_phase13_facts.py`
- Create: `tests/microbench/test_phase13_runtime.py`
- Modify: `tests/reference/gen_reference.py`
- Create: 3 ref JSONs

- [ ] **Step 1: test_phase13_facts.py:**

```python
"""Phase 13 microbench — graphs completion facts."""


def test_memset_node_cycles_50():
    """Memset node = 50 cycles."""
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    buf = np.full(8, 1, dtype=np.uint8)
    g = Graph()
    g.add_memset_node(buf=buf, value=0, n_bytes=8)
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    assert cycles == 50


def test_child_depth_3_levels():
    """3-level nested graph depth = 3."""
    from gpusim.graph.graph import Graph
    from gpusim.analysis.metrics import graph_child_depth
    g3 = Graph()
    g3.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                          params={}, kernel_name="leaf")
    g2 = Graph()
    g2.add_child_graph_node(graph=g3)
    g1 = Graph()
    g1.add_child_graph_node(graph=g2)
    g0 = Graph()
    g0.add_child_graph_node(graph=g1)
    assert graph_child_depth(g0) == 3


def test_update_count_tracks_calls():
    """Update count increments per call."""
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    exec = g.instantiate(cfg)
    for _ in range(5):
        exec.update_kernel_node_params(0, kernel_name="k_new")
    assert exec._update_count == 5
```

- [ ] **Step 2: test_phase13_runtime.py:**

```python
import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_graph_with_child_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_with_child"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_graph_update_replay_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_update_replay"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
```

- [ ] **Step 3: Append 3 kernel names to gen_reference.py:**

```python
"graph_memset_zero",
"graph_with_child",
"graph_update_replay",
```

- [ ] **Step 4: Create 3 ref JSONs:**

```bash
for k in graph_memset_zero graph_with_child graph_update_replay; do
  cat > tests/reference/data/$k.ref.json <<JSON
{
  "kernel": "$k",
  "phase": 13,
  "metrics": {
    "graph_child_depth": null,
    "graph_update_count": null
  },
  "tolerance": {
    "graph_child_depth_pct": 0,
    "graph_update_count_pct": 0
  },
  "notes": "Generated by gen_reference.py on real Hopper. null = not yet measured."
}
JSON
done
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/microbench/test_phase13_facts.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add tests/microbench/test_phase13_facts.py tests/microbench/test_phase13_runtime.py \
        tests/reference/gen_reference.py tests/reference/data/graph_*.ref.json
git commit -m "test(microbench+reference): Phase 13 facts + 3 ref stubs"
```

---

### Task 14: Phase 1-12 regression rename

```bash
git mv tests/parity/test_phase1_11_examples_unchanged.py tests/parity/test_phase1_12_examples_unchanged.py
```

Edit:
- Rename `PHASE_1_11_EXAMPLES` → `PHASE_1_12_EXAMPLES`
- Append 3 Phase 12 examples: `reduce_scatter_fsdp`, `send_recv_pipeline_parallel`, `pytorch_dist_simple`
- Update test function names from `phase_1_11_*` → `phase_1_12_*` if any

```bash
git add tests/parity/test_phase1_12_examples_unchanged.py
git commit -m "test(regression): rename phase1_11 → phase1_12 + 3 Phase 12 examples"
```

---

### Task 15: README v13 + final tag phase13-complete

**Files:**
- Modify: `README.md`

- [ ] **Update to v13:**
- Phase status: 1-13 ✅
- Phase 13 features section:
  - Child graph nodes (nested DAG)
  - Memset nodes (50-cycle overhead)
  - GraphExec.update_kernel_node_params (in-place modification between replays)
  - 2 metrics (graph_child_depth, graph_update_count)
  - 3 examples + 3 tutorials chapters 51-53
  - Backward compatible: Phase 1-12 unchanged
- Examples list: add 3 (was 49, now 52)
- Tutorials list: add 51-53 (was 50, now 53)

- [ ] **Run final suite + 3 examples:**

```
.venv/bin/pytest -q -m "not slow"
.venv/bin/python examples/graph_memset_zero/run.py
.venv/bin/python examples/graph_with_child/run.py
.venv/bin/python examples/graph_update_replay/run.py
```

- [ ] **Commit + tag:**

```bash
git add README.md
git commit -m "docs(readme): v13 — Phase 13 capabilities (graphs completion)"
git tag phase13-complete
```

---

### Task 16: Final sanity sweep

```
.venv/bin/pytest -q -m "not slow"
.venv/bin/pytest tests/parity/test_phase1_12_examples_unchanged.py -v
```

Phase 13 ships when all tasks complete + tags landed.

---

## End-of-plan checklist

- [ ] M1 (Memset + memset_zero): T1-T3
- [ ] M2 (Child graph + with_child): T4-T6
- [ ] M3 (Update API + update_replay): T7-T9
- [ ] M4 (2 metrics): T10-T11
- [ ] M5 (Tutorials + microbench + regression + README): T12-T16
- [ ] All 5 milestone tags + phase13-complete
- [ ] Phase 1-12 regression unbroken
- [ ] 3 new examples + 3 tutorials shipped
- [ ] README v13 reflects Phase 13
