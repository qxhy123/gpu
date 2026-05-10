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
