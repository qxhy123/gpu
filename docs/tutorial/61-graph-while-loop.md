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
