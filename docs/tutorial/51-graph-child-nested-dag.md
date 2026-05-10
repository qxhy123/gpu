# Chapter 51 — Child Graph Nodes and Nested DAG Execution

## What Is a Child Graph Node?

Phase 13 extends the gpusim graph API with a fourth node type: the **child graph node**. Unlike a kernel node, memcpy node, or event node — which each represent a single operation — a child graph node embeds an entire sub-graph as a single vertex in the parent DAG.

This mirrors `cudaGraphAddChildGraphNode` on real NVIDIA hardware. When the parent `GraphExec.launch()` reaches a child graph node, it recursively instantiates and launches the embedded graph, accumulating its cycle cost into the parent's total. The child graph itself is a fully independent `Graph` object with its own nodes, edges, and topological sort.

The key types introduced in Phase 13:

```python
@dataclass
class ChildGraphNodeArgs:
    graph: Graph   # the embedded sub-graph

class Graph:
    def add_child_graph_node(self, *, graph: Graph) -> int:
        ...
```

The returned integer is the new node's ID in the parent graph, which you can pass to `graph.add_dependency(parent_id, child_graph_node_id)` to wire ordering constraints just like any other node.

## The graph_with_child Demo

The `graph_with_child` example builds a two-level DAG: an outer graph containing a single child graph node, where the child graph itself chains two kernel nodes.

```python
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default
import numpy as np, pathlib

n = 32
A = np.arange(n, dtype=np.float32)
B = np.arange(n, dtype=np.float32)
OUT = np.zeros(n, dtype=np.float32)

cfg = load_default()
ptx = pathlib.Path("examples/graph_with_child/kernel.ptx").read_text()

# Build the child (inner) graph: 2 kernel nodes in a chain
inner = Graph()
n0 = inner.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                            params={"A": A, "B": B, "OUT": OUT},
                            kernel_name="vec_add_inner_0")
n1 = inner.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                            params={"A": A, "B": B, "OUT": OUT},
                            kernel_name="vec_add_inner_1")
inner.add_dependency(n0, n1)

# Build the outer graph with one child node
outer = Graph()
outer.add_child_graph_node(graph=inner)

exec = outer.instantiate(cfg)
cycles = exec.launch()
print(f"Total cycles: {cycles}")   # includes both inner kernels
print(f"OUT[0:4] = {list(OUT[0:4])}")
```

Run it:

```bash
python examples/graph_with_child/run.py
```

When `exec.launch()` fires:

1. The outer graph's topological sort produces one node: the child graph node.
2. For that node, the executor calls `inner.instantiate(cfg)` and immediately calls `.launch()` on the result.
3. The inner `GraphExec` runs its two kernel nodes in dependency order (n0 before n1).
4. Both inner kernel cycle costs accumulate into `total_cycles` in the outer exec.

The final `OUT` array equals `A + B` because the second kernel node overwrites `OUT` with the same computation — demonstrating that the inner nodes execute in topological order, not in parallel.

## 看模拟器

**用 `graph_child_depth` 度量嵌套深度：**

The `graph_child_depth` metric (added in Phase 13) walks the graph recursively and returns the maximum nesting depth:

```python
from gpusim.analysis.metrics import graph_child_depth

print(graph_child_depth(outer))   # → 1  (one level: outer → inner)
print(graph_child_depth(inner))   # → 0  (inner has no child graph nodes)
```

A flat graph (all kernel/memcpy/event nodes) returns 0. A graph with one level of child graph embedding returns 1. A graph that contains a child graph that itself contains another child graph returns 2.

The metric is defined recursively:

```python
def graph_child_depth(graph) -> int:
    if not graph.nodes:
        return 0
    max_depth = 0
    for n in graph.nodes:
        if n.type == "child_graph" and n.child_graph_args is not None:
            child_depth = 1 + graph_child_depth(n.child_graph_args.graph)
            max_depth = max(max_depth, child_depth)
    return max_depth
```

This is useful for validating that your graph construction is layered as intended — a depth of 0 on the outer graph means the child node was accidentally added to the wrong graph.

## 改一改

**尝试三层嵌套（child within child within outer）：**

Extend the demo to three levels by wrapping the inner graph inside yet another child:

```python
# Level 3 (deepest leaf)
leaf = Graph()
leaf.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                     params={"A": A, "B": B, "OUT": OUT},
                     kernel_name="leaf_k")

# Level 2: child of leaf
mid = Graph()
mid.add_child_graph_node(graph=leaf)

# Level 1: child of mid
top = Graph()
top.add_child_graph_node(graph=mid)

exec3 = top.instantiate(cfg)
cycles3 = exec3.launch()

from gpusim.analysis.metrics import graph_child_depth
assert graph_child_depth(top) == 2   # mid → leaf = depth 2 from top

print(f"3-level depth: {graph_child_depth(top)}")
print(f"Cycles: {cycles3}")
```

You can also add a memset node sibling to the child graph node in the outer graph, then wire a dependency so the memset runs before the child:

```python
import numpy as np
buf = np.full(n * 4, 99, dtype=np.uint8)
outer2 = Graph()
m0 = outer2.add_memset_node(buf=buf, value=0, n_bytes=n * 4)
c0 = outer2.add_child_graph_node(graph=inner)
outer2.add_dependency(m0, c0)   # memset fires before inner
```

## 真机对照

On real NVIDIA hardware, child graph nodes are created with `cudaGraphAddChildGraphNode`:

```c
cudaGraph_t parent, child;
cudaGraphCreate(&child, 0);

// ... add nodes to child ...

cudaGraphNode_t childNode;
cudaGraphAddChildGraphNode(&childNode, parent,
                            /*dependencies=*/NULL, /*numDeps=*/0,
                            child);   // embeds child into parent
```

| Aspect | `gpusim` | CUDA runtime |
|---|---|---|
| **API** | `graph.add_child_graph_node(graph=inner)` | `cudaGraphAddChildGraphNode(&node, parent, deps, ndeps, child)` |
| **Execution** | Recursive `instantiate + launch` at runtime | Driver re-maps child graph nodes into a single flat execution plan per `cudaGraphInstantiate` call |
| **Cycle accounting** | Child cycles accumulated into parent total | Child kernel time visible in Nsight Systems Graph swimlane |
| **Mutation** | Parent `GraphExec` re-instantiates child each replay | `cudaGraphExecChildGraphNodeSetParams` updates child params without re-instantiation |

The most important difference: real CUDA drivers **flatten** the child graph at `cudaGraphInstantiate` time, producing a single linear execution plan. The simulator instead re-instantiates the child graph at every `launch()` call. This is semantically equivalent but slightly slower — acceptable for a teaching simulator.
