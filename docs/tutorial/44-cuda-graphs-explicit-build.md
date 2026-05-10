# Chapter 44 — CUDA Graphs: Explicit Build

## Why Graphs?

Every time you call `Stream.launch()` in the simulator (or `cudaLaunchKernel` on hardware), the driver must re-validate the launch arguments, re-walk the dependency chain, and schedule the kernel afresh. For a sequence of short kernels — like a handful of elementwise operations or a small transformer attention step — that per-launch overhead dominates actual compute time.

CUDA Graphs solve this by capturing the kernel sequence **once**, compiling it into an immutable execution plan, and then replaying that plan with a single launch call. The overhead is paid once at instantiation; subsequent replays cost only the time to hand the pre-validated plan to the hardware scheduler.

The simulator's `Graph` class mirrors the CUDA runtime graph API exactly. You build a graph explicitly using three calls: `add_kernel_node`, `add_dependency`, and `instantiate`. The result is a `GraphExec` object you can replay as many times as you like.

## The Builder API

```python
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default

cfg = load_default()
g = Graph()

# Add three kernel nodes — returns integer node IDs
nid0 = g.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                          params={"A": A, "B": B, "OUT": OUT},
                          kernel_name="stage_0")
nid1 = g.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                          params={"A": A, "B": B, "OUT": OUT},
                          kernel_name="stage_1")
nid2 = g.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                          params={"A": A, "B": B, "OUT": OUT},
                          kernel_name="stage_2")

# Wire them into a linear chain: 0 → 1 → 2
g.add_dependency(nid0, nid1)
g.add_dependency(nid1, nid2)

# Instantiate: topo-sort + prevalidate
exec = g.instantiate(cfg)

# Replay — can call launch() as many times as needed
cycles = exec.launch()
print(f"3-kernel chain: {cycles} cycles")
```

Internally, `instantiate` runs Kahn's topological sort on the node-edge DAG to compute the execution order. The topo order is stored in `GraphExec.topo_order` and is fixed for the lifetime of the exec object. Each `launch()` replays the nodes in that order.

## The 3-Kernel Chain Demo

The complete demo lives at `examples/graph_explicit_build/run.py`. It builds a 3-node linear chain, instantiates, and replays once:

```python
python examples/graph_explicit_build/run.py
```

Expected output:
```
Graph (3-kernel chain): <N> cycles
OUT[0:4] = [<values>]
```

The cycle count reflects the sum of all three kernel executions in dependency order. The simulator serializes dependent nodes: `stage_1` cannot begin until `stage_0` completes, and `stage_2` cannot begin until `stage_1` completes.

## 看模拟器

**用 `graph_dag_depth` 和 `graph_node_type_breakdown` 量化图结构：**

After building the graph you can inspect its structure analytically without launching it:

```python
from gpusim.analysis.metrics import graph_dag_depth, graph_node_type_breakdown

depth = graph_dag_depth(g)
breakdown = graph_node_type_breakdown(g)

print(f"DAG depth: {depth}")           # longest dependency chain
print(f"Node breakdown: {breakdown}")  # {"kernel": 3, "memcpy": 0, "event": 0}
```

For the 3-kernel chain: `graph_dag_depth` returns 3 (nodes 0→1→2 form a chain of length 3). `graph_node_type_breakdown` returns `{"kernel": 3, "memcpy": 0, "event": 0}`.

`graph_dag_depth` is the critical path length. A chain of depth N means you must wait for N serial kernel completions before the graph finishes. A diamond of depth 3 (two parallel middle nodes) finishes in 3 serial waits instead of 4, making it faster even with 4 nodes.

## 改一改

**把链改成菱形依赖（diamond-shape dependency）：**

Change the linear chain (0→1→2→3) into a diamond (0→1, 0→2, 1→3, 2→3). Nodes 1 and 2 can execute in parallel on independent SMs; node 3 waits for both:

```python
g = Graph()
nids = []
for i in range(4):
    nid = g.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                             params={"A": A, "B": B, "OUT": OUT},
                             kernel_name=f"k{i}")
    nids.append(nid)

g.add_dependency(nids[0], nids[1])  # root → left branch
g.add_dependency(nids[0], nids[2])  # root → right branch
g.add_dependency(nids[1], nids[3])  # left branch → sink
g.add_dependency(nids[2], nids[3])  # right branch → sink

print(graph_dag_depth(g))  # 3, not 4: two branches run in parallel
```

Note that `graph_dag_depth` reports 3 for the diamond (source → one branch → sink). The simulator currently serializes all dependent nodes (it does not yet model true SM-level parallelism for independent graph branches), but the depth metric correctly reflects the theoretical minimum serial depth.

## 真机对照

On a real CUDA device the equivalent code uses the explicit-build graph API:

```c
// Create empty graph
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Add kernel nodes with cudaKernelNodeParams
cudaGraphNode_t n0, n1, n2;
cudaKernelNodeParams p = {...};
cudaGraphAddKernelNode(&n0, graph, NULL, 0, &p);     // no dependencies
cudaGraphAddKernelNode(&n1, graph, &n0, 1, &p);      // depends on n0
cudaGraphAddKernelNode(&n2, graph, &n1, 1, &p);      // depends on n1

// Instantiate and launch
cudaGraphExec_t exec;
cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
cudaGraphLaunch(exec, stream);
cudaStreamSynchronize(stream);
```

| Simulator | CUDA runtime |
|---|---|
| `Graph()` | `cudaGraphCreate()` |
| `add_kernel_node(...)` | `cudaGraphAddKernelNode()` |
| `add_dependency(a, b)` | dependency array in `cudaGraphAddKernelNode` |
| `g.instantiate(cfg)` | `cudaGraphInstantiate()` |
| `exec.launch()` | `cudaGraphLaunch()` |

One important difference: on the real API you pass dependency node handles directly to `cudaGraphAddKernelNode`. In the simulator's builder API, `add_dependency` takes integer node IDs returned by `add_kernel_node`, which is slightly more ergonomic for Python.

## Phase 11 Feature Summary So Far

This chapter introduced the explicit graph builder path. Chapter 45 will show the stream-capture path — how to build the same graph implicitly by recording a sequence of `stream.launch()` calls.
