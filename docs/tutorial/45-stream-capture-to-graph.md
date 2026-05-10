# Chapter 45 — Stream Capture to Graph

## The Capture Idea

Chapter 44 built a graph by hand — calling `add_kernel_node` and `add_dependency` for each node. That works, but it requires you to manage node IDs and wire dependencies explicitly. For a long training step with dozens of operations, this becomes tedious.

Stream capture offers a better path: **record first, then replay**. You put a stream into capture mode, issue your normal `stream.launch()` calls exactly as if you were running them immediately, and then stop capture. The stream silently records each launch as a graph node, inferring launch-order dependencies automatically. The result is a `Graph` object ready to instantiate.

This mirrors the philosophy of CUDA stream capture on hardware: no code change to the kernel calls themselves, only wrapping brackets around the sequence.

## begin_capture / end_capture

```python
from gpusim.api import Stream
from gpusim.config.loader import load_default

cfg = load_default()
s = Stream()

# Enter capture mode — subsequent launches are recorded, not executed
s.begin_capture()

# Issue three launches: implicit dependency from launch order
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"A": A, "B": B, "OUT": OUT},
          kernel_name="vec_add_0", config=cfg)
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"A": A, "B": B, "OUT": OUT},
          kernel_name="vec_add_1", config=cfg)
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"A": A, "B": B, "OUT": OUT},
          kernel_name="vec_add_2", config=cfg)

# End capture — returns a Graph object
g = s.end_capture()

print(f"Captured: {len(g.nodes)} nodes, {len(g.edges)} edges")
# Output: Captured: 3 nodes, 2 edges
```

The key detail: `begin_capture` sets an internal flag on the stream. While the flag is set, each call to `stream.launch()` appends a `GraphNode` to an internal node list and wires an edge from the previous node to the current one. When `end_capture` is called, it packages these nodes and edges into a `Graph` and clears the capture flag. Nothing executes during capture.

## The graph_capture_from_stream Demo

The complete example lives at `examples/graph_capture_from_stream/run.py`:

```bash
python examples/graph_capture_from_stream/run.py
```

Expected output:
```
Captured graph: 3 nodes, 2 edges
Replay: <N> cycles, OUT[0:4] = [<values>]
```

After `end_capture`, the resulting graph is identical to what you would have built explicitly in Chapter 44 for the same 3-kernel sequence. The edges count (2 for 3 nodes) reflects the implicit chain: node 0 → node 1 → node 2.

The demo then calls `g.instantiate(cfg)` and `exec.launch()` exactly as in the explicit-build chapter. The two paths — explicit build and stream capture — converge at the `Graph` object.

## 看模拟器

**检查捕获图的节点数和边数：**

After `end_capture` you can inspect the graph before instantiating it:

```python
s = Stream()
s.begin_capture()
for i in range(5):
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": OUT},
              kernel_name=f"k{i}", config=cfg)
g = s.end_capture()

print(f"nodes: {len(g.nodes)}")  # 5
print(f"edges: {len(g.edges)}")  # 4  (always n_nodes - 1 for a chain)

# Edge list shows the implicit dependencies
for parent, child in g.edges:
    print(f"  node {parent} → node {child}")
```

For N launches from a single stream, the captured graph always has exactly N nodes and N-1 edges forming a chain. This is because stream semantics guarantee that launch order implies dependency: launch N+1 cannot begin until launch N finishes.

The captured graph is structurally identical to one built with the explicit API. `graph_dag_depth(g)` returns N for an N-node chain.

## 改一改

**用 Event 在跨 stream 之间加依赖（cross-stream events）：**

Stream capture only tracks launches on the captured stream, so by default you get a chain for all operations on that stream. To capture a diamond or fork-join pattern, you need cross-stream events.

On real CUDA hardware, you can record an event on one stream during capture and wait for it on another captured stream, creating an inter-stream edge. The simulator's event API supports recording during capture:

```python
from gpusim.api import Stream, Event

s_main = Stream()
s_side = Stream()
ev = Event()

s_main.begin_capture()
s_main.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"A": A, "B": B, "OUT": OUT},
               kernel_name="root", config=cfg)
ev.record(s_main)  # record after root completes

# Side stream: waits for root, then runs in parallel
s_side.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"A": A, "B": B, "OUT": OUT},
               kernel_name="parallel_branch", config=cfg)
s_main.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"A": A, "B": B, "OUT": OUT},
               kernel_name="main_branch", config=cfg)

g = s_main.end_capture()
```

This pattern matches `cudaEventRecord` / `cudaStreamWaitEvent` used during `cudaStreamBeginCapture` on hardware to express cross-stream dependencies within a single graph.

## 真机对照

The CUDA runtime capture API:

```c
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// These launches are recorded, not executed
myKernel_A<<<grid, block, 0, stream>>>(args_A);
myKernel_B<<<grid, block, 0, stream>>>(args_B);
myKernel_C<<<grid, block, 0, stream>>>(args_C);

cudaGraph_t graph;
cudaStreamEndCapture(stream, &graph);

// Instantiate and replay
cudaGraphExec_t exec;
cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
for (int i = 0; i < 100; i++) {
    cudaGraphLaunch(exec, stream);
}
```

| Simulator | CUDA runtime |
|---|---|
| `s.begin_capture()` | `cudaStreamBeginCapture()` |
| `s.launch(...)` during capture | `kernel<<<...>>>()` during capture |
| `g = s.end_capture()` | `cudaStreamEndCapture(stream, &graph)` |
| `exec.launch()` | `cudaGraphLaunch(exec, stream)` |

The capture modes (`cudaStreamCaptureModeGlobal`, `ModeThreadLocal`, `ModeRelaxed`) control which streams are considered part of the same capture. The simulator's single-stream capture corresponds to `ModeThreadLocal` semantics: only launches on the explicitly captured stream are recorded; other streams run normally.

## Connection to Chapter 44

The explicit-build path (Chapter 44) and the stream-capture path (this chapter) produce identical `Graph` objects. The choice between them is a matter of ergonomics:

- **Explicit build**: better when you need precise control over the graph topology (e.g., injecting memcpy or event nodes between kernels, building diamond shapes).
- **Stream capture**: better when you are refactoring existing sequential code — wrap existing launch calls with `begin_capture`/`end_capture` and get a graph with zero other changes.

Chapter 46 will show how to measure the performance benefit of replaying the same graph many times.
