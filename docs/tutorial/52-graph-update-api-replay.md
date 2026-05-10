# Chapter 52 — Graph Update API and Replay with Param Swap

## Why Update Instead of Re-instantiate?

One of the key performance advantages of CUDA Graphs is that you pay the graph instantiation cost once and then replay the frozen execution plan cheaply on every iteration. But training workloads are not perfectly static: between replays you might want to change which input buffer a kernel reads from, adjust the grid dimensions for a variable-length batch, or retarget output pointers.

The naive solution — destroy the old graph, rebuild from scratch, re-instantiate, replay — defeats the point. `cudaGraphExecUpdate` on real NVIDIA hardware lets you swap out node parameters in-place on an already-instantiated `GraphExec`, without discarding the compiled execution plan.

Phase 13 ships the simulator equivalent: `GraphExec.update_kernel_node_params(node_id, **kwargs)`. It modifies the `KernelArgs` object for the specified node directly, so the next `launch()` picks up the new values automatically.

```python
class GraphExec:
    _update_count: int = 0

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

Allowed update fields: `ptx_src`, `grid`, `block`, `params`, `kernel_name`. Passing any other keyword raises `ValueError("unknown update field: ...")`. Targeting a non-kernel node (e.g. a memset node) raises `ValueError("not kernel")`.

## The graph_update_replay Demo

The `graph_update_replay` example captures a vector-addition kernel into a graph, replays it with one set of inputs, then swaps the input buffers and replays again — all without re-instantiating the graph.

```python
import numpy as np, pathlib
from gpusim.api import Stream
from gpusim.config.loader import load_default

n = 32
A1 = np.full(n, 1.0, dtype=np.float32)
B1 = np.full(n, 1.0, dtype=np.float32)
A2 = np.full(n, 5.0, dtype=np.float32)
B2 = np.full(n, 3.0, dtype=np.float32)
OUT = np.zeros(n, dtype=np.float32)

cfg = load_default()
ptx = pathlib.Path("examples/graph_update_replay/kernel.ptx").read_text()

# Capture the kernel into a graph using stream capture
s = Stream()
s.begin_capture()
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
         params={"A": A1, "B": B1, "OUT": OUT},
         kernel_name="vec_add", config=cfg)
g = s.end_capture()
exec = g.instantiate(cfg)

# Replay 1: A1 + B1 → each element = 2.0
exec.launch()
print(f"Replay 1: OUT[0:4] = {list(OUT[0:4])}")   # [2.0, 2.0, 2.0, 2.0]

# Swap params in-place — no re-instantiation
exec.update_kernel_node_params(0, params={"A": A2, "B": B2, "OUT": OUT})

# Replay 2: A2 + B2 → each element = 8.0
exec.launch()
print(f"Replay 2: OUT[0:4] = {list(OUT[0:4])}")   # [8.0, 8.0, 8.0, 8.0]

print(f"Update count: {exec._update_count}")   # 1
```

Run it:

```bash
python examples/graph_update_replay/run.py
```

The key sequence:

1. `s.begin_capture()` puts the stream into capture mode.
2. `s.launch(...)` records a kernel node into the graph (no actual execution).
3. `s.end_capture()` returns a `Graph` with one kernel node (node_id=0).
4. `g.instantiate(cfg)` creates a `GraphExec` with `_update_count=0`.
5. First `exec.launch()` runs the kernel with the original `params`.
6. `exec.update_kernel_node_params(0, params={...})` overwrites `node.kernel_args.params` in place and increments `_update_count` to 1.
7. Second `exec.launch()` runs the kernel with the new params, reading `A2` and `B2`.

## 看模拟器

**用 `graph_update_count` 监控参数替换频率：**

The `graph_update_count` metric (Phase 13) reads `GraphExec._update_count`:

```python
from gpusim.analysis.metrics import graph_update_count

print(graph_update_count(exec))   # → 1 after one update_kernel_node_params call
```

This is useful for verifying that your training loop is calling the update API at the expected cadence — for example, once per epoch boundary when you rotate dataset shards. If you expect 10 updates over a 100-step training loop and `graph_update_count` returns 100, you likely have an off-by-one in your update schedule.

In a multi-graph workload where each graph holds one layer's forward-pass kernel, you can collect `graph_update_count` for every graph and build a histogram to see which layers were updated most frequently.

## 改一改

**尝试在两次回放之间更新 grid/block 而不是 params：**

The update API is not limited to `params` — you can also swap `grid`, `block`, `ptx_src`, or `kernel_name`:

```python
# Capture with a 1×1 grid
s = Stream()
s.begin_capture()
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
         params={"A": A1, "B": B1, "OUT": OUT},
         kernel_name="vec_add", config=cfg)
g = s.end_capture()
exec = g.instantiate(cfg)

exec.launch()   # runs 1 CTA

# Scale up: update to 4 CTAs
exec.update_kernel_node_params(0, grid=(4,1,1))
exec.launch()   # now runs 4 CTAs
```

This pattern models variable-batch training where the grid size changes between steps. Try also swapping `block` from `(32,1,1)` to `(64,1,1)` and observe how the cycle count changes (more warps per CTA → higher occupancy in the simulator's warp scheduler).

## 真机对照

On real CUDA, the equivalent API is `cudaGraphExecUpdate`:

```c
cudaGraphExec_t exec;
cudaGraph_t updatedGraph;
cudaGraphExecUpdateResult updateResult;

// ... build updatedGraph with new params ...

cudaError_t err = cudaGraphExecUpdate(exec, updatedGraph, &updateResult);
if (err != cudaSuccess || updateResult != cudaGraphExecUpdateSuccess) {
    // Update failed — must re-instantiate
    cudaGraphInstantiate(&exec, updatedGraph, NULL, NULL, 0);
}
```

| Aspect | `gpusim` | CUDA runtime |
|---|---|---|
| **API** | `exec.update_kernel_node_params(node_id, **kwargs)` | `cudaGraphExecUpdate(exec, newGraph, &result)` takes a full new graph |
| **Granularity** | Per-node, per-field | Whole-graph diff (driver detects structural changes) |
| **Failure mode** | `ValueError` on bad field or non-kernel node | `updateResult != cudaGraphExecUpdateSuccess` → must re-instantiate |
| **Count tracking** | `exec._update_count` | Not built-in; instrument with profiler |
| **Structural changes** | Not supported (node count must match) | Not supported (topology must be identical) |

The most important restriction on real hardware: `cudaGraphExecUpdate` requires that the updated graph has **exactly the same topology** (same node types, same edge structure) as the originally instantiated graph. You can only change node parameters, not add or remove nodes. The simulator enforces the same constraint — you cannot change a kernel node's `node_id`, only its execution parameters.
