# Chapter 53 — Memset Nodes in CUDA Graphs

## What Is a Memset Node?

A **memset node** fills a contiguous memory buffer with a single byte value. In a CUDA Graph it is a first-class node type — alongside kernel, memcpy, and event nodes — that participates in the DAG's dependency ordering and executes during `GraphExec.launch()`.

On real NVIDIA hardware, `cudaGraphAddMemsetNode` configures a `CUDA_MEMSET_NODE_PARAMS` struct (pointer, pitch, width, height, element size, value) and the driver dispatches a dedicated memset kernel when the graph fires. In gpusim the operation is simpler: we fill a NumPy array with the specified byte value and add a fixed 50-cycle cost to model the PCIe / HBM write bandwidth overhead.

Phase 13 introduces two new types:

```python
@dataclass
class MemsetNodeArgs:
    buf: object    # numpy array to fill
    value: int     # byte value (0–255)
    n_bytes: int   # number of bytes to fill

class Graph:
    def add_memset_node(self, *, buf, value: int, n_bytes: int) -> int:
        ...
```

The returned node ID works exactly like any other node ID — pass it to `graph.add_dependency(a, b)` to wire ordering between memset nodes and their neighbors in the DAG.

## The graph_memset_zero Demo

The `graph_memset_zero` example builds a three-node graph: memset → kernel → memset. The first memset zeroes a scratch buffer before the kernel writes results elsewhere; the second memset zeroes the scratch buffer again as a cleanup step.

```python
import numpy as np, pathlib
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default

n = 32
buf = np.full(n * 4, 99, dtype=np.uint8)   # scratch buffer, initially 0x63
A = np.arange(n, dtype=np.float32)
B = np.arange(n, dtype=np.float32)
OUT = np.zeros(n, dtype=np.float32)

cfg = load_default()
ptx = pathlib.Path("examples/graph_memset_zero/kernel.ptx").read_text()

g = Graph()

# Node 0: pre-zero the scratch buffer
n0 = g.add_memset_node(buf=buf, value=0, n_bytes=n * 4)

# Node 1: vec_add writes into OUT (separate from buf)
n1 = g.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                        params={"A": A, "B": B, "OUT": OUT},
                        kernel_name="vec_add")
g.add_dependency(n0, n1)   # memset must complete before kernel

# Node 2: post-zero buf (cleanup)
n2 = g.add_memset_node(buf=buf, value=0, n_bytes=n * 4)
g.add_dependency(n1, n2)   # kernel must complete before cleanup

exec = g.instantiate(cfg)
cycles = exec.launch()

print(f"Total cycles: {cycles}")       # ≥ 100 (2 memsets × 50 + kernel)
print(f"buf all zeros: {(buf == 0).all()}")   # True
print(f"OUT[0:4] = {list(OUT[0:4])}")         # [0.0, 1.0, 2.0, 3.0]
```

Run it:

```bash
python examples/graph_memset_zero/run.py
```

The execution order produced by topological sort is: n0 (memset) → n1 (kernel) → n2 (memset). Each memset writes its value byte into every element of `buf` using NumPy's broadcast assignment (`buf[:] = value`) and adds exactly 50 cycles.

## 看模拟器

**观察 memset 节点在 launch 追踪中的周期贡献：**

The 50-cycle per-memset model is intentionally fixed — it abstracts over HBM bandwidth, DMA queue latency, and transfer size. You can verify it with a single-node graph:

```python
import numpy as np
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default

cfg = load_default()
buf = np.full(8, 1, dtype=np.uint8)
g = Graph()
g.add_memset_node(buf=buf, value=0, n_bytes=8)
exec = g.instantiate(cfg)
cycles = exec.launch()

assert cycles == 50   # always exactly 50
assert (buf == 0).all()
```

In the three-node demo, the total cycles are at minimum 100 (two memsets) plus the kernel's execution time. Use `graph_node_type_breakdown` from Phase 11 to confirm node counts:

```python
from gpusim.analysis.metrics import graph_node_type_breakdown
breakdown = graph_node_type_breakdown(g)
print(breakdown)   # {"kernel": 1, "memcpy": 0, "event": 0, "memset": 2}
```

The memset count appears under the key `"memset"` in the breakdown dict — any unrecognized type falls through to a dict.setdefault pattern and is counted by its type string.

## 改一改

**用 kernel fill 替换 memset — 观察周期差异：**

Replace one of the memset nodes with a kernel that fills the buffer (simulating a custom "bzero" kernel):

```python
fill_ptx = """
.visible .entry fill_zero(.param .u64 BUF, .param .u64 N_BYTES)
{
    .reg .u64 %rd<4>;
    .reg .u32 %r<4>;
    .reg .u8 %b0;
    ld.param.u64 %rd0, [BUF];
    ld.param.u64 %rd1, [N_BYTES];
    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %rd2, %r0;
    setp.ge.u64 %p0, %rd2, %rd1;
    @%p0 bra DONE;
    add.u64 %rd3, %rd0, %rd2;
    mov.u8 %b0, 0;
    st.global.u8 [%rd3], %b0;
    DONE: ret;
}
"""

g2 = Graph()
n0 = g2.add_kernel_node(ptx_src=fill_ptx, grid=(1,1,1), block=(32,1,1),
                         params={"BUF": buf, "N_BYTES": np.uint64(n * 4)},
                         kernel_name="fill_zero")
n1 = g2.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                         params={"A": A, "B": B, "OUT": OUT},
                         kernel_name="vec_add")
g2.add_dependency(n0, n1)

exec2 = g2.instantiate(cfg)
cycles2 = exec2.launch()
print(f"Kernel-fill + compute: {cycles2} cycles vs memset + compute: {cycles} cycles")
```

The kernel-fill approach will cost more cycles than the fixed-50 memset model because the simulator executes real PTX instructions. This illustrates why memset nodes exist as a dedicated type: the CUDA runtime dispatches a highly optimized DMA/memset engine that is fundamentally faster than a generic shader core kernel for large buffer clears.

## 真机对照

On real NVIDIA hardware, memset nodes use `cudaGraphAddMemsetNode`:

```c
CUDA_MEMSET_NODE_PARAMS memsetParams = {
    .dst      = devPtr,         // device pointer
    .pitch    = width * sizeof(float),
    .value    = 0,              // byte value to fill
    .elementSize = 1,           // 1/2/4 bytes per element
    .width    = width,
    .height   = height,
};

cudaGraphNode_t memsetNode;
cudaGraphAddMemsetNode(&memsetNode, graph,
                        /*deps=*/NULL, /*numDeps=*/0,
                        &memsetParams, ctx);
```

| Aspect | `gpusim` | CUDA runtime |
|---|---|---|
| **API** | `graph.add_memset_node(buf=arr, value=v, n_bytes=n)` | `cudaGraphAddMemsetNode(&node, graph, deps, ndeps, &params, ctx)` |
| **Buffer type** | `numpy.ndarray` | Device pointer (`CUdeviceptr`) |
| **Fill granularity** | Byte-level (`buf[:] = value`) | Configurable: 1/2/4 bytes per element |
| **Cycle model** | Fixed 50 cycles regardless of size | Actual DMA bandwidth (H100: ~3.3 TB/s HBM) |
| **Pitch / 2D** | Not modeled | Supports pitched 2D layouts for texture data |

The most important practical difference is the **fixed 50-cycle model** in gpusim vs. real bandwidth-dependent latency on hardware. On an H100 SXM5 clearing a 64 MB buffer, the actual memset time is roughly `64 MB / 3.3 TB/s ≈ 19 μs`. The 50-cycle constant is calibrated for small buffers in tutorial examples; for production workloads you would replace it with a bandwidth-based formula `n_bytes / hbm_bw_bytes_per_cycle`.

Chapters 51–53 complete Phase 13's graph API expansion. Together with Chapters 44–47 (Phase 11 core graphs) and Chapters 48–50 (Phase 12 NCCL), gpusim now covers the full CUDA programming model from PTX kernels through distributed collective operations and CUDA Graph-based replay.
