# Chapter 57 — Persistent Pipeline: Producer-Consumer ⭐ Capstone

## The Producer-Consumer Pattern

The most powerful application of persistent kernels is the **producer-consumer pipeline**: one persistent kernel produces data into a buffer, a second persistent kernel consumes it. On real CUDA hardware this pattern appears in inference servers (batch assembly → inference → postprocessing) and training pipelines (data preprocessing → forward pass → gradient computation).

In gpusim, both stages are modeled as separate `PersistentKernel` instances sharing a `WorkQueue`. The producer fills output buffers and places them on the consumer's queue; the consumer drains that queue.

Phase 14 ships this pattern in the `persistent_pipeline` example — the capstone of Chapters 54–57.

## The persistent_pipeline Demo

```python
import numpy as np, pathlib
from gpusim.persistent.queue import WorkQueue
from gpusim.persistent.kernel import PersistentKernel
from gpusim.config.loader import load_default

cfg = load_default()
ptx = pathlib.Path("examples/persistent_pipeline/kernel.ptx").read_text()

# Stage 1: producer — fills 3 output buffers with value 1
producer_q = WorkQueue()
out_bufs = []
for _ in range(3):
    ob = np.zeros(32, dtype=np.uint32)
    out_bufs.append(ob)
    producer_q.push({"OUT": ob})
producer_q.stop()

producer = PersistentKernel(
    ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
    params_template={}, work_queue=producer_q,
    kernel_name="producer",
)
producer_results = producer.start(cfg)

print(f"Producer processed {len(producer_results)} items")   # 3
for ob in out_bufs:
    print(ob.sum())   # 32 — each of 32 threads wrote 1
```

Run it:

```bash
python examples/persistent_pipeline/run.py
```

In the simulation the producer stage runs to completion before the consumer stage begins (sequential scheduling matches the `PersistentKernel.start()` blocking semantics). On real CUDA hardware both stages overlap on independent SM partitions.

## 看模拟器

**用两个 Recorder 分别追踪 producer 和 consumer：**

```python
from gpusim.trace.recorder import Recorder

rec_prod = Recorder()
rec_cons = Recorder()

# Producer stage
producer_q = WorkQueue()
for _ in range(3):
    producer_q.push({"OUT": np.zeros(32, dtype=np.uint32)})
producer_q.stop()
producer = PersistentKernel(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                             params_template={}, work_queue=producer_q,
                             kernel_name="producer")
producer.start(cfg, recorder=rec_prod)

# Consumer stage (uses producer's output buffers as input)
consumer_q = WorkQueue()
for ob in out_bufs:
    consumer_q.push({"OUT": ob})   # re-process producer outputs
consumer_q.stop()
consumer = PersistentKernel(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                             params_template={}, work_queue=consumer_q,
                             kernel_name="consumer")
consumer.start(cfg, recorder=rec_cons)

# Both recorders have is_persistent=True events
prod_events = [e for e in rec_prod.kernel_launch_events if e.is_persistent]
cons_events = [e for e in rec_cons.kernel_launch_events if e.is_persistent]
print(f"Producer launches: {len(prod_events)}")   # 3
print(f"Consumer launches: {len(cons_events)}")   # 3

# All events have correct kernel_name
assert all(e.kernel_name == "producer" for e in prod_events)
assert all(e.kernel_name == "consumer" for e in cons_events)
```

**用 persistent_kernel_throughput 对比两阶段吞吐：**

```python
import pandas as pd
from gpusim.analysis.metrics import persistent_kernel_throughput

def make_df(rec):
    return pd.DataFrame([
        {"is_persistent": e.is_persistent,
         "stream_id": i,
         "parent_kernel_id": e.parent_kernel_id}
        for i, e in enumerate(rec.kernel_launch_events)
    ])

prod_rate = persistent_kernel_throughput(make_df(rec_prod), total_cycles=3000)
cons_rate = persistent_kernel_throughput(make_df(rec_cons), total_cycles=3000)
print(f"Producer: {prod_rate:.2f} iters/1000 cycles")
print(f"Consumer: {cons_rate:.2f} iters/1000 cycles")
```

In a balanced pipeline both rates should be approximately equal. If the producer rate is lower (slower kernel), the consumer will be starved; if higher, the consumer queue will grow unbounded.

## 改一改

**扩展为三阶段流水线：**

Add a third "postprocessing" stage that reads the consumer's outputs:

```python
# Stage 3: post-process (double each element — requires a different PTX)
postproc_ptx = """
.visible .entry double_write(.param .u64 OUT) {
    .reg .u64 %rd<4>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r2, [%rd2];
    shl.b32 %r3, %r2, 1;        // value * 2
    st.global.u32 [%rd2], %r3;
    ret;
}
"""

# Reuse out_bufs from producer (already filled with 1)
postproc_q = WorkQueue()
for ob in out_bufs:
    postproc_q.push({"OUT": ob})
postproc_q.stop()

postproc = PersistentKernel(
    ptx_src=postproc_ptx, grid=(1,1,1), block=(32,1,1),
    params_template={}, work_queue=postproc_q,
    kernel_name="postproc",
)
postproc.start(cfg)

for ob in out_bufs:
    print(ob.sum())   # 64 — each 1 doubled to 2, sum = 32 × 2
```

**测试空 producer 直接 stop：**

```python
q = WorkQueue()
q.stop()   # nothing pushed

pk = PersistentKernel(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                      params_template={}, work_queue=q)
r = pk.start(cfg)
assert r == []   # no items processed → empty results list
```

## 真机对照

The producer-consumer persistent kernel pattern is the foundation of NVIDIA's Triton Inference Server and TensorRT-LLM:

| Aspect | `gpusim` | Real CUDA inference server |
|---|---|---|
| **Producer** | `PersistentKernel` filling NumPy arrays | CUDA kernel reading from pinned host memory ring buffer |
| **Consumer** | `PersistentKernel` reading same arrays | Persistent inference kernel polling device-side queue |
| **Handoff** | Sequential: producer completes → consumer starts | Concurrent: producer fills queue; consumer polls atomically |
| **Queue** | `WorkQueue` (Python deque) | Lock-free SPSC/MPSC ring buffer (device global memory) |
| **Backpressure** | Immediate (all items pre-loaded) | Blocking push when queue full (producer stalls) |
| **Stage isolation** | Separate `PersistentKernel` instances | Separate persistent kernel grids on dedicated SM partitions |

The key practical difference: real persistent pipelines achieve zero-copy handoff between stages using shared device memory. In gpusim, the handoff is through NumPy arrays on the host — functionally equivalent but without the intra-device memory traffic modeling.

For real workloads, the persistent pipeline throughput is gated by the slowest stage (Amdahl's law for pipelines). On H100 hardware with NVLink, you can also pipeline across GPUs: producer on GPU 0, consumer on GPU 1, with NVLink for the handoff buffer. Combine Phase 14's persistent kernel API with Phase 10's NVLink fabric model to simulate multi-GPU persistent pipelines.

---

This chapter completes Phase 14. Together, Chapters 54–57 cover:
- **54**: `WorkQueue` + `PersistentKernel` basics (server pattern, 5 work items)
- **55**: Dynamic queue growth (push in batches before stop)
- **56**: Dynamic parallelism (parent triggers child via `device_launch`)
- **57**: Capstone producer-consumer pipeline ⭐

Phase 14 adds persistent kernels and dynamic parallelism to gpusim's simulation model, rounding out the CUDA programming model coverage from PTX kernels (Phase 1) through CUDA Graphs (Phase 11–13) and now persistent execution patterns (Phase 14).
