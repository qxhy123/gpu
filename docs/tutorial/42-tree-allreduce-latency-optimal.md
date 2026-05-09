# Chapter 42 — Tree Allreduce: Latency-Optimal Collective

## The Latency vs. Bandwidth Trade-off

Ring allreduce is bandwidth-optimal: it transfers the minimum number of bytes per link to complete the reduction. But it pays for that bandwidth efficiency with latency: it takes 2*(N-1) sequential transfer steps. For a 4-GPU ring that is 6 steps; for an 8-GPU ring it is 14.

For small messages, transfer time is dominated by link latency (the fixed `latency_cycles` per hop) rather than bandwidth. In that regime, fewer steps always win. The binary-tree algorithm reduces step count to 2*log2(N) — just 4 steps for N=4 and 6 for N=8 — at the cost of not using all links equally.

This is the fundamental algorithmic trade-off in collective communications: ring for large messages, tree for small messages.

## The Tree Algorithm: 2*log2(N) Steps

Tree allreduce proceeds in two phases using a butterfly (recursive doubling) pattern:

**Reduce phase** (log2(N) steps): In step k, each rank XOR-partners with rank `rank ^ (1 << k)`. One member of each pair sends its partial sum to the other. After log2(N) steps, one rank (the root, rank 0) holds the fully reduced value.

**Broadcast phase** (log2(N) more steps): The root broadcasts back down the tree using the same partner pattern in reverse. After log2(N) more steps, every rank holds the complete reduced result.

For N=4 and rank=0:
- Step 0 (k=0): partner = 0 ^ 1 = 1
- Step 1 (k=1): partner = 0 ^ 2 = 2
- Broadcast step 0: partner = 0 ^ 1 = 1
- Broadcast step 1: partner = 0 ^ 2 = 2

Total: 4 transfers, matching 2*log2(4) = 4.

The simulator's `Comm._allreduce_tree` implements this as two sequential loops:

```python
log_n = max(1, int(math.log2(n)))
cycle = 0
for step in range(log_n):           # reduce phase
    partner = self.rank ^ (1 << step)
    if 0 <= partner < n:
        cycle = self.system.nvlink_fabric.transfer(
            src_gpu=self.rank, dst_gpu=partner,
            n_bytes=send_buf.nbytes, arrival_cycle=cycle,
        )
for step in range(log_n):           # broadcast phase
    partner = self.rank ^ (1 << step)
    if 0 <= partner < n:
        cycle = self.system.nvlink_fabric.transfer(
            src_gpu=self.rank, dst_gpu=partner,
            n_bytes=send_buf.nbytes, arrival_cycle=cycle,
        )
```

## The 4096-Byte Threshold

`Comm.allreduce` automatically selects the algorithm:

```python
def allreduce(self, send_buf, recv_buf, op="sum") -> int:
    n_bytes = send_buf.nbytes
    threshold = 4096
    algorithm = "ring" if n_bytes >= threshold else "tree"
    ...
```

Messages smaller than 4096 bytes use tree; 4096 bytes and above use ring. This threshold is hardcoded in the simulator to demonstrate both paths conveniently. NCCL uses 256 KB in practice (Chapter 41).

## Running the Tree Demo

```bash
python examples/tree_allreduce/run.py
```

The demo uses a 16-element float32 array (64 bytes — well below 4096):

```python
cfg = load_default()
cfg.n_gpus = 4
sys = MultiGpuSystem.from_config(cfg)
rec = Recorder()
comm = Comm(rank=0, world_size=4, system=sys)
comm._recorder = rec
send = np.full(16, 1.0, dtype=np.float32)  # 64 bytes — tree path
recv = np.zeros(16, dtype=np.float32)
cycles = comm.allreduce(send, recv, op="sum")
print(f"Algorithm chosen: {rec.collective_events[-1].algorithm}")  # "tree"
```

## 看模拟器

**验证 2*log2(N) 步数和算法自动选择：**

Attach a recorder and compare step counts for ring vs. tree on the same 4-GPU system:

```python
from gpusim.trace.recorder import Recorder
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default
import numpy as np

cfg = load_default()
cfg.n_gpus = 4
sys = MultiGpuSystem.from_config(cfg)
rec = Recorder()
comm = Comm(rank=0, world_size=4, system=sys)
comm._recorder = rec

# Small message — tree
send_small = np.full(8, 1.0, dtype=np.float32)   # 32 bytes
recv_small = np.zeros(8, dtype=np.float32)
comm.allreduce(send_small, recv_small, op="sum")
ev_tree = rec.collective_events[-1]
print(f"Tree: algo={ev_tree.algorithm}, steps={ev_tree.n_steps}")  # tree, 4

# Large message — ring
send_large = np.full(1024, 1.0, dtype=np.float32) # 4096 bytes
recv_large = np.zeros(1024, dtype=np.float32)
comm.allreduce(send_large, recv_large, op="sum")
ev_ring = rec.collective_events[-1]
print(f"Ring: algo={ev_ring.algorithm}, steps={ev_ring.n_steps}")  # ring, 6
```

The `collective_events[-1].algorithm` field records which branch `allreduce` took. Combine this with `nvlink_transfer_events` to count actual fabric operations and verify they match the theoretical step counts.

## 改一改

**测量延迟优势：在小消息上对比 ring 和 tree：**

Force both algorithms on the same small message by calling the private methods directly:

```python
cfg = load_default()
cfg.n_gpus = 4
sys = MultiGpuSystem.from_config(cfg)
comm = Comm(rank=0, world_size=4, system=sys)
send = np.full(8, 1.0, dtype=np.float32)  # 32 bytes
recv = np.zeros(8, dtype=np.float32)

cycles_tree = comm._allreduce_tree(send, recv, op="sum")
# Reset link busy_until
for link in sys.nvlink_fabric.links.values():
    link.busy_until = 0
cycles_ring = comm._allreduce_ring(send, recv, op="sum")

print(f"Tree: {cycles_tree} cycles (2*log2(4) = 4 steps)")
print(f"Ring: {cycles_ring} cycles (2*(4-1) = 6 steps)")
print(f"Tree advantage: {cycles_ring - cycles_tree} cycles")
```

For small messages the latency difference is dominated by the step count difference (4 vs. 6) multiplied by the fixed 100-cycle link latency. The tree saves 2 × 100 = 200 cycles on a 32-byte message — a 33 % reduction. For a 4096-byte message, transfer time dwarfs latency and ring becomes competitive again due to better bandwidth balance.

## 真机对照

NCCL implements tree allreduce using a double binary tree (DBT) algorithm for messages below 256 KB. The DBT variant ensures that all ranks are active in both tree phases (no idle ranks), improving utilization compared to a naive single-tree reduction.

| Property | Simulator | NCCL (H100 DGX) |
|---|---|---|
| **Algorithm trigger** | `n_bytes < 4096` | `n_bytes < 256 KB` (protocol-tuned) |
| **Step count** | 2*log2(N) | 2*log2(N) for DBT reduce+broadcast |
| **Step pattern** | Butterfly XOR partners | Same, with double-tree load balancing |
| **BW per step** | Full tensor size | Full tensor size (no chunking) |

At 8 GPUs, tree saves 14-6 = 8 steps compared to ring — a 57 % step count reduction. NCCL's measured crossover on NVLink 4 hardware is roughly at 256 KB where the bandwidth savings of ring begin to outweigh the latency savings of tree. The simulator's 4096-byte crossover is ~60× smaller, tuned so that both algorithms appear in typical demo workloads without requiring megabyte-scale buffers.

The latency-bandwidth trade-off is fundamental to all network collective algorithms, not just GPU collectives. The same principle applies in MPI (Allreduce uses Rabenseifner's algorithm at high counts) and in Ethernet-based RDMA clusters where tree is favored for control messages and ring or recursive halving-doubling for gradient tensors.
