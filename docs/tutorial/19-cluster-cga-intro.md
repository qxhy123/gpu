# Chapter 19 — Hopper Cluster (CGA) 入门

## 从单 SM CTA 到 Cluster CGA 的演进

The simulator's Phase 4 chapters (16–17) showed how the GPU dispatches multiple independent CTAs across SMs — each CTA has its own private shared memory and communicates with others only through global memory (L2 or HBM). This model works well but leaves two efficiency gaps: cross-CTA data exchange must pass through L2 (high latency), and each CTA independently fetches the same tiles from global memory even when neighbour CTAs need the same data.

Hopper's **Cluster** architecture (also called **CGA**, Cooperative Grid Architecture) closes both gaps. A cluster is a group of CTAs that are co-scheduled onto a fixed set of SMs within a single GPC (Graphics Processing Cluster). Because the SMs share a common interconnect inside the GPC, cluster members can read each other's shared memory directly at smem bandwidth — without involving L2 at all. The simulator models this with two additions: a configurable `cluster_size` and a new address space, `shared::cluster`.

## DeviceConfig.cluster_size と cluster_rank

Two fields on the warp identify a thread's position in the cluster hierarchy:

- `cluster_id` — which cluster this CTA belongs to (analogous to `ctaid.x` across the grid).
- `cluster_rank` — this CTA's index within its cluster (0 to `cluster_size − 1`).

In PTX you read the rank with:

```ptx
getctarank.u32 %r_rank;
```

On the Python side, enable clustering by setting `DeviceConfig.cluster_size` before the run:

```python
from gpusim.config.loader import load_default
cfg = load_default()
cfg.cluster_size = 4        # 4 CTAs per cluster
res = gpusim.run(..., config=cfg)
```

The grid must be an exact multiple of `cluster_size`; otherwise the simulator raises a `ValueError` at launch time.

## 两阶段异步 Cluster Barrier

Inside a cluster, CTAs synchronise using a **two-phase async barrier** — not `bar.sync` (which only synchronises threads within one CTA):

```ptx
barrier.cluster.arrive;   // non-blocking: CTA announces it has reached the barrier
barrier.cluster.wait;     // blocking:     CTA waits until all cluster members have arrived
```

`barrier.cluster.arrive` is *non-blocking*: it records the arrival and lets the warp continue executing subsequent instructions. `barrier.cluster.wait` is *blocking*: it stalls the warp until every CTA in the cluster has issued its `arrive`. This separation allows a CTA to do useful work between `arrive` and `wait` (for example, issue a TMA copy for the next iteration while waiting for neighbours to catch up).

## mapa 和 ld/st.shared::cluster

The key hardware mechanism is the **cluster-scoped shared memory address space**. Within a cluster, each CTA can encode a *remote* CTA's smem pointer using `mapa.shared::cluster`:

```ptx
// Encode: (rank << 24) | smem_byte_offset
// rank=0, offset=0 -> address that refers to CTA 0's smem[0]
mov.u64 %rd_offset, 0;
mov.u32 %r_rank, 0;
mapa.shared::cluster %rd_remote, %rd_offset, %r_rank;
```

The resulting `%rd_remote` is an opaque 64-bit pointer in the `shared::cluster` address space. Dereferencing it with `ld.shared::cluster` performs an actual load from the named remote CTA's shared memory:

```ptx
ld.shared::cluster.f32 %f0, [%rd_remote];
```

Stores work symmetrically with `st.shared::cluster`. The transfer travels over the intra-GPC interconnect and does not touch L2 or HBM.

## 走通 cluster_basic

The `examples/cluster_basic/kernel.ptx` is the simplest possible cluster demo: each CTA writes its `ctaid.x` to global memory, then executes a cluster barrier.

```bash
python examples/cluster_basic/run.py
```

Expected output:

```
cluster_basic: cycles=420
  out = [0, 1]
```

The kernel launches a 2-CTA grid with `cluster_size=2`. CTA 0 and CTA 1 are assigned to the same cluster. Thread 0 of each CTA writes its CTA index to `OUT[ctaid.x]`, then both CTAs rendezvous at `barrier.cluster.arrive` / `barrier.cluster.wait`. The barrier ensures both writes are visible before the kernel returns.

## 看模拟器

Open `report.html` at section **§19 Cluster dispatch + barrier timeline**. The Perfetto trace shows:

- Two CTAs dispatched simultaneously to adjacent SMs (they share a cluster slot).
- The `barrier.cluster.arrive` events on each SM appear at slightly different cycles — the non-blocking nature means each CTA posts its arrival independently.
- The `barrier.cluster.wait` stall shows as a `CLUSTER_BARRIER_WAIT` event on the SM that arrived first; the second SM's wait resolves immediately because the first has already posted.

To explore cluster sizing, change `cfg.cluster_size = 4` in `run.py` (and set `grid=(4,1,1)` to match). Rerun and observe the barrier timeline: all four CTAs must post `arrive` before any `wait` resolves, so the stall on fast CTAs grows proportionally to dispatch jitter.

## 改一改

**Grid not divisible by cluster_size:** Set `cfg.cluster_size = 3` with `grid=(4,1,1)`. The simulator raises:

```
ValueError: grid size 4 is not divisible by cluster_size 3
```

This matches the real hardware constraint: a cluster must be fully occupied.

**Larger cluster_size → longer dispatch wait:** Set `cluster_size = 8` with `grid=(8,1,1)`. In the HTML trace, the `CLUSTER_BARRIER_WAIT` stall on the first-arriving CTA now spans more cycles because all 8 CTAs must post `arrive` before any `wait` unblocks. Large clusters amplify launch latency; production code typically uses `cluster_size ∈ {1, 2, 4, 8}`.

## 真机对照

On a real H100, clusters must fit within a single **GPC** (Graphics Processing Cluster), which contains up to 9 SMs. That is the hard upper bound on `cluster_size`. The simulator simplifies this constraint: it allows any `cluster_size` that divides the grid evenly and fits in the configured `n_sm`, without modelling GPC topology. The intra-GPC interconnect in real hardware sustains smem-to-smem bandwidth comparable to local smem access (measured at ~10 TB/s aggregate per GPC on H100), which is why cluster-based data sharing dramatically outperforms L2-routed exchanges.
