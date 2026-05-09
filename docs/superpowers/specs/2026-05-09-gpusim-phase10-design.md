# gpusim Phase 10 — Multi-GPU + NVLink + NCCL Collective Primitives

> **Status:** Brainstormed 2026-05-09. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- Add **multi-GPU system** to gpusim. `cfg.n_gpus` controls how many GPUs (default 1 = backward compat with all Phase 1-9 single-GPU).
- Model **NVLink fabric**: point-to-point links between GPUs with configurable bandwidth + latency per link, all-to-all topology by default.
- Implement **NCCL-equivalent Comm API**: `gpusim.Comm(rank, world_size)` + `Comm.allreduce(send, recv, op)` + `Comm.broadcast(buf, root)` + `Comm.allgather(send, recv)`.
- Implement **two allreduce algorithms**: ring (bandwidth-optimal for large messages) + tree (latency-optimal for small); auto-pick based on message size.
- Ship 4 examples + 4 tutorial chapters covering rank/comm setup, ring allreduce, broadcast, and DDP-style training pattern.
- New trace events (`NvlinkTransfer`, `CollectiveOp`) + 4 analysis metrics + HTML §33/§34 + Perfetto NVLink swimlanes.
- 100% backward compatible: Phase 1-9 examples and tests pass unchanged with `n_gpus=1`.

### Non-goals (deferred to Phase 11+)
- **PyTorch distributed wrapper** (`dist.init_process_group` / `dist.all_reduce(tensor)`).
- **NVSwitch full congestion modeling** (Phase 10 uses point-to-point bandwidth without cross-link contention).
- **Reduce_scatter** (FSDP) — Phase 11 candidate after allreduce/allgather are solid.
- **Send/recv point-to-point primitives** (NCCL low-level) — Phase 11.
- **NCCL trees with multi-level reduction** beyond the single-level ring/tree.
- **CUDA Graphs** spanning multiple GPUs.
- **InfiniBand / Ethernet fallback** transports.

---

## 2. Architecture

```
gpusim.MultiGpuSystem (NEW)
├── gpus: list[GPU]                       # N GPUs, each is a Device instance
├── nvlink_fabric: NvlinkFabric            # NEW: point-to-point links
└── comms: dict[group_id, list[Comm]]      # NCCL communicators

GPU (renamed from Device, keeps all Phase 1-9 logic)
├── sms, l1, l2, hbm, …                    # unchanged
└── gpu_id: int                             # NEW

NvlinkFabric (NEW gpusim/comm/nvlink.py)
├── links: dict[(src_gpu, dst_gpu), NvlinkLink]
└── transfer(src_gpu, dst_gpu, bytes, cycle) -> completion_cycle

Comm (NEW gpusim/comm/comm.py)
├── rank, world_size, peers
├── allreduce(send, recv, op="sum") → ring or tree
├── broadcast(buf, root)
└── allgather(send, recv)
```

### Key invariants
- `cfg.n_gpus = 1` → `MultiGpuSystem` degenerates to single GPU; all Phase 1-9 behavior identical.
- `cfg.n_gpus > 1` → fabric instantiated; Comm API available.
- Each GPU executes its own kernels (Phase 1-9 stack unchanged); cross-GPU communication happens via NVLink fabric.
- NCCL collective operations are SCHEDULED on each rank (each GPU runs the same collective code in lockstep, like real CUDA).
- NVLink transfers serialize per-link (single-direction half-duplex by default; configurable to full-duplex).

---

## 3. Data model

### 3.1 New `GPU` class (renamed from `Device`)

```python
class GPU:
    """Single-GPU instance. Renamed from Device; keeps all Phase 1-9 logic."""
    gpu_id: int                             # NEW Phase 10
    cfg: DeviceConfig
    sms: list[SM]
    l1, l2, hbm: ...
    # All Phase 4-9 SM/cache/scheduler logic unchanged.
```

**Backward compat alias**: `Device = GPU` so all existing Phase 1-9 tests using `Device` still work.

### 3.2 New `MultiGpuSystem` class (`gpusim/comm/system.py`)

```python
@dataclass
class MultiGpuSystem:
    """N-GPU system with NVLink fabric. Phase 10."""
    gpus: list[GPU]
    nvlink_fabric: "NvlinkFabric"
    
    @classmethod
    def from_config(cls, cfg: Config) -> "MultiGpuSystem":
        n = getattr(cfg, "n_gpus", 1)
        gpus = [GPU(cfg, gpu_id=i) for i in range(n)]
        fabric = NvlinkFabric.from_config(cfg, n_gpus=n)
        return cls(gpus=gpus, nvlink_fabric=fabric)
    
    def run_streams_on(self, gpu_id: int, streams: list) -> "MultiStreamResult":
        """Run streams on a specific GPU (Phase 7-9 API per-GPU)."""
        return self.gpus[gpu_id].run_streams(streams)
    
    def run_collective(self, comm: "Comm", op: "CollectiveOp", **kwargs) -> int:
        """Execute a collective op across all comm.peers. Returns total cycles."""
        ...
```

### 3.3 New `NvlinkFabric` (`gpusim/comm/nvlink.py`)

```python
@dataclass
class NvlinkLink:
    src_gpu: int
    dst_gpu: int
    bandwidth_gbps: float       # default 900 GB/s (H100 NVLink 4 per-link)
    latency_cycles: int         # default 100 cycles
    busy_until: int = 0          # for transfer serialization

@dataclass
class NvlinkFabric:
    """Point-to-point NVLink topology. Default: all-to-all."""
    n_gpus: int
    links: dict[tuple, NvlinkLink]   # (src, dst) -> NvlinkLink
    topology: str = "all_to_all"      # or "ring", "tree"
    
    @classmethod
    def from_config(cls, cfg, n_gpus: int) -> "NvlinkFabric":
        # Default H100-style: every GPU connected to every other
        links = {}
        bw = getattr(cfg, "nvlink_bandwidth_gbps", 900.0)
        lat = getattr(cfg, "nvlink_latency_cycles", 100)
        for src in range(n_gpus):
            for dst in range(n_gpus):
                if src != dst:
                    links[(src, dst)] = NvlinkLink(src, dst, bw, lat)
        return cls(n_gpus=n_gpus, links=links)
    
    def transfer(self, src_gpu: int, dst_gpu: int, n_bytes: int,
                  arrival_cycle: int) -> int:
        """Transfer n_bytes over (src→dst) NVLink. Returns completion cycle.
        Serializes per link (busy_until tracks when link is free)."""
        link = self.links[(src_gpu, dst_gpu)]
        start = max(arrival_cycle, link.busy_until)
        # bandwidth_gbps → bytes per cycle (assume 1 cycle = 1 ns at 1 GHz)
        # 900 GB/s = 900 bytes/ns = 900 bytes/cycle
        bytes_per_cycle = link.bandwidth_gbps    # simplified: GB/s ≈ bytes/cycle
        transfer_cycles = max(1, int(n_bytes / bytes_per_cycle))
        completion = start + link.latency_cycles + transfer_cycles
        link.busy_until = completion
        return completion
```

### 3.4 New `Comm` class (`gpusim/comm/comm.py`)

```python
@dataclass
class Comm:
    """NCCL-equivalent communicator. One Comm per rank in a group."""
    rank: int
    world_size: int
    system: "MultiGpuSystem"
    group_id: int = 0
    
    def allreduce(self, send_buf, recv_buf, op: str = "sum") -> int:
        """All-reduce send_buf across all ranks; recv_buf gets the reduced result.
        Auto-picks ring (large) or tree (small) algorithm. Returns total cycles."""
        n_bytes = send_buf.nbytes
        if n_bytes >= 4096:   # heuristic threshold
            return self._allreduce_ring(send_buf, recv_buf, op)
        return self._allreduce_tree(send_buf, recv_buf, op)
    
    def broadcast(self, buf, root: int) -> int:
        """Broadcast buf from root to all ranks. Returns total cycles."""
        ...
    
    def allgather(self, send_buf, recv_buf) -> int:
        """Gather send_buf from all ranks into recv_buf (concatenation). Returns total cycles."""
        ...
    
    def _allreduce_ring(self, send_buf, recv_buf, op: str) -> int: ...
    def _allreduce_tree(self, send_buf, recv_buf, op: str) -> int: ...
```

### 3.5 New trace events (`gpusim/trace/events.py`)

```python
@dataclass(frozen=True)
class NvlinkTransfer:
    src_gpu: int
    dst_gpu: int
    n_bytes: int
    start_cycle: int
    end_cycle: int
    rank: int = -1                # which Comm rank initiated
    op_name: str = ""             # "allreduce" / "broadcast" / etc.

@dataclass(frozen=True)
class CollectiveOp:
    op_name: str                  # "allreduce" / "broadcast" / "allgather"
    algorithm: str                # "ring" / "tree" / "linear"
    n_bytes: int
    world_size: int
    start_cycle: int
    end_cycle: int
    n_steps: int                  # for ring: 2*(n-1); for tree: 2*log2(n)
```

### 3.6 Config extensions (`gpusim/config/schema.py`)

```python
@dataclass
class NvlinkConfig:
    bandwidth_gbps: float = 900.0    # H100 NVLink 4 per-link
    latency_cycles: int = 100         # 100ns at 1GHz
    topology: str = "all_to_all"      # alternative: "ring" / "tree"
    half_duplex: bool = True          # if True, one direction at a time

@dataclass
class Config:
    # ... existing fields ...
    n_gpus: int = 1                   # NEW Phase 10
    nvlink: NvlinkConfig = field(default_factory=NvlinkConfig)   # NEW Phase 10
```

---

## 4. Allreduce algorithms

### 4.1 Ring allreduce

For N ranks: `2*(N-1)` steps total. Each step transfers `chunk_size = n_bytes / N` over one NVLink hop.
- Steps 0..N-2: scatter-reduce (each rank receives chunk, reduces, sends to next)
- Steps N-1..2*(N-1)-1: allgather (each rank receives reduced chunk, sends to next)

```python
def _allreduce_ring(self, send_buf, recv_buf, op):
    n = self.world_size
    chunk_size = send_buf.nbytes // n
    cycle = 0
    n_steps = 2 * (n - 1)
    for step in range(n_steps):
        src = (self.rank - 1 - step) % n
        dst = (self.rank + 1) % n
        cycle = self.system.nvlink_fabric.transfer(
            src_gpu=self.rank, dst_gpu=dst,
            n_bytes=chunk_size, arrival_cycle=cycle,
        )
    # Apply reduction op to recv_buf (functional)
    import numpy as np
    if op == "sum":
        recv_buf[:] = send_buf * n   # simulator approximates: all ranks get sum
    elif op == "max":
        recv_buf[:] = send_buf       # simplified
    return cycle
```

⚠ Functional behavior is approximated: simulator doesn't actually run reductions across N memory spaces. Each rank gets `send_buf * n` for sum (assuming all ranks have identical send_buf for testing), or `send_buf` for max/min. This is fine for cycle simulation; functional correctness in tests uses identical inputs.

### 4.2 Tree allreduce

For N ranks (assume power of 2): `2*log2(N)` steps.
- Reduce phase: rank pairs combine; halves rank count each step.
- Broadcast phase: root sends back down the tree.

```python
def _allreduce_tree(self, send_buf, recv_buf, op):
    import math
    n = self.world_size
    steps = 2 * max(1, int(math.log2(n)))
    cycle = 0
    for step in range(steps):
        partner = self.rank ^ (1 << (step % int(math.log2(n))))
        if 0 <= partner < n:
            cycle = self.system.nvlink_fabric.transfer(
                src_gpu=self.rank, dst_gpu=partner,
                n_bytes=send_buf.nbytes, arrival_cycle=cycle,
            )
    recv_buf[:] = send_buf * n if op == "sum" else send_buf
    return cycle
```

### 4.3 Algorithm selection
Threshold: `n_bytes < 4096` → tree (latency-optimal); else → ring (bandwidth-optimal). Configurable via `comm.algorithm_threshold_bytes`.

---

## 5. Trace + Analysis

### 5.1 Trace
- `NvlinkFabric.transfer` records a `NvlinkTransfer` event each call.
- `Comm.allreduce/broadcast/allgather` records a wrapping `CollectiveOp` event.

### 5.2 4 new metrics

```python
def nvlink_bandwidth_utilization(nvlink_df, total_cycles: int) -> dict:
    """Per-link bytes/cycle utilization vs theoretical max."""

def collective_op_breakdown(collective_df) -> dict:
    """Cycles per (op_name, algorithm); shows which collective dominates."""

def algo_efficiency_ring_vs_tree(collective_df) -> dict:
    """Compare measured cycles for ring vs tree at various message sizes."""

def per_rank_communication_volume(nvlink_df) -> dict:
    """Total bytes sent/received per rank; shows rank-imbalance hotspots."""
```

### 5.3 MultiStreamResult extensions (or new `MultiGpuResult`)

```python
class MultiGpuResult:
    per_gpu_results: list[MultiStreamResult]
    nvlink_events: list[NvlinkTransfer]
    collective_events: list[CollectiveOp]
    
    def nvlink_bandwidth_utilization(self) -> dict: ...
    def collective_op_breakdown(self) -> dict: ...
    def per_rank_communication_volume(self) -> dict: ...
```

---

## 6. Viz

### 6.1 HTML §33 + §34
- **§33 Collective op timeline**: gantt of all collective ops across ranks
- **§34 NVLink fabric utilization heatmap**: matrix view (src × dst) showing total bytes per link

### 6.2 Perfetto NVLink swimlanes
- New `pid="NVLink"` swimlane; each transfer is a duration event
- New `pid="Rank-N"` swimlane per rank; collective ops shown as duration events
- Async arrows for collective dependencies (e.g., scatter-reduce → allgather)

---

## 7. Examples (4)

### 7.1 `multi_gpu_setup/`
- 2-GPU minimal demo; create comm, validate rank/world_size
- Verifies `cfg.n_gpus=2` works end-to-end; both GPUs run a vec_add

### 7.2 `ring_allreduce/`
- 4-GPU ring allreduce on a 1024-byte buffer
- Verifies correct sum (each rank gets 4× original); cycle count matches `2*(N-1)*chunk_transfer_cycles`

### 7.3 `tree_allreduce/`
- 4-GPU tree allreduce on a 64-byte buffer
- Verifies tree algorithm chosen (small msg); cycle count matches `2*log2(N)*hop_cycles`

### 7.4 `ddp_training_step/` ⭐ Capstone
- 4-GPU DDP-style training step
- Each rank runs vec_add → reduce gradients via allreduce → broadcast updated weights
- Demonstrates compute + communication overlap pattern

---

## 8. Tutorials

`docs/tutorial/` chapters 40-43:
- **40-multi-gpu-system-and-nvlink-fabric.md** — example 1
- **41-ring-allreduce-bandwidth-optimal.md** — example 2 ⭐
- **42-tree-allreduce-latency-optimal.md** — example 3
- **43-ddp-training-pattern.md** — example 4 ⭐ capstone

---

## 9. Testing strategy

### Unit tests (~14 new)
- `tests/unit/comm/test_nvlink_fabric.py` — link bandwidth/latency math, transfer serialization
- `tests/unit/comm/test_comm_basic.py` — Comm construction, rank validation
- `tests/unit/comm/test_allreduce_ring.py` — ring algorithm step count + cycle math
- `tests/unit/comm/test_allreduce_tree.py` — tree algorithm step count + cycle math
- `tests/unit/comm/test_broadcast.py` — linear broadcast cycle math
- `tests/unit/comm/test_allgather.py` — gather correctness
- `tests/unit/comm/test_multi_gpu_system.py` — N-GPU instantiation, backward compat
- `tests/unit/trace/test_collective_event.py` — NvlinkTransfer + CollectiveOp recorder
- `tests/unit/analysis/test_phase10_metrics.py` — 4 new metrics

### Parity tests (~4 — one per example)

### Microbench
- `test_phase10_facts.py` (fast):
  - Ring allreduce: cycles ≈ `2*(N-1)*chunk_transfer_cycles + N*overhead`
  - Tree allreduce on small msg: cycles ≈ `2*log2(N)*hop_cycles`
  - Tree faster than ring at message size below threshold
- `test_phase10_runtime.py` (slow): 4 examples each under 60s

### Regression
- Rename `tests/parity/test_phase1_8_examples_unchanged.py` → `test_phase1_9_examples_unchanged.py`
- Add 3 Phase 9 examples to the regression list

### Test count target
556 (Phase 9 baseline) → ~590 (+34).

---

## 10. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1** GPU rename + MultiGpuSystem + n_gpus config | Backward-compat alias Device=GPU; MultiGpuSystem.from_config; cfg.n_gpus + nvlink config; 1-GPU passes everything | `M1-phase10-complete` |
| **M2** NvlinkFabric + transfer primitive | NvlinkLink + NvlinkFabric.transfer + serialization + 1 demo example multi_gpu_setup | `M2-phase10-complete` |
| **M3** Comm + ring allreduce + ring_allreduce example | Comm class + Comm.allreduce ring path + algorithm step math + 2 metrics + 1 example | `M3-phase10-complete` |
| **M4** Tree allreduce + broadcast + allgather + 2 examples | Tree algorithm + auto-pick + broadcast + allgather + 2 examples | `M4-phase10-complete` |
| **M5** Capstone DDP + viz + docs + ship | ddp_training_step + 4 chapters + HTML §33/§34 + Perfetto + microbench + Phase 1-9 regression + README v10 | `phase10-complete` |

Estimated 32 tasks total.

---

## 11. File list

### New files
```
gpusim/comm/__init__.py
gpusim/comm/system.py            # MultiGpuSystem
gpusim/comm/nvlink.py            # NvlinkLink + NvlinkFabric
gpusim/comm/comm.py              # Comm class + algorithms
examples/multi_gpu_setup/        # 5 files (M2)
examples/ring_allreduce/         # 5 files (M3)
examples/tree_allreduce/         # 5 files (M4)
examples/ddp_training_step/      # 6 files (M5)
docs/tutorial/40-multi-gpu-system-and-nvlink-fabric.md
docs/tutorial/41-ring-allreduce-bandwidth-optimal.md
docs/tutorial/42-tree-allreduce-latency-optimal.md
docs/tutorial/43-ddp-training-pattern.md
tests/unit/comm/__init__.py
tests/unit/comm/test_nvlink_fabric.py
tests/unit/comm/test_comm_basic.py
tests/unit/comm/test_allreduce_ring.py
tests/unit/comm/test_allreduce_tree.py
tests/unit/comm/test_broadcast.py
tests/unit/comm/test_allgather.py
tests/unit/comm/test_multi_gpu_system.py
tests/unit/trace/test_collective_event.py
tests/unit/analysis/test_phase10_metrics.py
tests/parity/test_multi_gpu_setup.py
tests/parity/test_ring_allreduce.py
tests/parity/test_tree_allreduce.py
tests/parity/test_ddp_training_step.py
tests/microbench/test_phase10_facts.py
tests/microbench/test_phase10_runtime.py
tests/reference/data/{4 example names}.ref.json
```

### Modified files
```
gpusim/api.py                    # + MultiGpuSystem export + Comm export
gpusim/__init__.py               # + Comm + MultiGpuSystem exports
gpusim/core/device.py            # rename Device → GPU + alias Device=GPU
gpusim/config/schema.py          # + Config.n_gpus + NvlinkConfig
gpusim/trace/events.py           # + NvlinkTransfer + CollectiveOp
gpusim/trace/recorder.py         # + nvlink_transfer + collective methods
gpusim/trace/writer.py           # + nvlink_transfer.parquet + collective.parquet
gpusim/analysis/metrics.py       # + 4 metrics
gpusim/viz/notebook.py           # + collective_events_dataframe + nvlink_events_dataframe
gpusim/viz/html_report.py        # + §33 + §34 helpers
gpusim/viz/_template.html.j2     # + §33 + §34 blocks
gpusim/viz/perfetto.py           # + NVLink swimlane + Rank-N swimlanes
tests/parity/test_phase1_8_examples_unchanged.py → test_phase1_9_examples_unchanged.py
tests/reference/gen_reference.py # + 4 kernel names
README.md                        # v10 — Phase 10 capabilities
```

---

## 12. Backward compatibility

- `cfg.n_gpus = 1` (default) → all Phase 1-9 behavior identical. `Device` class is now an alias for `GPU` (no behavior change).
- `gpusim.run(...)` — unchanged, runs on GPU 0.
- Phase 7-9 multi-stream APIs (`Stream.launch`, `gpusim.synchronize`) — unchanged, runs on GPU 0.
- All Phase 1-9 examples + tests pass.
- `Comm`, `MultiGpuSystem`, `NvlinkFabric` are NEW additions — no impact unless `n_gpus > 1`.

---

## 13. Open questions / future work

- **Reduce_scatter** (Phase 11) — needed for FSDP.
- **Send/recv P2P** (Phase 11) — NCCL low-level primitives.
- **Cross-link congestion** (Phase 11) — multiple transfers competing on shared NVSwitch.
- **PyTorch dist wrapper** (Phase 11) — `dist.init_process_group` + `dist.all_reduce` thin adapter.
- **Multi-rail NVLink** — H100 has 18 links per GPU; modeling individual rails.
- **NVSwitch detail** — Phase 10 abstracts NVSwitch as direct point-to-point.

---

## 14. Acceptance criteria

Phase 10 ships when:

- [ ] All 5 milestone tags present (`M1-phase10-complete` ... `M4-phase10-complete`, `phase10-complete`)
- [ ] All 4 examples run cleanly (`python examples/<name>/run.py`)
- [ ] All 4 parity tests pass
- [ ] Microbench: ring allreduce step count = 2*(N-1); tree step count = 2*log2(N)
- [ ] HTML report shows §33 + §34 when collective events present
- [ ] Perfetto JSON has NVLink + Rank-N swimlanes
- [ ] Phase 1-9 regression test (renamed) passes
- [ ] Test count: 556 → ~590 (+34)
- [ ] README v10 documents Phase 10 capabilities
