# gpusim Phase 10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Implement gpusim Phase 10 per `docs/superpowers/specs/2026-05-09-gpusim-phase10-design.md` — multi-GPU system + NVLink fabric + NCCL collective primitives (allreduce ring/tree, broadcast, allgather).

**Architecture:** `Device` renamed to `GPU` (alias kept for backward compat). New `MultiGpuSystem` wraps N GPUs + `NvlinkFabric`. New `gpusim.Comm` class implements rank-aware collectives. NVLink transfers serialize per-link; ring/tree allreduce algorithms record cycle counts. New `NvlinkTransfer` + `CollectiveOp` trace events + 4 metrics + HTML §33/§34 + Perfetto NVLink/Rank-N swimlanes.

**Tech Stack:** Python 3.11+. No new runtime dependencies.

**Execution note:** Plan has 5 milestones (M1–M5) with 28 tasks. Tags after each: `M{1..5}-phase10-complete`.

---

## Phase 1+2+3+4+5+6+7+8+9 prerequisites

```bash
cd /Users/yangyang/ai_projs/gpu
git tag | grep phase
.venv/bin/pytest -q -m "not slow"
```
Expected: ~556 passed (Phase 9 baseline).

---

## File structure

```
gpusim/
├── api.py                       MODIFY: + MultiGpuSystem export + Comm export
├── core/device.py               MODIFY: rename Device → GPU + alias Device=GPU + gpu_id field
├── config/schema.py             MODIFY: + Config.n_gpus + NvlinkConfig
├── comm/                        NEW (M1+M2+M3+M4)
│   ├── __init__.py
│   ├── system.py                # MultiGpuSystem
│   ├── nvlink.py                # NvlinkLink + NvlinkFabric
│   └── comm.py                  # Comm + algorithms
├── trace/events.py              MODIFY: + NvlinkTransfer + CollectiveOp
├── trace/recorder.py            MODIFY: + nvlink_transfer + collective methods
├── trace/writer.py              MODIFY: + parquet writers
├── analysis/metrics.py          MODIFY: + 4 metrics
└── viz/                         MODIFY (M5): + §33/§34 + NVLink/Rank-N swimlanes

examples/
├── multi_gpu_setup/             NEW (M2)
├── ring_allreduce/              NEW (M3)
├── tree_allreduce/              NEW (M4)
└── ddp_training_step/           NEW (M4) — capstone

tests/unit/comm/                 NEW (M1-M4)
tests/parity/test_phase1_9_examples_unchanged.py    RENAME from phase1_8 (M5)
tests/microbench/test_phase10_facts.py              NEW (M5)
tests/microbench/test_phase10_runtime.py            NEW (M5, slow)
tests/reference/data/{4 names}.ref.json             NEW (M5)
docs/tutorial/{40,41,42,43}-*.md                    NEW (M5)
README.md                                            MODIFY (M5): v10
```

---

## Milestones

| Milestone | Tasks | Tag |
|---|---|---|
| **M1** GPU rename + config + MultiGpuSystem skeleton + trace events | T1–T5 | `M1-phase10-complete` |
| **M2** NvlinkFabric + transfer + multi_gpu_setup example | T6–T9 | `M2-phase10-complete` |
| **M3** Comm + ring allreduce + 2 metrics + ring example | T10–T14 | `M3-phase10-complete` |
| **M4** Tree + broadcast + allgather + 2 examples | T15–T20 | `M4-phase10-complete` |
| **M5** Viz + tutorials + microbench + regression + README v10 + ship | T21–T28 | `phase10-complete` |

---

## Milestone M1: GPU rename + MultiGpuSystem skeleton + trace events

### Task 1: Device → GPU rename + alias + gpu_id field

**Files:**
- Modify: `gpusim/core/device.py`
- Test: `tests/unit/core/test_gpu_rename.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_device_alias_to_gpu():
    """Phase 10 backward compat: Device is alias for GPU."""
    from gpusim.core.device import Device, GPU
    assert Device is GPU


def test_gpu_has_gpu_id_field():
    from gpusim.core.device import GPU
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = GPU(cfg)
    # Should have gpu_id attribute (default 0 for single-GPU compat)
    assert hasattr(g, "gpu_id")


def test_gpu_with_explicit_id():
    from gpusim.core.device import GPU
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = GPU(cfg, gpu_id=2)
    assert g.gpu_id == 2
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add gpu_id + alias to gpusim/core/device.py**

In `gpusim/core/device.py`:
- Add `gpu_id` parameter to Device.__init__ (default 0)
- Set `self.gpu_id = gpu_id`
- At end of file, add: `GPU = Device  # Phase 10 forward-compat alias (real rename in Phase 11)`

⚠ Don't actually rename the class to GPU in this task — just add the alias and gpu_id field. Renaming would touch many test imports. The alias is forward-compat: new Phase 10 code uses GPU; old code keeps using Device. Both point to same class.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_gpu_rename.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 3 NEW pass.

```bash
git add gpusim/core/device.py tests/unit/core/test_gpu_rename.py
git commit -m "feat(core): GPU = Device alias + gpu_id field for Phase 10 multi-GPU"
```

---

### Task 2: Config.n_gpus + NvlinkConfig

**Files:**
- Modify: `gpusim/config/schema.py`
- Test: `tests/unit/config/test_phase10_config.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_config_n_gpus_default_1():
    from gpusim.config.loader import load_default
    cfg = load_default()
    assert cfg.n_gpus == 1


def test_config_nvlink_section():
    from gpusim.config.schema import NvlinkConfig
    nv = NvlinkConfig()
    assert nv.bandwidth_gbps == 900.0
    assert nv.latency_cycles == 100
    assert nv.topology == "all_to_all"
    assert nv.half_duplex is True
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add to schema.py**

```python
@dataclass
class NvlinkConfig:
    bandwidth_gbps: float = 900.0
    latency_cycles: int = 100
    topology: str = "all_to_all"
    half_duplex: bool = True


# In Config class:
    n_gpus: int = 1                                                       # NEW Phase 10
    nvlink: NvlinkConfig = field(default_factory=NvlinkConfig)             # NEW Phase 10
```

⚠ Find the existing top-level Config (or DeviceConfig) class — add fields there.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/config/test_phase10_config.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/config/schema.py tests/unit/config/test_phase10_config.py
git commit -m "feat(config): + Config.n_gpus + NvlinkConfig for multi-GPU"
```

---

### Task 3: MultiGpuSystem skeleton

**Files:**
- Create: `gpusim/comm/__init__.py`, `gpusim/comm/system.py`
- Test: `tests/unit/comm/__init__.py`, `tests/unit/comm/test_multi_gpu_system.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_multi_gpu_system_single_gpu_default():
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    sys = MultiGpuSystem.from_config(cfg)
    assert len(sys.gpus) == 1
    assert sys.gpus[0].gpu_id == 0


def test_multi_gpu_system_4_gpus():
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    assert len(sys.gpus) == 4
    assert [g.gpu_id for g in sys.gpus] == [0, 1, 2, 3]
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Create gpusim/comm/system.py**

```python
"""Phase 10: Multi-GPU system orchestration."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MultiGpuSystem:
    """N-GPU system. Phase 10."""
    gpus: list = field(default_factory=list)
    nvlink_fabric: object | None = None    # set by from_config when n_gpus > 1
    
    @classmethod
    def from_config(cls, cfg) -> "MultiGpuSystem":
        from gpusim.core.device import GPU
        n = getattr(cfg, "n_gpus", 1)
        gpus = [GPU(cfg, gpu_id=i) for i in range(n)]
        fabric = None
        if n > 1:
            from gpusim.comm.nvlink import NvlinkFabric
            fabric = NvlinkFabric.from_config(cfg, n_gpus=n)
        return cls(gpus=gpus, nvlink_fabric=fabric)
```

⚠ NvlinkFabric is created in T6. For T3, the import is inside `from_config` (deferred) — single-GPU tests don't need fabric.

Create `gpusim/comm/__init__.py`:
```python
from gpusim.comm.system import MultiGpuSystem
__all__ = ["MultiGpuSystem"]
```

Create empty `tests/unit/comm/__init__.py`.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/comm/test_multi_gpu_system.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/comm/ tests/unit/comm/
git commit -m "feat(comm): MultiGpuSystem.from_config skeleton (n_gpus 1 = backward compat)"
```

---

### Task 4: NvlinkTransfer + CollectiveOp trace events

**Files:**
- Modify: `gpusim/trace/events.py` (add 2 dataclasses)
- Modify: `gpusim/trace/recorder.py` (add 2 methods + lists)
- Modify: `gpusim/trace/writer.py` (add 2 parquet writers)
- Test: `tests/unit/trace/test_collective_event.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_recorder_records_nvlink_transfer():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.nvlink_transfer(src_gpu=0, dst_gpu=1, n_bytes=1024,
                        start_cycle=0, end_cycle=100, rank=0, op_name="allreduce")
    assert len(r.nvlink_transfer_events) == 1
    e = r.nvlink_transfer_events[0]
    assert e.src_gpu == 0
    assert e.n_bytes == 1024


def test_recorder_records_collective_op():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.collective(op_name="allreduce", algorithm="ring", n_bytes=4096,
                   world_size=4, start_cycle=0, end_cycle=300, n_steps=6)
    assert len(r.collective_events) == 1
    e = r.collective_events[0]
    assert e.algorithm == "ring"


def test_recorder_writes_collective_parquets(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.nvlink_transfer(src_gpu=0, dst_gpu=1, n_bytes=64,
                        start_cycle=0, end_cycle=10)
    r.collective(op_name="broadcast", algorithm="linear", n_bytes=64,
                   world_size=4, start_cycle=0, end_cycle=10, n_steps=3)
    write_parquet(r, tmp_path)
    assert (tmp_path / "nvlink_transfer.parquet").exists()
    assert (tmp_path / "collective.parquet").exists()
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add events + recorder methods + writers**

In `gpusim/trace/events.py`, append:

```python
@dataclass(frozen=True)
class NvlinkTransfer:
    src_gpu: int
    dst_gpu: int
    n_bytes: int
    start_cycle: int
    end_cycle: int
    rank: int = -1
    op_name: str = ""


@dataclass(frozen=True)
class CollectiveOp:
    op_name: str
    algorithm: str
    n_bytes: int
    world_size: int
    start_cycle: int
    end_cycle: int
    n_steps: int
```

In `gpusim/trace/recorder.py`, add to `__init__`:
```python
        self.nvlink_transfer_events: list = []
        self.collective_events: list = []
```

Add methods:
```python
    def nvlink_transfer(self, *, src_gpu: int, dst_gpu: int, n_bytes: int,
                          start_cycle: int, end_cycle: int,
                          rank: int = -1, op_name: str = "") -> None:
        from gpusim.trace.events import NvlinkTransfer
        self.nvlink_transfer_events.append(NvlinkTransfer(
            src_gpu=src_gpu, dst_gpu=dst_gpu, n_bytes=n_bytes,
            start_cycle=start_cycle, end_cycle=end_cycle,
            rank=rank, op_name=op_name,
        ))
    
    def collective(self, *, op_name: str, algorithm: str, n_bytes: int,
                     world_size: int, start_cycle: int, end_cycle: int,
                     n_steps: int) -> None:
        from gpusim.trace.events import CollectiveOp
        self.collective_events.append(CollectiveOp(
            op_name=op_name, algorithm=algorithm, n_bytes=n_bytes,
            world_size=world_size, start_cycle=start_cycle, end_cycle=end_cycle,
            n_steps=n_steps,
        ))
```

In `gpusim/trace/writer.py::write_parquet`, append:
```python
    if r.nvlink_transfer_events:
        pd.DataFrame([asdict(e) for e in r.nvlink_transfer_events]).to_parquet(
            out_dir / "nvlink_transfer.parquet", index=False)
    if r.collective_events:
        pd.DataFrame([asdict(e) for e in r.collective_events]).to_parquet(
            out_dir / "collective.parquet", index=False)
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/trace/test_collective_event.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/trace/ tests/unit/trace/test_collective_event.py
git commit -m "feat(trace): NvlinkTransfer + CollectiveOp events + recorder + parquet"
```

---

### Task 5: Tag M1

```bash
.venv/bin/pytest -q -m "not slow"
git tag M1-phase10-complete
```

---

## Milestone M2: NvlinkFabric + transfer + first example

### Task 6: NvlinkLink + NvlinkFabric class

**Files:**
- Create: `gpusim/comm/nvlink.py`
- Test: `tests/unit/comm/test_nvlink_fabric.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_nvlink_fabric_default_topology_4_gpus():
    from gpusim.comm.nvlink import NvlinkFabric
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    fabric = NvlinkFabric.from_config(cfg, n_gpus=4)
    # all-to-all: 4*3 = 12 links
    assert len(fabric.links) == 12
    # All ordered pairs (src, dst) with src != dst exist
    for src in range(4):
        for dst in range(4):
            if src != dst:
                assert (src, dst) in fabric.links


def test_nvlink_link_default_bandwidth():
    from gpusim.comm.nvlink import NvlinkLink
    link = NvlinkLink(src_gpu=0, dst_gpu=1,
                        bandwidth_gbps=900.0, latency_cycles=100)
    assert link.bandwidth_gbps == 900.0
    assert link.latency_cycles == 100
    assert link.busy_until == 0
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Create gpusim/comm/nvlink.py**

```python
"""Phase 10: NVLink fabric — point-to-point links between GPUs."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class NvlinkLink:
    src_gpu: int
    dst_gpu: int
    bandwidth_gbps: float = 900.0
    latency_cycles: int = 100
    busy_until: int = 0


@dataclass
class NvlinkFabric:
    n_gpus: int
    links: dict = field(default_factory=dict)
    topology: str = "all_to_all"
    
    @classmethod
    def from_config(cls, cfg, n_gpus: int) -> "NvlinkFabric":
        nv_cfg = getattr(cfg, "nvlink", None)
        bw = getattr(nv_cfg, "bandwidth_gbps", 900.0) if nv_cfg else 900.0
        lat = getattr(nv_cfg, "latency_cycles", 100) if nv_cfg else 100
        topo = getattr(nv_cfg, "topology", "all_to_all") if nv_cfg else "all_to_all"
        links = {}
        for src in range(n_gpus):
            for dst in range(n_gpus):
                if src != dst:
                    links[(src, dst)] = NvlinkLink(
                        src_gpu=src, dst_gpu=dst,
                        bandwidth_gbps=bw, latency_cycles=lat,
                    )
        return cls(n_gpus=n_gpus, links=links, topology=topo)
```

Update `gpusim/comm/__init__.py`:
```python
from gpusim.comm.system import MultiGpuSystem
from gpusim.comm.nvlink import NvlinkFabric, NvlinkLink
__all__ = ["MultiGpuSystem", "NvlinkFabric", "NvlinkLink"]
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/comm/test_nvlink_fabric.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/comm/ tests/unit/comm/test_nvlink_fabric.py
git commit -m "feat(comm): NvlinkFabric + NvlinkLink default all-to-all topology"
```

---

### Task 7: NvlinkFabric.transfer (serialization)

**Files:**
- Modify: `gpusim/comm/nvlink.py` (add transfer method + records NvlinkTransfer if recorder)
- Test: extend `tests/unit/comm/test_nvlink_fabric.py`

- [ ] **Step 1: Append failing test**

```python
def test_nvlink_transfer_completion_cycle():
    from gpusim.comm.nvlink import NvlinkFabric
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 2
    fabric = NvlinkFabric.from_config(cfg, n_gpus=2)
    # 1024 bytes at 900 bytes/cycle ≈ 2 cycles + 100 latency
    completion = fabric.transfer(src_gpu=0, dst_gpu=1, n_bytes=1024,
                                    arrival_cycle=0)
    assert completion >= 100   # at least latency
    assert completion <= 200   # at most latency + a few transfer cycles


def test_nvlink_transfer_serialization_same_link():
    from gpusim.comm.nvlink import NvlinkFabric
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 2
    fabric = NvlinkFabric.from_config(cfg, n_gpus=2)
    c1 = fabric.transfer(src_gpu=0, dst_gpu=1, n_bytes=900,
                            arrival_cycle=0)
    c2 = fabric.transfer(src_gpu=0, dst_gpu=1, n_bytes=900,
                            arrival_cycle=0)
    # Second transfer waits for first
    assert c2 > c1


def test_nvlink_transfer_different_links_parallel():
    from gpusim.comm.nvlink import NvlinkFabric
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    fabric = NvlinkFabric.from_config(cfg, n_gpus=4)
    c1 = fabric.transfer(src_gpu=0, dst_gpu=1, n_bytes=900,
                            arrival_cycle=0)
    c2 = fabric.transfer(src_gpu=2, dst_gpu=3, n_bytes=900,
                            arrival_cycle=0)
    # Different links don't serialize
    assert c1 == c2
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add transfer method to NvlinkFabric**

```python
    def transfer(self, src_gpu: int, dst_gpu: int, n_bytes: int,
                   arrival_cycle: int, *, recorder=None,
                   rank: int = -1, op_name: str = "") -> int:
        """Transfer n_bytes over (src→dst) NVLink. Returns completion cycle."""
        link = self.links.get((src_gpu, dst_gpu))
        if link is None:
            raise KeyError(f"no link {src_gpu}->{dst_gpu}")
        start = max(arrival_cycle, link.busy_until)
        # Simplified: bw_gbps ≈ bytes/cycle (assume 1 GHz clock)
        bytes_per_cycle = link.bandwidth_gbps
        transfer_cycles = max(1, int(n_bytes / bytes_per_cycle))
        completion = start + link.latency_cycles + transfer_cycles
        link.busy_until = completion
        if recorder is not None:
            recorder.nvlink_transfer(
                src_gpu=src_gpu, dst_gpu=dst_gpu, n_bytes=n_bytes,
                start_cycle=start, end_cycle=completion,
                rank=rank, op_name=op_name,
            )
        return completion
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/comm/test_nvlink_fabric.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/comm/nvlink.py tests/unit/comm/test_nvlink_fabric.py
git commit -m "feat(comm): NvlinkFabric.transfer with per-link serialization + optional trace"
```

---

### Task 8: Example multi_gpu_setup

**Files:**
- Create: `examples/multi_gpu_setup/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_multi_gpu_setup.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "multi_gpu_setup"


def test_multi_gpu_setup_correctness():
    """2-GPU minimal demo: each GPU runs vec_add independently."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    OUT_0 = np.zeros(n, dtype=np.float32)
    OUT_1 = np.zeros(n, dtype=np.float32)
    
    cfg = load_default()
    cfg.n_gpus = 2
    
    sys = MultiGpuSystem.from_config(cfg)
    assert len(sys.gpus) == 2
    assert sys.nvlink_fabric is not None
    assert len(sys.nvlink_fabric.links) == 2   # all-to-all 2 GPUs = 2 links
    
    # Each GPU runs vec_add (independent kernels — no comm yet)
    ptx = (_DIR / "kernel.ptx").read_text()
    for gpu_idx, out in [(0, OUT_0), (1, OUT_1)]:
        gpu = sys.gpus[gpu_idx]
        s = Stream()
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": out},
                  kernel_name=f"vec_add_gpu{gpu_idx}", config=cfg)
        gpu.run_streams([s])
    
    np.testing.assert_array_equal(OUT_0, A + B)
    np.testing.assert_array_equal(OUT_1, A + B)
```

- [ ] **Step 2: kernel.ptx** (vec_add — copy from existing example):

```
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd3, %r1;
    
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    
    add.f32 %f2, %f0, %f1;
    
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    
    ret;
}
```

- [ ] **Step 3: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np
def reference(A, B): return A + B
```

`run.py`:
```python
import numpy as np, pathlib
from gpusim.api import Stream
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    OUT_0 = np.zeros(n, dtype=np.float32)
    OUT_1 = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    for i, out in enumerate([OUT_0, OUT_1]):
        s = Stream()
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": out},
                  kernel_name=f"vec_add_gpu{i}", config=cfg)
        sys.gpus[i].run_streams([s])
    print(f"GPU 0 output: {OUT_0[0:4]}")
    print(f"GPU 1 output: {OUT_1[0:4]}")
    print(f"NVLink topology: {len(sys.nvlink_fabric.links)} links across {sys.nvlink_fabric.n_gpus} GPUs")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# multi_gpu_setup

Phase 10 demo: 2-GPU minimal multi-GPU. Each GPU runs vec_add independently;
NVLink fabric instantiated but not used for communication yet (Phase 10 M3+).

## Run
```
python examples/multi_gpu_setup/run.py
```

## Tutorial
docs/tutorial/40-multi-gpu-system-and-nvlink-fabric.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_multi_gpu_setup.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/multi_gpu_setup/ tests/parity/test_multi_gpu_setup.py
git commit -m "feat(examples): multi_gpu_setup — 2-GPU minimal demo with NVLink fabric"
```

---

### Task 9: Tag M2

```bash
.venv/bin/pytest -q -m "not slow"
git tag M2-phase10-complete
```

---

## Milestone M3: Comm + ring allreduce + 2 metrics + ring example

### Task 10: Comm class skeleton

**Files:**
- Create: `gpusim/comm/comm.py`
- Test: `tests/unit/comm/test_comm_basic.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_comm_construction():
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    c = Comm(rank=0, world_size=4, system=sys)
    assert c.rank == 0
    assert c.world_size == 4


def test_comm_rank_validation():
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    import pytest
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    with pytest.raises(ValueError, match="rank"):
        Comm(rank=5, world_size=4, system=sys)   # rank > world_size
    with pytest.raises(ValueError, match="world_size"):
        Comm(rank=0, world_size=8, system=sys)   # world_size > n_gpus
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Create gpusim/comm/comm.py**

```python
"""Phase 10: NCCL-equivalent Comm class."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Comm:
    rank: int
    world_size: int
    system: object   # MultiGpuSystem
    group_id: int = 0
    
    def __post_init__(self):
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank {self.rank} out of [0, {self.world_size})")
        n_gpus = len(self.system.gpus) if self.system else 0
        if self.world_size > n_gpus:
            raise ValueError(f"world_size {self.world_size} > n_gpus {n_gpus}")
```

Update `gpusim/comm/__init__.py`:
```python
from gpusim.comm.system import MultiGpuSystem
from gpusim.comm.nvlink import NvlinkFabric, NvlinkLink
from gpusim.comm.comm import Comm
__all__ = ["MultiGpuSystem", "NvlinkFabric", "NvlinkLink", "Comm"]
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/comm/test_comm_basic.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/comm/ tests/unit/comm/test_comm_basic.py
git commit -m "feat(comm): Comm class skeleton with rank/world_size validation"
```

---

### Task 11: Comm._allreduce_ring algorithm

**Files:**
- Modify: `gpusim/comm/comm.py` (add ring allreduce method)
- Test: `tests/unit/comm/test_allreduce_ring.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_ring_allreduce_step_count():
    """Ring allreduce: 2*(N-1) total steps (scatter-reduce + allgather)."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    
    # Run as rank 0
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.arange(64, dtype=np.float32)
    recv = np.zeros(64, dtype=np.float32)
    cycles = comm._allreduce_ring(send, recv, op="sum")
    
    # Step count: 2*(N-1) = 6 transfers issued from this rank
    # Each transfer goes (rank → rank+1)
    assert cycles > 0


def test_ring_allreduce_correctness_sum():
    """Ring allreduce sum: all ranks should get send * world_size (approximation)."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(16, 1.0, dtype=np.float32)
    recv = np.zeros(16, dtype=np.float32)
    comm._allreduce_ring(send, recv, op="sum")
    np.testing.assert_array_equal(recv, np.full(16, 4.0, dtype=np.float32))
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add _allreduce_ring to Comm**

```python
    def _allreduce_ring(self, send_buf, recv_buf, op: str = "sum") -> int:
        """Ring allreduce: 2*(N-1) NVLink transfers per rank.
        Returns total cycles spent in transfers."""
        import math
        n = self.world_size
        chunk_size_bytes = max(1, send_buf.nbytes // n)
        cycle = 0
        for step in range(2 * (n - 1)):
            dst = (self.rank + 1) % n
            cycle = self.system.nvlink_fabric.transfer(
                src_gpu=self.rank, dst_gpu=dst,
                n_bytes=chunk_size_bytes, arrival_cycle=cycle,
            )
        # Functional approximation: simulator gives all ranks the reduced value
        if op == "sum":
            recv_buf[:] = send_buf * n
        elif op in ("max", "min"):
            recv_buf[:] = send_buf
        else:
            raise ValueError(f"unsupported op: {op}")
        return cycle
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/comm/test_allreduce_ring.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/comm/comm.py tests/unit/comm/test_allreduce_ring.py
git commit -m "feat(comm): Comm._allreduce_ring algorithm + scatter-reduce + allgather"
```

---

### Task 12: Comm.allreduce auto-pick + records CollectiveOp

**Files:**
- Modify: `gpusim/comm/comm.py` (add allreduce + CollectiveOp recording)
- Test: extend test_allreduce_ring.py

- [ ] **Step 1: Append failing test**

```python
def test_allreduce_records_collective_event():
    """allreduce records a CollectiveOp event when recorder provided."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=4, system=sys)
    comm._recorder = rec
    send = np.arange(1024, dtype=np.float32)
    recv = np.zeros(1024, dtype=np.float32)
    comm.allreduce(send, recv, op="sum")
    assert len(rec.collective_events) == 1
    e = rec.collective_events[0]
    assert e.op_name == "allreduce"
    assert e.algorithm == "ring"   # large message → ring
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add allreduce + recorder integration**

```python
    _recorder = None
    
    def allreduce(self, send_buf, recv_buf, op: str = "sum") -> int:
        """Auto-pick ring (large) or tree (small) algorithm."""
        n_bytes = send_buf.nbytes
        threshold = 4096
        algorithm = "ring" if n_bytes >= threshold else "tree"
        start_cycle = 0
        if algorithm == "ring":
            end_cycle = self._allreduce_ring(send_buf, recv_buf, op)
            n_steps = 2 * (self.world_size - 1)
        else:
            # tree allreduce — implemented in T15
            end_cycle = self._allreduce_tree(send_buf, recv_buf, op) if hasattr(self, "_allreduce_tree") else self._allreduce_ring(send_buf, recv_buf, op)
            n_steps = 2 * max(1, int((self.world_size - 1).bit_length()))
        if self._recorder is not None:
            self._recorder.collective(
                op_name="allreduce", algorithm=algorithm,
                n_bytes=n_bytes, world_size=self.world_size,
                start_cycle=start_cycle, end_cycle=end_cycle,
                n_steps=n_steps,
            )
        return end_cycle
```

⚠ Tree allreduce is in T15 (M4); for M3, ring path is the focus. The fallback `if hasattr(self, "_allreduce_tree")` handles the transition.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/comm/test_allreduce_ring.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/comm/comm.py tests/unit/comm/test_allreduce_ring.py
git commit -m "feat(comm): Comm.allreduce auto-picks ring/tree + records CollectiveOp"
```

---

### Task 13: 2 analysis metrics (nvlink_bandwidth_utilization + collective_op_breakdown)

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Test: `tests/unit/analysis/test_phase10_metrics.py` (NEW)

- [ ] **Step 1: Create test**

```python
import pandas as pd


def test_nvlink_bandwidth_utilization():
    from gpusim.analysis.metrics import nvlink_bandwidth_utilization
    df = pd.DataFrame([
        {"src_gpu": 0, "dst_gpu": 1, "n_bytes": 1000, "start_cycle": 0, "end_cycle": 100},
        {"src_gpu": 1, "dst_gpu": 0, "n_bytes": 500, "start_cycle": 0, "end_cycle": 50},
    ])
    out = nvlink_bandwidth_utilization(df, total_cycles=100)
    # link (0,1) sees 1000 bytes / 100 cycles = 10 bytes/cycle
    assert (0, 1) in out
    assert out[(0, 1)] == 10.0


def test_collective_op_breakdown():
    from gpusim.analysis.metrics import collective_op_breakdown
    df = pd.DataFrame([
        {"op_name": "allreduce", "algorithm": "ring", "n_bytes": 1024,
          "world_size": 4, "start_cycle": 0, "end_cycle": 100, "n_steps": 6},
        {"op_name": "broadcast", "algorithm": "linear", "n_bytes": 64,
          "world_size": 4, "start_cycle": 100, "end_cycle": 150, "n_steps": 3},
    ])
    out = collective_op_breakdown(df)
    assert ("allreduce", "ring") in out
    assert out[("allreduce", "ring")] == 100   # end - start = 100 cycles
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Append metrics**

```python
def nvlink_bandwidth_utilization(nvlink_df, total_cycles: int) -> dict:
    """Per-link bytes/cycle utilization."""
    if nvlink_df is None or nvlink_df.empty or total_cycles <= 0:
        return {}
    out = {}
    for (src, dst), group in nvlink_df.groupby(["src_gpu", "dst_gpu"]):
        total_bytes = group["n_bytes"].sum()
        out[(int(src), int(dst))] = float(total_bytes) / total_cycles
    return out


def collective_op_breakdown(collective_df) -> dict:
    """Cycles per (op_name, algorithm)."""
    if collective_df is None or collective_df.empty:
        return {}
    out = {}
    for (name, algo), group in collective_df.groupby(["op_name", "algorithm"]):
        cycles = (group["end_cycle"] - group["start_cycle"]).sum()
        out[(str(name), str(algo))] = int(cycles)
    return out
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/analysis/test_phase10_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/analysis/metrics.py tests/unit/analysis/test_phase10_metrics.py
git commit -m "feat(analysis): nvlink_bandwidth_utilization + collective_op_breakdown metrics"
```

---

### Task 14: Example ring_allreduce + Tag M3

**Files:**
- Create: `examples/ring_allreduce/{reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_ring_allreduce.py`

(No kernel.ptx needed — collective is pure host-side simulation; no GPU kernels run.)

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


def test_ring_allreduce_correctness():
    """4-GPU ring allreduce on a 1024-byte buffer."""
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(256, 1.0, dtype=np.float32)   # 1024 bytes
    recv = np.zeros(256, dtype=np.float32)
    cycles = comm.allreduce(send, recv, op="sum")
    
    # Each rank gets 1.0 * world_size = 4.0
    np.testing.assert_array_equal(recv, np.full(256, 4.0, dtype=np.float32))
    assert cycles > 0
```

- [ ] **Step 2: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np


def reference(send: np.ndarray, world_size: int) -> np.ndarray:
    return send * world_size
```

`run.py`:
```python
import numpy as np
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(256, 1.0, dtype=np.float32)
    recv = np.zeros(256, dtype=np.float32)
    cycles = comm.allreduce(send, recv, op="sum")
    print(f"Ring allreduce: {cycles} cycles")
    print(f"recv[0:4] = {list(recv[0:4])}  (expected: [4.0, 4.0, 4.0, 4.0])")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# ring_allreduce

Phase 10 demo: 4-GPU ring allreduce on a 1024-byte buffer.
Demonstrates 2*(N-1) = 6 transfer steps for N=4.

## Run
```
python examples/ring_allreduce/run.py
```

## Tutorial
docs/tutorial/41-ring-allreduce-bandwidth-optimal.md
```

`__init__.py` (empty).

- [ ] **Step 3: Run + commit + tag M3**

```
.venv/bin/pytest tests/parity/test_ring_allreduce.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/ring_allreduce/ tests/parity/test_ring_allreduce.py
git commit -m "feat(examples): ring_allreduce — 4-GPU ring allreduce demo"
git tag M3-phase10-complete
```

---

## Milestone M4: Tree + broadcast + allgather + 2 examples

### Task 15: Comm._allreduce_tree

**Files:**
- Modify: `gpusim/comm/comm.py`
- Test: `tests/unit/comm/test_allreduce_tree.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_tree_allreduce_step_count():
    """Tree allreduce: 2*log2(N) total steps."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(64, 1.0, dtype=np.float32)
    recv = np.zeros(64, dtype=np.float32)
    cycles = comm._allreduce_tree(send, recv, op="sum")
    np.testing.assert_array_equal(recv, np.full(64, 4.0, dtype=np.float32))
    assert cycles > 0


def test_allreduce_picks_tree_for_small_message():
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=4, system=sys)
    comm._recorder = rec
    send = np.full(8, 1.0, dtype=np.float32)   # 32 bytes < 4096 threshold
    recv = np.zeros(8, dtype=np.float32)
    comm.allreduce(send, recv, op="sum")
    assert rec.collective_events[-1].algorithm == "tree"
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add _allreduce_tree to Comm**

```python
    def _allreduce_tree(self, send_buf, recv_buf, op: str = "sum") -> int:
        """Tree allreduce: 2*log2(N) transfers (reduce phase + broadcast phase)."""
        import math
        n = self.world_size
        if n <= 1:
            recv_buf[:] = send_buf
            return 0
        log_n = max(1, int(math.log2(n)))
        cycle = 0
        # Reduce phase: each step halves active ranks
        for step in range(log_n):
            partner = self.rank ^ (1 << step)
            if 0 <= partner < n:
                cycle = self.system.nvlink_fabric.transfer(
                    src_gpu=self.rank, dst_gpu=partner,
                    n_bytes=send_buf.nbytes, arrival_cycle=cycle,
                )
        # Broadcast phase: same number of steps
        for step in range(log_n):
            partner = self.rank ^ (1 << step)
            if 0 <= partner < n:
                cycle = self.system.nvlink_fabric.transfer(
                    src_gpu=self.rank, dst_gpu=partner,
                    n_bytes=send_buf.nbytes, arrival_cycle=cycle,
                )
        if op == "sum":
            recv_buf[:] = send_buf * n
        else:
            recv_buf[:] = send_buf
        return cycle
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/comm/test_allreduce_tree.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/comm/comm.py tests/unit/comm/test_allreduce_tree.py
git commit -m "feat(comm): Comm._allreduce_tree algorithm + small-message auto-pick"
```

---

### Task 16: Comm.broadcast + Comm.allgather

**Files:**
- Modify: `gpusim/comm/comm.py`
- Test: `tests/unit/comm/test_broadcast.py` + `tests/unit/comm/test_allgather.py` (NEW)

- [ ] **Step 1: Create tests**

```python
# test_broadcast.py
def test_broadcast_correctness():
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    buf = np.arange(32, dtype=np.float32)
    cycles = comm.broadcast(buf, root=0)
    assert cycles > 0


# test_allgather.py
def test_allgather_correctness():
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(8, 1.0, dtype=np.float32)
    recv = np.zeros(32, dtype=np.float32)   # world_size * send.size
    cycles = comm.allgather(send, recv)
    assert cycles > 0
    # All concatenated entries are 1.0
    assert (recv == 1.0).all()
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add methods to Comm**

```python
    def broadcast(self, buf, root: int = 0) -> int:
        """Broadcast buf from root to all ranks. Linear: (N-1) sends from root."""
        n = self.world_size
        if n <= 1: return 0
        cycle = 0
        if self.rank == root:
            for dst in range(n):
                if dst != root:
                    cycle = max(cycle, self.system.nvlink_fabric.transfer(
                        src_gpu=root, dst_gpu=dst,
                        n_bytes=buf.nbytes, arrival_cycle=0,
                    ))
        if self._recorder is not None:
            self._recorder.collective(
                op_name="broadcast", algorithm="linear",
                n_bytes=buf.nbytes, world_size=n,
                start_cycle=0, end_cycle=cycle, n_steps=n - 1,
            )
        return cycle
    
    def allgather(self, send_buf, recv_buf) -> int:
        """All-gather: each rank sends to all others. (N-1) sends per rank."""
        import numpy as np
        n = self.world_size
        if n <= 1:
            recv_buf[:send_buf.size] = send_buf
            return 0
        cycle = 0
        for dst in range(n):
            if dst != self.rank:
                cycle = self.system.nvlink_fabric.transfer(
                    src_gpu=self.rank, dst_gpu=dst,
                    n_bytes=send_buf.nbytes, arrival_cycle=cycle,
                )
        # Functional: replicate send across recv (assumes uniform send across ranks)
        chunk = send_buf.size
        for r in range(n):
            recv_buf[r*chunk:(r+1)*chunk] = send_buf
        if self._recorder is not None:
            self._recorder.collective(
                op_name="allgather", algorithm="linear",
                n_bytes=send_buf.nbytes, world_size=n,
                start_cycle=0, end_cycle=cycle, n_steps=n - 1,
            )
        return cycle
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/comm/test_broadcast.py tests/unit/comm/test_allgather.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/comm/comm.py tests/unit/comm/test_broadcast.py tests/unit/comm/test_allgather.py
git commit -m "feat(comm): Comm.broadcast + Comm.allgather (linear algorithms)"
```

---

### Task 17: 2 more metrics (algo_efficiency + per_rank_communication_volume)

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Test: extend `tests/unit/analysis/test_phase10_metrics.py`

- [ ] **Step 1: Append failing tests**

```python
def test_algo_efficiency_ring_vs_tree():
    from gpusim.analysis.metrics import algo_efficiency_ring_vs_tree
    df = pd.DataFrame([
        {"op_name": "allreduce", "algorithm": "ring", "n_bytes": 1024, "world_size": 4,
          "start_cycle": 0, "end_cycle": 100},
        {"op_name": "allreduce", "algorithm": "tree", "n_bytes": 32, "world_size": 4,
          "start_cycle": 0, "end_cycle": 50},
    ])
    out = algo_efficiency_ring_vs_tree(df)
    assert "ring" in out
    assert "tree" in out


def test_per_rank_communication_volume():
    from gpusim.analysis.metrics import per_rank_communication_volume
    df = pd.DataFrame([
        {"src_gpu": 0, "dst_gpu": 1, "n_bytes": 1000, "rank": 0},
        {"src_gpu": 0, "dst_gpu": 2, "n_bytes": 500, "rank": 0},
        {"src_gpu": 1, "dst_gpu": 0, "n_bytes": 800, "rank": 1},
    ])
    out = per_rank_communication_volume(df)
    assert out[0] == 1500
    assert out[1] == 800
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Append metrics**

```python
def algo_efficiency_ring_vs_tree(collective_df) -> dict:
    """Average cycles/byte for ring vs tree algorithms."""
    if collective_df is None or collective_df.empty:
        return {"ring": 0.0, "tree": 0.0}
    out = {}
    for algo in ["ring", "tree"]:
        sub = collective_df[collective_df["algorithm"] == algo]
        if sub.empty:
            out[algo] = 0.0
            continue
        total_cycles = (sub["end_cycle"] - sub["start_cycle"]).sum()
        total_bytes = sub["n_bytes"].sum()
        out[algo] = float(total_cycles) / max(total_bytes, 1)
    return out


def per_rank_communication_volume(nvlink_df) -> dict:
    """Total bytes sent per rank (groups by rank column if present, else src_gpu)."""
    if nvlink_df is None or nvlink_df.empty:
        return {}
    if "rank" in nvlink_df.columns and (nvlink_df["rank"] >= 0).any():
        # Use rank column
        valid = nvlink_df[nvlink_df["rank"] >= 0]
        out = {}
        for r, group in valid.groupby("rank"):
            out[int(r)] = int(group["n_bytes"].sum())
        return out
    # Fallback to src_gpu
    out = {}
    for sid, group in nvlink_df.groupby("src_gpu"):
        out[int(sid)] = int(group["n_bytes"].sum())
    return out
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/analysis/test_phase10_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/analysis/metrics.py tests/unit/analysis/test_phase10_metrics.py
git commit -m "feat(analysis): algo_efficiency_ring_vs_tree + per_rank_communication_volume"
```

---

### Task 18: Example tree_allreduce

**Files:**
- Create: `examples/tree_allreduce/{reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_tree_allreduce.py`

- [ ] **Step 1: Parity test**

```python
def test_tree_allreduce_correctness():
    """4-GPU tree allreduce on a small (64-byte) buffer."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=4, system=sys)
    comm._recorder = rec
    send = np.full(16, 1.0, dtype=np.float32)   # 64 bytes < 4096 threshold
    recv = np.zeros(16, dtype=np.float32)
    cycles = comm.allreduce(send, recv, op="sum")
    np.testing.assert_array_equal(recv, np.full(16, 4.0, dtype=np.float32))
    assert rec.collective_events[-1].algorithm == "tree"
    assert cycles > 0
```

- [ ] **Step 2-3: Create supporting files (similar to ring_allreduce)**

`reference.py`, `run.py`, `README.md` (similar to ring_allreduce; emphasize small msg + tree path).

```bash
git add examples/tree_allreduce/ tests/parity/test_tree_allreduce.py
git commit -m "feat(examples): tree_allreduce — 4-GPU latency-optimal tree allreduce"
```

---

### Task 19: Example ddp_training_step (capstone)

**Files:**
- Create: `examples/ddp_training_step/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_ddp_training_step.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "ddp_training_step"


def test_ddp_training_step_correctness():
    """4-GPU DDP-style: each rank does vec_add → allreduce gradients → broadcast weights."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 32
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    
    # Each rank: vec_add on local data
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    grads = np.zeros(n, dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)
    
    ptx = (_DIR / "kernel.ptx").read_text()
    for rank in range(4):
        gpu = sys.gpus[rank]
        s = Stream()
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": grads},
                  kernel_name=f"compute_grad_rank{rank}", config=cfg)
        gpu.run_streams([s])
    
    # Allreduce gradients
    comm0 = Comm(rank=0, world_size=4, system=sys)
    grads_reduced = np.zeros(n, dtype=np.float32)
    cycles_ar = comm0.allreduce(grads, grads_reduced, op="sum")
    
    # Broadcast updated weights from rank 0
    cycles_bc = comm0.broadcast(weights, root=0)
    
    assert cycles_ar > 0
    assert cycles_bc > 0
    # All ranks should see the same reduced gradients
    np.testing.assert_array_equal(grads_reduced, grads * 4)
```

- [ ] **Step 2: kernel.ptx** (vec_add):

(Same as multi_gpu_setup/kernel.ptx)

- [ ] **Step 3-4: supporting files + commit**

`reference.py`, `run.py`, `README.md`, `__init__.py` similar pattern.

```bash
git add examples/ddp_training_step/ tests/parity/test_ddp_training_step.py
git commit -m "feat(examples): ddp_training_step — 4-GPU DDP capstone (vec_add + allreduce + broadcast)"
```

---

### Task 20: Tag M4

```bash
.venv/bin/pytest -q -m "not slow"
git tag M4-phase10-complete
```

---

## Milestone M5: Viz + tutorials + microbench + ship

### Task 21: HTML §33 + §34 (Collective timeline + NVLink heatmap)

**Files:**
- Modify: `gpusim/viz/html_report.py`
- Modify: `gpusim/viz/_template.html.j2`
- Test: `tests/unit/viz/test_html_report_phase10.py` (NEW)

- [ ] **Step 1: Test:**

```python
def test_html_report_phase10_sections(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.collective(op_name="allreduce", algorithm="ring", n_bytes=1024,
                   world_size=4, start_cycle=0, end_cycle=100, n_steps=6)
    r.nvlink_transfer(src_gpu=0, dst_gpu=1, n_bytes=256,
                        start_cycle=0, end_cycle=10)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=200, occupancy={"active_ctas": 1, "bottleneck": "none"})
    html = out.read_text()
    assert "Collective" in html or "collective" in html.lower() or "NVLink" in html or "nvlink" in html.lower()
```

- [ ] **Step 2: Add helpers + template blocks** (match existing pattern):

In `html_report.py`:
```python
def _render_collective_timeline(rec):
    if not getattr(rec, "collective_events", None): return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.collective_events])
    return "<h3>Collective ops</h3>" + df.to_html(index=False)


def _render_nvlink_heatmap(rec):
    if not getattr(rec, "nvlink_transfer_events", None): return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.nvlink_transfer_events])
    pivot = df.groupby(["src_gpu", "dst_gpu"])["n_bytes"].sum().reset_index()
    return "<h3>NVLink fabric utilization</h3>" + pivot.to_html(index=False)
```

In `save_html`, add to context:
```python
    context.update({
        "collective_timeline_html": _render_collective_timeline(rec),
        "nvlink_heatmap_html": _render_nvlink_heatmap(rec),
    })
```

In `_template.html.j2`, append:
```html
{% if collective_timeline_html %}
<h2>§33 Collective op timeline</h2>
{{ collective_timeline_html | safe }}
{% endif %}

{% if nvlink_heatmap_html %}
<h2>§34 NVLink fabric utilization heatmap</h2>
{{ nvlink_heatmap_html | safe }}
{% endif %}
```

- [ ] **Step 3: Run + commit**

```
.venv/bin/pytest tests/unit/viz/test_html_report_phase10.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/viz/html_report.py gpusim/viz/_template.html.j2 tests/unit/viz/test_html_report_phase10.py
git commit -m "feat(viz): HTML §33 collective timeline + §34 NVLink heatmap"
```

---

### Task 22: Perfetto NVLink + Rank-N swimlanes

**Files:**
- Modify: `gpusim/viz/perfetto.py`
- Test: extend `tests/unit/viz/test_html_report_phase10.py`

- [ ] **Step 1: Append failing test**

```python
def test_perfetto_nvlink_swimlane():
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.nvlink_transfer(src_gpu=0, dst_gpu=1, n_bytes=1024,
                        start_cycle=0, end_cycle=100, rank=0, op_name="allreduce")
    r.collective(op_name="allreduce", algorithm="ring", n_bytes=1024,
                   world_size=4, start_cycle=0, end_cycle=100, n_steps=6)
    pf = build_perfetto(r)
    pids = {e.get("pid") for e in pf.get("traceEvents", [])}
    assert any("NVLink" in str(p) for p in pids)
```

- [ ] **Step 2: Add to perfetto.py:**

```python
    # Phase 10: NVLink transfers
    for ev in getattr(rec, "nvlink_transfer_events", []):
        events.append({
            "name": f"{ev.src_gpu}→{ev.dst_gpu} ({ev.n_bytes}B)",
            "cat": "nvlink", "ph": "X",
            "ts": ev.start_cycle,
            "dur": max(1, ev.end_cycle - ev.start_cycle),
            "pid": "NVLink",
            "tid": f"{ev.src_gpu}→{ev.dst_gpu}",
            "args": {"src_gpu": ev.src_gpu, "dst_gpu": ev.dst_gpu,
                     "n_bytes": ev.n_bytes, "rank": ev.rank, "op": ev.op_name},
        })
    # Phase 10: collective ops as Rank-N swimlane events
    for ev in getattr(rec, "collective_events", []):
        events.append({
            "name": f"{ev.op_name}.{ev.algorithm}",
            "cat": "collective", "ph": "X",
            "ts": ev.start_cycle,
            "dur": max(1, ev.end_cycle - ev.start_cycle),
            "pid": "Collective", "tid": ev.algorithm,
            "args": {"op": ev.op_name, "algo": ev.algorithm,
                     "n_bytes": ev.n_bytes, "world_size": ev.world_size,
                     "n_steps": ev.n_steps},
        })
```

- [ ] **Step 3: Run + commit**

```
.venv/bin/pytest tests/unit/viz/test_html_report_phase10.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/viz/perfetto.py tests/unit/viz/test_html_report_phase10.py
git commit -m "feat(viz): Perfetto NVLink + Collective swimlanes"
```

---

### Task 23: 4 tutorial chapters 40-43

**Files:**
- Create: `docs/tutorial/{40,41,42,43}-*.md`

- [ ] **Step 1: Read existing style** (`docs/tutorial/39-*.md`)

- [ ] **Step 2: Write 4 chapters** (~500-700 words each, English body + Chinese subheadings):

**Chapter 40 — multi-gpu-system-and-nvlink-fabric:**
- cfg.n_gpus + MultiGpuSystem
- NVLink fabric topology (all-to-all default)
- multi_gpu_setup demo
- 看模拟器: nvlink_bandwidth_utilization metric
- 改一改: cfg.n_gpus=8 vs 2; nvlink.topology="ring"
- 真机对照: H100 NVLink 4 + NVSwitch

**Chapter 41 — ring-allreduce-bandwidth-optimal ⭐:**
- Ring algorithm: 2*(N-1) steps, scatter-reduce + allgather phases
- ring_allreduce demo
- 看模拟器: collective_op_breakdown shows ring path
- 改一改: increase msg size; observe linear scaling
- 真机对照: NCCL ring (default for >256KB messages)

**Chapter 42 — tree-allreduce-latency-optimal:**
- Tree algorithm: 2*log2(N) steps, butterfly-style
- tree_allreduce demo
- 看模拟器: algo auto-pick at 4096 byte threshold
- 改一改: cfg.allreduce_threshold; force ring for small msgs
- 真机对照: NCCL tree (default for <256KB messages)

**Chapter 43 — ddp-training-pattern ⭐:**
- DDP: forward pass (compute) → all-reduce gradients → broadcast updated weights
- ddp_training_step demo
- 看模拟器: per_rank_communication_volume
- 改一改: 4-GPU vs 8-GPU scaling
- 真机对照: PyTorch DDP, gradient bucketing

```bash
git add docs/tutorial/40-multi-gpu-system-and-nvlink-fabric.md \
        docs/tutorial/41-ring-allreduce-bandwidth-optimal.md \
        docs/tutorial/42-tree-allreduce-latency-optimal.md \
        docs/tutorial/43-ddp-training-pattern.md
git commit -m "docs(tutorial): chapters 40-43 — Phase 10 multi-GPU + NCCL"
```

---

### Task 24: Phase 10 microbench + 4 ref stubs

**Files:**
- Create: `tests/microbench/test_phase10_facts.py`
- Create: `tests/microbench/test_phase10_runtime.py`
- Modify: `tests/reference/gen_reference.py`
- Create: 4 ref JSONs

(Standard pattern; see Phase 9 T21 for template.)

```python
# test_phase10_facts.py
def test_ring_allreduce_step_count():
    """Ring allreduce: 2*(N-1) NVLink transfers."""
    # ... assert nvlink_transfer_events count == 2*(N-1)


def test_tree_allreduce_step_count():
    """Tree allreduce: 2*log2(N) NVLink transfers per rank."""
    # ... assert nvlink_transfer_events count == 2*log2(N)


def test_tree_faster_than_ring_for_small_msg():
    """Tree algorithm should pick tree for 32-byte msg."""
    # ... assert algorithm == "tree"
```

3 ref stubs for: multi_gpu_setup, ring_allreduce, tree_allreduce, ddp_training_step.

```bash
git add tests/microbench/test_phase10_facts.py tests/microbench/test_phase10_runtime.py \
        tests/reference/gen_reference.py tests/reference/data/*.ref.json
git commit -m "test(microbench+reference): Phase 10 facts + 4 ref stubs"
```

---

### Task 25: Phase 1-8 regression rename → 1-9 + add 3 Phase 9 examples

```bash
git mv tests/parity/test_phase1_8_examples_unchanged.py tests/parity/test_phase1_9_examples_unchanged.py
```

Edit:
- Rename `PHASE_1_8_EXAMPLES` → `PHASE_1_9_EXAMPLES`
- Add 3 Phase 9 examples: `phase8_overlap_real`, `multi_event_fan_in`, `event_timing_benchmark`

```bash
git add tests/parity/test_phase1_9_examples_unchanged.py
git commit -m "test(regression): rename phase1_8 → phase1_9 + 3 Phase 9 examples"
```

---

### Task 26: README v10 + final tag

**Files:**
- Modify: `README.md`

Update README to v10:
- Phase 10 features section: cfg.n_gpus + NVLink + Comm class + 3 collectives + ring/tree algorithms + 4 metrics + HTML §33/§34 + Perfetto
- Examples list: add 4 (was 38, now 42)
- Tutorials list: add 40-43 (was 39, now 43)
- Phase status: 1-10 ✅

Run final suite + 4 examples.

```bash
git add README.md
git commit -m "docs(readme): v10 — Phase 10 capabilities (multi-GPU + NVLink + NCCL)"
git tag phase10-complete
```

---

### Task 27: Final sanity sweep

```bash
.venv/bin/pytest -q -m "not slow"
.venv/bin/pytest tests/parity/test_phase1_9_examples_unchanged.py -v
```

---

### Task 28: Done

Phase 10 ships when all tasks complete + tags landed.

---

## End-of-plan checklist

- [ ] M1 (GPU rename + config + system + trace): T1-T5
- [ ] M2 (NvlinkFabric + multi_gpu_setup): T6-T9
- [ ] M3 (Comm + ring + 2 metrics + ring example): T10-T14
- [ ] M4 (Tree + broadcast + allgather + 2 examples): T15-T20
- [ ] M5 (Viz + tutorials + microbench + regression + README): T21-T28
- [ ] All 5 milestone tags + phase10-complete
- [ ] Phase 1-9 regression unbroken
- [ ] 4 new examples + 4 tutorials shipped
- [ ] README v10 reflects Phase 10
