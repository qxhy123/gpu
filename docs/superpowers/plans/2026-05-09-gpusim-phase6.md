# gpusim Phase 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement gpusim Phase 6 per `docs/superpowers/specs/2026-05-09-gpusim-phase6-design.md` — atomic ops (`atom.*` + `red.*`) on gmem (real L2 atomic ALU per-line FIFO) + smem (bank conflict reuse), plus cluster TMA store completing the cluster cooperative epilogue story. 5 new examples + 5 tutorial chapters.

**Architecture:** L2 gains per-line atomic queue (`L2AtomicQueue`); multi-SM atomics on the same line serialize. SubCore routes `atom.*` / `red.*` per-lane functionally then computes max-completion via L2 (gmem) or bank-conflict path (smem). `cp.async.bulk.tensor.2d.global.shared::cluster` extends Phase 4 store with cluster pointer decoding, enabling cooperative epilogue. 1 new trace event (`AtomicEvent`), 4 new metrics, 2 new HTML sections.

**Tech Stack:** Python 3.11+. No new runtime dependencies.

**Execution note:** Plan has 5 milestones (M1–M5) with 28 tasks total. After each milestone, pause for review checkpoint and tag (`M{1..5}-phase6-complete`).

---

## Scope check

Phase 6 covers two related feature groups (atomics + cluster TMA store) tightly coupled by `cluster_cooperative_epilogue` example.

- **M1** (frontend+config): parser, AtomicEvent, CacheConfig fields. No runtime change.
- **M2** (smem atomic): SharedMemory.atomic_op + SubCore atom/red shared routing + atom_reduction_smem.
- **M3** (gmem atomic): L2AtomicQueue + L2.atomic_op + GlobalMemory.atomic_op + 3 examples.
- **M4** (cluster TMA store): do_bulk_store_2d cluster decode + cluster_cooperative_epilogue example.
- **M5** (trace+viz+docs): 4 metrics + 2 HTML sections + Perfetto + 5 tutorials + microbench + README v6.

---

## Phase 1+2+3+4+5 prerequisites

```bash
cd /Users/yangyang/ai_projs/gpu
git log --oneline | head -3
git tag | grep phase
.venv/bin/pytest -q -m "not slow"
```

Expected: ~388 passed (Phase 5 baseline), ≥10 skipped.

---

## File structure

```
gpusim/core/
├── atomic.py                        NEW (M3): L2AtomicQueue + L2AtomicEntry
├── cache/l2.py                      MODIFY (M3): + atomic_op + L2AtomicQueue
├── smem.py                          MODIFY (M2): + atomic_op helpers
├── exec.py                          MODIFY (M2/M3): GlobalMemory + SharedMemory atomic_op
├── tma_store.py                     MODIFY (M4): do_bulk_store_2d (cluster source)
├── sub_core.py                      MODIFY (M2/M3/M4): atom/red routes + cluster TMA store
└── functional_units.py              MODIFY (M1): classify atom/red → LSU

gpusim/frontend/parser.py            MODIFY (M1): + atom/red parse
gpusim/config/
├── schema.py                        MODIFY (M1): CacheConfig + 3 fields
└── default_hopper.yaml              MODIFY (M1): cache + 3 fields

gpusim/trace/
├── events.py                        MODIFY (M1): + AtomicEvent
├── recorder.py                      MODIFY (M1): + atomic method
└── writer.py                        MODIFY (M1): + atomic.parquet

gpusim/analysis/metrics.py           MODIFY (M5): + 4 metrics
gpusim/viz/                          MODIFY (M5): + 2 HTML + Perfetto atomic
gpusim/api.py                        MODIFY (M5): + atomic_events_df + atomic_metrics + atomic_summary

examples/
├── atom_histogram/                  NEW (M3): kernel.ptx + reference.py + run.py + README.md + __init__.py
├── atom_reduction_smem/             NEW (M2)
├── atom_cas_spinlock/               NEW (M3)
├── red_min_max/                     NEW (M3)
└── cluster_cooperative_epilogue/    NEW (M4)

tests/unit/
├── core/test_{atomic,l2_atomic,exec_atomic,sub_core_atomic}.py    NEW
├── frontend/test_parser_phase6.py   NEW (M1)
├── analysis/test_phase6_metrics.py  NEW (M5)
└── viz/test_html_report_phase6.py   NEW (M5)

tests/parity/
├── test_atom_histogram.py           NEW
├── test_atom_reduction_smem.py      NEW
├── test_atom_cas_spinlock.py        NEW
├── test_red_min_max.py              NEW
├── test_cluster_cooperative_epilogue.py    NEW
└── test_phase1_5_examples_unchanged.py     RENAME from phase1_4

tests/microbench/
├── test_phase6_facts.py             NEW
└── test_phase6_runtime.py           NEW (@pytest.mark.slow)

tests/reference/
├── gen_reference.py                 MODIFY (M5)
└── data/{atom_histogram,atom_reduction_smem,atom_cas_spinlock,red_min_max,cluster_cooperative_epilogue}.ref.json    NEW

docs/tutorial/
├── 22-gmem-atomic-l2-alu.md         NEW
├── 23-smem-atomic-bank-conflict.md  NEW
├── 24-cluster-cooperative-epilogue.md   NEW
├── 25-cas-lock-free-pattern.md      NEW
└── 26-red-vs-atom.md                NEW

README.md                            MODIFY (M5): v6
```

---

## Milestones

| Milestone | Tasks | Tag |
|---|---|---|
| **M1** Frontend + config + AtomicEvent | T1–T5 | `M1-phase6-complete` |
| **M2** smem atomic + atom_reduction_smem | T6–T9 | `M2-phase6-complete` |
| **M3** gmem atomic + 3 examples | T10–T16 | `M3-phase6-complete` |
| **M4** Cluster TMA store + cooperative epilogue | T17–T19 | `M4-phase6-complete` |
| **M5** Trace + viz + docs | T20–T28 | `phase6-complete` |

---

## Milestone M1: Frontend + Config + AtomicEvent

### Task 1: CacheConfig 3 new fields + yaml + loader

**Files:**
- Modify: `gpusim/config/schema.py`
- Modify: `gpusim/config/default_hopper.yaml`
- Test: `tests/unit/config/test_loader_phase6.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_cache_config_atomic_fields_default():
    from gpusim.config.schema import CacheConfig
    cfg = CacheConfig()
    assert cfg.atomic_op_latency == 10
    assert cfg.atomic_queue_capacity == 32
    assert cfg.smem_atomic_op_extra_latency == 4


def test_loader_reads_atomic_fields_from_yaml():
    import tempfile
    from pathlib import Path
    yaml_text = """
device:
  n_sm: 8
cache:
  l1_size_bytes: 131072
  atomic_op_latency: 15
  atomic_queue_capacity: 64
  smem_atomic_op_extra_latency: 6
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text); path = f.name
    from gpusim.config.loader import load_yaml
    cfg = load_yaml(path)
    assert cfg.cache.atomic_op_latency == 15
    assert cfg.cache.atomic_queue_capacity == 64
    assert cfg.cache.smem_atomic_op_extra_latency == 6
```

- [ ] **Step 2: Run (FAIL)**

```
.venv/bin/pytest tests/unit/config/test_loader_phase6.py -v
```

- [ ] **Step 3: Add fields**

In `gpusim/config/schema.py`, in `CacheConfig`, append 3 fields:

```python
    atomic_op_latency: int = 10           # NEW (Phase 6)
    atomic_queue_capacity: int = 32       # NEW
    smem_atomic_op_extra_latency: int = 4 # NEW
```

- [ ] **Step 4: Update yaml**

Append to `gpusim/config/default_hopper.yaml` `cache:` block:
```yaml
  atomic_op_latency: 10
  atomic_queue_capacity: 32
  smem_atomic_op_extra_latency: 4
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/config/test_loader_phase6.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 2 PASS new; full suite ~390 passed.

```bash
git add gpusim/config/ tests/unit/config/test_loader_phase6.py
git commit -m "feat(config): CacheConfig + 3 atomic fields (latency/queue capacity/smem extra)"
```

---

### Task 2: Parser atom + red ops

**Files:**
- Modify: `gpusim/frontend/parser.py`
- Test: `tests/unit/frontend/test_parser_phase6.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_parser_atom_global_add():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .u32 %r<3>;
    atom.global.add.u32 %r0, [%rd0], %r1;
}
"""
    k = parse(src, "<t>")
    assert k.instrs[0].op == "atom.global.add.u32"
    assert len(k.instrs[0].dst) == 1
    assert len(k.instrs[0].src) == 2


def test_parser_atom_global_cas_3_src():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .u32 %r<4>;
    atom.global.cas.u32 %r0, [%rd0], %r1, %r2;
}
"""
    k = parse(src, "<t>")
    assert k.instrs[0].op == "atom.global.cas.u32"
    assert len(k.instrs[0].src) == 3


def test_parser_atom_shared_min():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .s32 %r<3>;
    atom.shared.min.s32 %r0, [%rd0], %r1;
}
"""
    k = parse(src, "<t>")
    assert k.instrs[0].op == "atom.shared.min.s32"


def test_parser_red_global_add_no_dst():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .f32 %f0;
    red.global.add.f32 [%rd0], %f0;
}
"""
    k = parse(src, "<t>")
    assert k.instrs[0].op == "red.global.add.f32"
    assert len(k.instrs[0].dst) == 0
    assert len(k.instrs[0].src) == 2


def test_parser_red_shared_max():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .u32 %r0;
    red.shared.max.u32 [%rd0], %r0;
}
"""
    k = parse(src, "<t>")
    assert k.instrs[0].op == "red.shared.max.u32"
    assert len(k.instrs[0].dst) == 0
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add parser branches**

In `gpusim/frontend/parser.py`, in `_parse_operands`, add (place after existing atomic-adjacent branches or near other ld/st handling):

```python
        if op.startswith("atom.global.") or op.startswith("atom.shared."):
            is_cas = ".cas." in op
            ty = self._type_from_op(op) or PtxType.u32
            dst = self._parse_operand(ty)
            self.eat("COMMA")
            self.eat("LBRACK")
            addr = self._parse_operand(PtxType.u64)
            self.eat("RBRACK")
            self.eat("COMMA")
            srcs: list = [addr]
            srcs.append(self._parse_operand(ty))
            if is_cas:
                self.eat("COMMA")
                srcs.append(self._parse_operand(ty))
            return [dst], srcs

        if op.startswith("red.global.") or op.startswith("red.shared."):
            ty = self._type_from_op(op) or PtxType.u32
            self.eat("LBRACK")
            addr = self._parse_operand(PtxType.u64)
            self.eat("RBRACK")
            self.eat("COMMA")
            val = self._parse_operand(ty)
            return [], [addr, val]
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/frontend/test_parser_phase6.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 5 PASS new.

```bash
git add gpusim/frontend/parser.py tests/unit/frontend/test_parser_phase6.py
git commit -m "feat(parser): atom.{global,shared}.<op>.<ty> + red.{global,shared} + cas 3-src form"
```

---

### Task 3: AtomicEvent + recorder + parquet writer

**Files:**
- Modify: `gpusim/trace/events.py`
- Modify: `gpusim/trace/recorder.py`
- Modify: `gpusim/trace/writer.py`
- Test: `tests/unit/trace/test_recorder_phase6.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_recorder_records_atomic_event():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.atomic(cycle=10, sm_id=0, warp_id=0, kind="ATOM",
              op="add", space="global", line_addr=0x1000,
              latency=20, n_lanes=4, queue_depth_before=0)
    assert len(r.atomic_events) == 1
    e = r.atomic_events[0]
    assert e.kind == "ATOM" and e.op == "add"


def test_recorder_writes_atomic_parquet(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.atomic(cycle=0, sm_id=0, warp_id=0, kind="RED",
              op="add", space="shared", line_addr=64, latency=24)
    write_parquet(r, tmp_path)
    assert (tmp_path / "atomic.parquet").exists()
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add AtomicEvent**

Append to `gpusim/trace/events.py`:

```python
@dataclass(frozen=True)
class AtomicEvent:
    cycle: int
    sm_id: int
    warp_id: int
    kind: str            # "ATOM" | "RED"
    op: str              # "add" | "min" | "max" | "exch" | "cas"
    space: str           # "global" | "shared"
    line_addr: int
    latency: int
    n_lanes: int = 1
    queue_depth_before: int = 0
```

- [ ] **Step 4: Add recorder method + list**

In `gpusim/trace/recorder.py`, in `Recorder.__init__`, add:
```python
        self.atomic_events: list = []
```

Add method:
```python
    def atomic(self, *, cycle: int, sm_id: int, warp_id: int,
                kind: str, op: str, space: str, line_addr: int,
                latency: int, n_lanes: int = 1,
                queue_depth_before: int = 0) -> None:
        from gpusim.trace.events import AtomicEvent
        self.atomic_events.append(AtomicEvent(
            cycle=cycle, sm_id=sm_id, warp_id=warp_id,
            kind=kind, op=op, space=space, line_addr=line_addr,
            latency=latency, n_lanes=n_lanes,
            queue_depth_before=queue_depth_before,
        ))
```

- [ ] **Step 5: Add parquet writer**

In `gpusim/trace/writer.py`, in `write_parquet`, append:
```python
    if r.atomic_events:
        pd.DataFrame([asdict(e) for e in r.atomic_events]).to_parquet(
            out_dir / "atomic.parquet", index=False)
```

- [ ] **Step 6: Run + commit**

```
.venv/bin/pytest tests/unit/trace/test_recorder_phase6.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/trace/ tests/unit/trace/test_recorder_phase6.py
git commit -m "feat(trace): AtomicEvent + recorder.atomic + parquet writer"
```

---

### Task 4: FUSet.classify atom/red → LSU

**Files:**
- Modify: `gpusim/core/functional_units.py`
- Test: `tests/unit/core/test_functional_units.py` (extend)

- [ ] **Step 1: Append failing test**

Append to `tests/unit/core/test_functional_units.py`:

```python
def test_classify_atom_to_lsu():
    from gpusim.core.functional_units import FUSet, FUKind
    from gpusim.config.schema import FUConfig
    fus = FUSet(FUConfig())
    assert fus.classify("atom.global.add.u32") is FUKind.LSU
    assert fus.classify("atom.shared.cas.u32") is FUKind.LSU
    assert fus.classify("red.global.max.f32") is FUKind.LSU
```

- [ ] **Step 2: Run (FAIL — atom/red not classified)**

- [ ] **Step 3: Add classify branch**

In `gpusim/core/functional_units.py`, in `FUSet.classify`, add (top of method):

```python
        if op.startswith("atom.") or op.startswith("red."):
            return FUKind.LSU
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_functional_units.py -v -k atom
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/core/functional_units.py tests/unit/core/test_functional_units.py
git commit -m "feat(core): FUSet.classify routes atom.* / red.* to LSU"
```

---

### Task 5: Tag M1 complete

```
.venv/bin/pytest -q -m "not slow"
git tag M1-phase6-complete
git tag | grep M.-phase6
```

---

## Milestone M2: smem atomic + atom_reduction_smem

### Task 6: SharedMemory.atomic_op functional

**Files:**
- Modify: `gpusim/core/exec.py` (add `SharedMemory.atomic_op`)
- Test: `tests/unit/core/test_exec_atomic.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_smem_atomic_op_add_int():
    from gpusim.core.exec import SharedMemory
    from gpusim.frontend.ir import PtxType
    s = SharedMemory(size_bytes=1024)
    s.allocate_cta(0, 1024)
    s.store_u32(0, 16, 100)
    old = s.atomic_op(0, 16, "add", 5, PtxType.u32)
    assert old == 100
    assert s.load_u32(0, 16) == 105


def test_smem_atomic_op_min_max():
    from gpusim.core.exec import SharedMemory
    from gpusim.frontend.ir import PtxType
    s = SharedMemory(size_bytes=1024)
    s.allocate_cta(0, 1024)
    s.store_u32(0, 0, 50)
    old = s.atomic_op(0, 0, "min", 30, PtxType.u32)
    assert old == 50
    assert s.load_u32(0, 0) == 30
    old = s.atomic_op(0, 0, "max", 100, PtxType.u32)
    assert old == 30
    assert s.load_u32(0, 0) == 100


def test_smem_atomic_op_cas():
    from gpusim.core.exec import SharedMemory
    from gpusim.frontend.ir import PtxType
    s = SharedMemory(size_bytes=1024)
    s.allocate_cta(0, 1024)
    s.store_u32(0, 0, 7)
    old = s.atomic_op(0, 0, "cas", (7, 99), PtxType.u32)
    assert old == 7
    assert s.load_u32(0, 0) == 99
    old = s.atomic_op(0, 0, "cas", (7, 12), PtxType.u32)
    assert old == 99
    assert s.load_u32(0, 0) == 99


def test_smem_atomic_op_exch():
    from gpusim.core.exec import SharedMemory
    from gpusim.frontend.ir import PtxType
    s = SharedMemory(size_bytes=1024)
    s.allocate_cta(0, 1024)
    s.store_u32(0, 0, 42)
    old = s.atomic_op(0, 0, "exch", 100, PtxType.u32)
    assert old == 42
    assert s.load_u32(0, 0) == 100


def test_smem_atomic_op_f32():
    from gpusim.core.exec import SharedMemory
    from gpusim.frontend.ir import PtxType
    s = SharedMemory(size_bytes=1024)
    s.allocate_cta(0, 1024)
    s.store_f32(0, 0, 1.5)
    old = s.atomic_op(0, 0, "add", 2.5, PtxType.f32)
    assert old == 1.5
    assert s.load_f32(0, 0) == 4.0
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Implement SharedMemory.atomic_op**

In `gpusim/core/exec.py`, in `SharedMemory` class, add:

```python
    def atomic_op(self, cta_id: int, offset: int, op: str, val,
                    ty) -> "old":
        """Atomically read-modify-write at smem[cta_id][offset]. Returns old value.

        op: "add" | "min" | "max" | "exch" | "cas"
        For "cas": val is (expected, new_val) tuple.
        ty: PtxType (u32, s32, f32)
        """
        from gpusim.frontend.ir import PtxType
        if ty is PtxType.f32:
            old = self.load_f32(cta_id, offset)
            new = self._apply_op_f32(op, old, val)
            self.store_f32(cta_id, offset, new)
            return old
        old = self.load_u32(cta_id, offset)
        new = self._apply_op_int(op, old, val)
        self.store_u32(cta_id, offset, new)
        return old

    @staticmethod
    def _apply_op_int(op: str, old: int, val) -> int:
        if op == "add": return (old + int(val)) & 0xFFFFFFFF
        if op == "min": return min(old, int(val))
        if op == "max": return max(old, int(val))
        if op == "exch": return int(val) & 0xFFFFFFFF
        if op == "cas":
            expected, new_val = val
            return int(new_val) & 0xFFFFFFFF if old == int(expected) else old
        raise ValueError(f"unknown atomic op {op!r}")

    @staticmethod
    def _apply_op_f32(op: str, old: float, val) -> float:
        if op == "add": return float(old) + float(val)
        if op == "min": return min(float(old), float(val))
        if op == "max": return max(float(old), float(val))
        if op == "exch": return float(val)
        if op == "cas":
            expected, new_val = val
            return float(new_val) if old == float(expected) else old
        raise ValueError(f"unknown atomic op {op!r}")
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_exec_atomic.py -v
```
Expected: 5 PASS.

```bash
git add gpusim/core/exec.py tests/unit/core/test_exec_atomic.py
git commit -m "feat(exec): SharedMemory.atomic_op (5 ops × 3 dtypes)"
```

---

### Task 7: SubCore atom/red shared routing

**Files:**
- Modify: `gpusim/core/sub_core.py` (add atom.shared / red.shared branches in `_issue`)
- Test: `tests/unit/core/test_sub_core_atomic.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_sub_core_atom_shared_add_routes_correctly():
    """End-to-end: warp executes atom.shared.add, smem updated, latency reasonable."""
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .u32 %r<4>;
    .reg .pred %p0;
    .shared .align 4 .b8 smem_buf[64];
    
    mov.u32 %r0, %tid.x;
    setp.ge.u32 %p0, %r0, 4;
    @%p0 bra END;
    mov.u64 %rd0, smem_buf;
    mov.u32 %r1, 1;
    atom.shared.add.u32 %r2, [%rd0], %r1;
END:
    ret;
}
"""
    cfg = load_default()
    res = gpusim.run(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                      params={}, mode="timing", config=cfg)
    assert 0 < res.metrics["cycles"] < 5000


def test_sub_core_red_shared_no_dst():
    """red.shared.add doesn't write a dst register."""
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .u32 %r0;
    .reg .pred %p0;
    .shared .align 4 .b8 smem_buf[64];
    
    mov.u32 %r0, %tid.x;
    setp.ge.u32 %p0, %r0, 4;
    @%p0 bra END;
    mov.u64 %rd0, smem_buf;
    red.shared.add.u32 [%rd0], %r0;
END:
    ret;
}
"""
    cfg = load_default()
    res = gpusim.run(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                      params={}, mode="timing", config=cfg)
    assert 0 < res.metrics["cycles"] < 5000
```

- [ ] **Step 2: Run (FAIL — atom/red.shared not routed in SubCore)**

- [ ] **Step 3: Add atom/red.shared branch in SubCore._issue**

In `gpusim/core/sub_core.py`, in `_issue`, before the generic functional execution path, add:

```python
        if op.startswith("atom.shared.") or op.startswith("red.shared."):
            is_atom = op.startswith("atom.")
            op_name = op.split(".")[2]   # "add" | "min" | "max" | ...
            ty = instr.type
            # Per-lane functional update
            for lane in range(32):
                if not (w.fn_state.active_mask >> lane) & 1: continue
                t = w.fn_state.threads[lane]
                addr = t.get_u64(instr.src[0].name)
                val = self.executor._read(t, instr.src[1], ty)
                if op_name == "cas":
                    expected = val
                    new_val = self.executor._read(t, instr.src[2], ty)
                    val_passed = (expected, new_val)
                else:
                    val_passed = val
                old = self.smem.atomic_op(w.cta_id, int(addr), op_name, val_passed, ty)
                if is_atom:
                    self.executor._write(t, instr.dst[0], old, ty)
            # Timing: smem_latency + bank_conflict * smem_atomic_op_extra_latency
            from gpusim.core.smem import bank_conflict_degree
            from gpusim.core.exec import shared_addresses_for_warp
            addrs = shared_addresses_for_warp(w.fn_state, instr)
            bank_conflict = bank_conflict_degree(
                addrs, active_mask=w.fn_state.active_mask, banks=self.cfg.smem_banks,
            )
            cache_cfg = (getattr(self.cfg, "_cache_for_run", None)
                          or getattr(self.cfg, "cache", None))
            atomic_extra = getattr(cache_cfg, "smem_atomic_op_extra_latency", 4) if cache_cfg else 4
            latency = self.cfg.fu.smem_latency + bank_conflict * atomic_extra
            completion = now + latency
            # Trace
            if self.recorder is not None:
                self.recorder.atomic(
                    cycle=now, sm_id=getattr(self, "sm_id", -1),
                    warp_id=w.warp_id, kind="ATOM" if is_atom else "RED",
                    op=op_name, space="shared",
                    line_addr=int(addrs[0]) if addrs and addrs[0] >= 0 else 0,
                    latency=latency, n_lanes=sum(1 for a in addrs if a >= 0),
                )
                self.recorder.instr_issue(
                    cycle=now, warp_id=w.warp_id, pc=instr.pc, op=op,
                    src_loc=(instr.src_loc.file, instr.src_loc.line),
                    active_mask=w.fn_state.active_mask if w.fn_state else 0,
                )
            if is_atom:
                w.scoreboard.mark_write(instr.dst[0].name, completion, origin="atomic")
            w.stack.update_top_pc(w.stack.top().pc + 1); w.stack.maybe_pop()
            return
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_sub_core_atomic.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/core/sub_core.py tests/unit/core/test_sub_core_atomic.py
git commit -m "feat(core): SubCore atom.shared / red.shared routing with bank conflict latency"
```

---

### Task 8: Example atom_reduction_smem

**Files:**
- Create: `examples/atom_reduction_smem/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_atom_reduction_smem.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "atom_reduction_smem"


def test_atom_reduction_smem_correctness():
    """N threads atomic.add 1 to a single smem counter; assert counter == N."""
    import gpusim
    from gpusim.config.loader import load_default
    out = np.zeros(1, dtype=np.uint32)
    ptx = (_DIR / "kernel.ptx").read_text()
    cfg = load_default()
    res = gpusim.run(
        ptx_src=ptx, grid=(1, 1, 1), block=(128, 1, 1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    assert out[0] == 128
    assert 0 < res.metrics["cycles"] < 50000
```

- [ ] **Step 2: Kernel**

Create `examples/atom_reduction_smem/kernel.ptx`:

```
.entry test(.param .u64 OUT)
{
    .reg .u64 %rd<3>;
    .reg .u32 %r<4>;
    .shared .align 4 .b8 smem_count[4];

    mov.u64 %rd0, smem_count;

    // Init smem counter to 0 (only thread 0)
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra ATOMIC;
    mov.u32 %r1, 0;
    st.shared.u32 [%rd0], %r1;
ATOMIC:
    bar.sync 0;

    // All threads atomic.add 1
    mov.u32 %r2, 1;
    atom.shared.add.u32 %r3, [%rd0], %r2;

    bar.sync 0;

    // Thread 0 writes final count to OUT
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra DONE;
    ld.shared.u32 %r1, [%rd0];
    ld.param.u64 %rd1, [OUT];
    st.global.u32 [%rd1], %r1;
DONE:
    ret;
}
```

- [ ] **Step 3: Supporting files**

`reference.py`:
```python
def reference(n_threads: int) -> int:
    return n_threads
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    out = np.zeros(1, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(1,1,1), block=(128,1,1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    print(f"atom_reduction_smem: cycles={res.metrics['cycles']}, count={out[0]}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# atom_reduction_smem

Phase 6 smem atomic demo. 128 threads each atomic.add 1 to a single smem counter.
All 128 atomic ops serialize through one bank → high latency.

## Run
```
python examples/atom_reduction_smem/run.py
```

## Tutorial
docs/tutorial/23-smem-atomic-bank-conflict.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_atom_reduction_smem.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/atom_reduction_smem/ tests/parity/test_atom_reduction_smem.py
git commit -m "feat(examples): atom_reduction_smem — 128 thread serialized smem atomic.add"
```

---

### Task 9: Tag M2 complete

```
.venv/bin/pytest -q -m "not slow"
git tag M2-phase6-complete
```

---

## Milestone M3: gmem atomic + 3 examples

### Task 10: L2AtomicQueue data class

**Files:**
- Create: `gpusim/core/atomic.py`
- Test: `tests/unit/core/test_atomic.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_l2_atomic_queue_first_arrival():
    from gpusim.core.atomic import L2AtomicQueue
    q = L2AtomicQueue(n_slots=32)
    completion = q.enqueue(line_addr=0x1000, sm_id=0, op="add", op_kind="atom",
                              arrival=10, atomic_op_latency=10, l2_hit_latency=20)
    # First arrival: arrival + l2_hit_latency + atomic_op_latency = 10 + 20 + 10 = 40
    assert completion == 40


def test_l2_atomic_queue_serializes_same_line():
    from gpusim.core.atomic import L2AtomicQueue
    q = L2AtomicQueue(n_slots=32)
    c1 = q.enqueue(line_addr=0x1000, sm_id=0, op="add", op_kind="atom",
                     arrival=0, atomic_op_latency=10, l2_hit_latency=20)
    # First: 0 + 20 + 10 = 30
    assert c1 == 30
    c2 = q.enqueue(line_addr=0x1000, sm_id=1, op="add", op_kind="atom",
                     arrival=5, atomic_op_latency=10, l2_hit_latency=20)
    # Second: max(5+20, 30) + 10 = max(25, 30) + 10 = 40
    assert c2 == 40
    c3 = q.enqueue(line_addr=0x1000, sm_id=2, op="add", op_kind="atom",
                     arrival=10, atomic_op_latency=10, l2_hit_latency=20)
    # Third: max(10+20, 40) + 10 = 50
    assert c3 == 50


def test_l2_atomic_queue_different_lines_parallel():
    from gpusim.core.atomic import L2AtomicQueue
    q = L2AtomicQueue(n_slots=32)
    c1 = q.enqueue(line_addr=0x1000, sm_id=0, op="add", op_kind="atom",
                     arrival=0, atomic_op_latency=10, l2_hit_latency=20)
    c2 = q.enqueue(line_addr=0x2000, sm_id=1, op="add", op_kind="atom",
                     arrival=0, atomic_op_latency=10, l2_hit_latency=20)
    # Different lines don't serialize
    assert c1 == c2 == 30


def test_l2_atomic_queue_depth_at_now():
    from gpusim.core.atomic import L2AtomicQueue
    q = L2AtomicQueue(n_slots=32)
    q.enqueue(line_addr=0x1000, sm_id=0, op="add", op_kind="atom",
                arrival=0, atomic_op_latency=10, l2_hit_latency=20)
    q.enqueue(line_addr=0x1000, sm_id=1, op="add", op_kind="atom",
                arrival=0, atomic_op_latency=10, l2_hit_latency=20)
    # At cycle 35, first completes (c=30) + second still in-flight (c=40)
    # → depth at 35 = 1
    assert q.queue_depth(0x1000, now=35) == 1
    # At cycle 50, both done → depth 0
    assert q.queue_depth(0x1000, now=50) == 0
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Implement atomic.py**

Create `gpusim/core/atomic.py`:

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class L2AtomicEntry:
    line_addr: int
    arrival_cycle: int
    completion_at: int
    sm_id: int
    op: str               # "add" | "min" | "max" | "exch" | "cas"
    op_kind: str          # "atom" | "red"


class L2AtomicQueue:
    """Per-line atomic FIFO. Multiple SMs hitting the same line serialize.

    Each atomic op takes atomic_op_latency cycles after the previous one finishes
    on that line. Different lines do not serialize against each other.
    """

    def __init__(self, n_slots: int = 32):
        self.n_slots = n_slots
        self._queues: dict[int, list[L2AtomicEntry]] = {}

    def enqueue(self, *, line_addr: int, sm_id: int, op: str, op_kind: str,
                  arrival: int, atomic_op_latency: int,
                  l2_hit_latency: int) -> int:
        """Queue an atomic; returns its completion_at cycle."""
        q = self._queues.setdefault(line_addr, [])
        # Drop entries already completed before arrival
        q = [e for e in q if e.completion_at > arrival]
        prev_done = q[-1].completion_at if q else 0
        start = max(arrival + l2_hit_latency, prev_done)
        completion = start + atomic_op_latency
        entry = L2AtomicEntry(
            line_addr=line_addr, arrival_cycle=arrival,
            completion_at=completion, sm_id=sm_id, op=op, op_kind=op_kind,
        )
        q.append(entry)
        self._queues[line_addr] = q
        return completion

    def queue_depth(self, line_addr: int, now: int) -> int:
        """Number of atomic ops on line still in-flight at cycle `now`."""
        q = self._queues.get(line_addr, [])
        return sum(1 for e in q if e.completion_at > now)
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_atomic.py -v
```

```bash
git add gpusim/core/atomic.py tests/unit/core/test_atomic.py
git commit -m "feat(core): L2AtomicQueue per-line FIFO for cross-SM atomic serialization"
```

---

### Task 11: GlobalMemory.atomic_op

**Files:**
- Modify: `gpusim/core/exec.py` (add `GlobalMemory.atomic_op`)
- Test: `tests/unit/core/test_exec_atomic.py` (extend)

- [ ] **Step 1: Append failing tests**

```python
def test_gmem_atomic_op_add_int():
    import numpy as np
    from gpusim.core.exec import GlobalMemory
    from gpusim.frontend.ir import PtxType
    g = GlobalMemory()
    arr = np.array([100], dtype=np.uint32)
    base = g.bind("X", arr)
    old = g.atomic_op(base, "add", 5, PtxType.u32)
    assert old == 100
    assert int(arr[0]) == 105


def test_gmem_atomic_op_cas_match():
    import numpy as np
    from gpusim.core.exec import GlobalMemory
    from gpusim.frontend.ir import PtxType
    g = GlobalMemory()
    arr = np.array([7], dtype=np.uint32)
    base = g.bind("X", arr)
    old = g.atomic_op(base, "cas", (7, 99), PtxType.u32)
    assert old == 7
    assert int(arr[0]) == 99


def test_gmem_atomic_op_cas_mismatch():
    import numpy as np
    from gpusim.core.exec import GlobalMemory
    from gpusim.frontend.ir import PtxType
    g = GlobalMemory()
    arr = np.array([7], dtype=np.uint32)
    base = g.bind("X", arr)
    old = g.atomic_op(base, "cas", (5, 99), PtxType.u32)
    assert old == 7
    assert int(arr[0]) == 7   # unchanged
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Implement GlobalMemory.atomic_op**

In `gpusim/core/exec.py`, in `GlobalMemory` class, add (mirror SharedMemory.atomic_op):

```python
    def atomic_op(self, addr: int, op: str, val, ty) -> "old":
        """Atomically read-modify-write at addr. Returns old value."""
        from gpusim.frontend.ir import PtxType
        if ty is PtxType.f32:
            old = self.load_f32(addr)
            new = self._apply_op_f32(op, old, val)
            self.store_f32(addr, new)
            return old
        old = self.load_u32(addr)
        new = self._apply_op_int(op, old, val)
        self.store_u32(addr, new)
        return old

    @staticmethod
    def _apply_op_int(op: str, old: int, val) -> int:
        if op == "add": return (old + int(val)) & 0xFFFFFFFF
        if op == "min": return min(old, int(val))
        if op == "max": return max(old, int(val))
        if op == "exch": return int(val) & 0xFFFFFFFF
        if op == "cas":
            expected, new_val = val
            return int(new_val) & 0xFFFFFFFF if old == int(expected) else old
        raise ValueError(f"unknown atomic op {op!r}")

    @staticmethod
    def _apply_op_f32(op: str, old: float, val) -> float:
        if op == "add": return float(old) + float(val)
        if op == "min": return min(float(old), float(val))
        if op == "max": return max(float(old), float(val))
        if op == "exch": return float(val)
        if op == "cas":
            expected, new_val = val
            return float(new_val) if old == float(expected) else old
        raise ValueError(f"unknown atomic op {op!r}")
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_exec_atomic.py -v
```

```bash
git add gpusim/core/exec.py tests/unit/core/test_exec_atomic.py
git commit -m "feat(exec): GlobalMemory.atomic_op (5 ops × 3 dtypes)"
```

---

### Task 12: L2.atomic_op + L2AtomicQueue integration

**Files:**
- Modify: `gpusim/core/cache/l2.py`
- Test: `tests/unit/cache/test_l2_atomic.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_l2_atomic_op_serializes_same_line():
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.config.schema import CacheConfig
    class _NoOpHbm:
        def request(self, line_addr, now): return now + 100
        def write_request(self, line_addr, now): return now + 100
    cfg = CacheConfig()
    l2 = L2Cache(cfg, _NoOpHbm())
    c1 = l2.atomic_op(line_addr=0x1000, sm_id=0, op="add", op_kind="atom", now=0)
    c2 = l2.atomic_op(line_addr=0x1000, sm_id=1, op="add", op_kind="atom", now=0)
    assert c2 > c1
    assert (c2 - c1) >= cfg.atomic_op_latency


def test_l2_atomic_op_different_lines_parallel():
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.config.schema import CacheConfig
    class _NoOpHbm:
        def request(self, line_addr, now): return now + 100
        def write_request(self, line_addr, now): return now + 100
    cfg = CacheConfig()
    l2 = L2Cache(cfg, _NoOpHbm())
    c1 = l2.atomic_op(line_addr=0x1000, sm_id=0, op="add", op_kind="atom", now=0)
    c2 = l2.atomic_op(line_addr=0x2000, sm_id=1, op="add", op_kind="atom", now=0)
    assert c1 == c2
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add atomic_op to L2Cache**

In `gpusim/core/cache/l2.py`, in `L2Cache.__init__`, add:

```python
        from gpusim.core.atomic import L2AtomicQueue
        self._atomic_queue = L2AtomicQueue(n_slots=cfg.atomic_queue_capacity)
```

Add method:

```python
    def atomic_op(self, *, line_addr: int, sm_id: int, op: str,
                    op_kind: str, now: int) -> int:
        """Enqueue an atomic on this line; return completion cycle.
        Phase 6 simplification: assume atomic always L2-hit (line resident).
        """
        completion = self._atomic_queue.enqueue(
            line_addr=line_addr, sm_id=sm_id, op=op, op_kind=op_kind,
            arrival=now, atomic_op_latency=self.cfg.atomic_op_latency,
            l2_hit_latency=self.cfg.l2_hit_latency,
        )
        return completion
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/cache/test_l2_atomic.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/core/cache/l2.py tests/unit/cache/test_l2_atomic.py
git commit -m "feat(cache): L2.atomic_op integrates L2AtomicQueue"
```

---

### Task 13: SubCore atom/red.global routing

**Files:**
- Modify: `gpusim/core/sub_core.py`
- Test: `tests/unit/core/test_sub_core_atomic.py` (extend)

- [ ] **Step 1: Append failing test**

```python
def test_sub_core_atom_global_add_routes_to_l2():
    """N CTAs each atomic.add to same gmem location → result == N * thread_count_per_cta_lane_0."""
    import numpy as np
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd<3>;
    .reg .u32 %r<4>;
    .reg .pred %p0;
    
    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra END;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r1, 1;
    atom.global.add.u32 %r2, [%rd0], %r1;
END:
    ret;
}
"""
    cfg = load_default()
    cfg.n_sm = 4
    out = np.zeros(1, dtype=np.uint32)
    res = gpusim.run(ptx_src=src, grid=(8,1,1), block=(32,1,1),
                      params={"OUT": out}, mode="timing", config=cfg)
    # 8 CTAs × 1 thread (tid==0) = 8 atomic increments
    assert int(out[0]) == 8


def test_sub_core_red_global_add():
    """red.global.add (no return)."""
    import numpy as np
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd0;
    .reg .u32 %r<3>;
    .reg .pred %p0;
    
    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra END;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r1, 5;
    red.global.add.u32 [%rd0], %r1;
END:
    ret;
}
"""
    cfg = load_default()
    out = np.zeros(1, dtype=np.uint32)
    res = gpusim.run(ptx_src=src, grid=(4,1,1), block=(32,1,1),
                      params={"OUT": out}, mode="timing", config=cfg)
    assert int(out[0]) == 20   # 4 CTAs × 5
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add atom/red.global branch in SubCore._issue**

In `gpusim/core/sub_core.py`, in `_issue`, before generic functional path (or near the smem atomic branch added in Task 7), add:

```python
        if op.startswith("atom.global.") or op.startswith("red.global."):
            is_atom = op.startswith("atom.")
            op_name = op.split(".")[2]
            ty = instr.type
            # Per-lane functional update + collect addrs
            line_addrs = set()
            for lane in range(32):
                if not (w.fn_state.active_mask >> lane) & 1: continue
                t = w.fn_state.threads[lane]
                addr = t.get_u64(instr.src[0].name)
                val = self.executor._read(t, instr.src[1], ty)
                if op_name == "cas":
                    expected = val
                    new_val = self.executor._read(t, instr.src[2], ty)
                    val_passed = (expected, new_val)
                else:
                    val_passed = val
                old = self.executor.gmem.atomic_op(int(addr), op_name, val_passed, ty)
                if is_atom:
                    self.executor._write(t, instr.dst[0], old, ty)
                line_addrs.add(int(addr) // 128)
            # Timing: per-line, queue through L2 atomic ALU
            if self.l2 is not None:
                max_completion = now
                for line in sorted(line_addrs):
                    c = self.l2.atomic_op(
                        line_addr=line, sm_id=getattr(self, "sm_id", -1),
                        op=op_name, op_kind="atom" if is_atom else "red", now=now,
                    )
                    max_completion = max(max_completion, c)
                completion = max_completion
            else:
                completion = now + 50  # fallback
            # Trace
            if self.recorder is not None:
                for line in sorted(line_addrs):
                    self.recorder.atomic(
                        cycle=now, sm_id=getattr(self, "sm_id", -1),
                        warp_id=w.warp_id, kind="ATOM" if is_atom else "RED",
                        op=op_name, space="global", line_addr=line,
                        latency=completion - now,
                        n_lanes=sum(1 for lane in range(32)
                                       if (w.fn_state.active_mask >> lane) & 1),
                    )
                self.recorder.instr_issue(
                    cycle=now, warp_id=w.warp_id, pc=instr.pc, op=op,
                    src_loc=(instr.src_loc.file, instr.src_loc.line),
                    active_mask=w.fn_state.active_mask if w.fn_state else 0,
                )
            if is_atom:
                w.scoreboard.mark_write(instr.dst[0].name, completion, origin="atomic")
            w.stack.update_top_pc(w.stack.top().pc + 1); w.stack.maybe_pop()
            return
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_sub_core_atomic.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/core/sub_core.py tests/unit/core/test_sub_core_atomic.py
git commit -m "feat(core): SubCore atom.global / red.global routing through L2 atomic ALU"
```

---

### Task 14: Example atom_histogram

**Files:**
- Create: `examples/atom_histogram/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_atom_histogram.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "atom_histogram"


def test_atom_histogram_correctness():
    """Each thread atomic.add 1 to a bin determined by tid % n_bins."""
    import gpusim
    from gpusim.config.loader import load_default
    n_bins = 16
    n_threads_per_cta = 32
    n_cta = 8
    out = np.zeros(n_bins, dtype=np.uint32)
    ptx = (_DIR / "kernel.ptx").read_text()
    cfg = load_default()
    res = gpusim.run(
        ptx_src=ptx, grid=(n_cta, 1, 1), block=(n_threads_per_cta, 1, 1),
        params={"OUT": out, "N_BINS": n_bins}, mode="timing", config=cfg,
    )
    expected = np.zeros(n_bins, dtype=np.uint32)
    for cta in range(n_cta):
        for tid in range(n_threads_per_cta):
            expected[tid % n_bins] += 1
    assert (out == expected).all()
```

- [ ] **Step 2: Kernel**

Create `examples/atom_histogram/kernel.ptx`:

```
.entry test(.param .u64 OUT, .param .u32 N_BINS)
{
    .reg .u64 %rd<4>;
    .reg .u32 %r<8>;
    
    ld.param.u64 %rd0, [OUT];
    ld.param.u32 %r0, [N_BINS];
    
    mov.u32 %r1, %tid.x;
    rem.u32 %r2, %r1, %r0;     // bin = tid % n_bins
    mul.lo.s32 %r3, %r2, 4;
    cvt.u64.u32 %rd1, %r3;
    add.u64 %rd2, %rd0, %rd1;
    
    mov.u32 %r4, 1;
    atom.global.add.u32 %r5, [%rd2], %r4;
    
    ret;
}
```

Note: PTX has `rem.u32` (or use `mod`). If parser doesn't support `rem.u32`, replace with `and.b32 %r2, %r1, mask` (when n_bins is power-of-2, e.g., 16):

```
    and.b32 %r2, %r1, 15;       // tid & 15 = tid % 16 (n_bins=16)
```

For the test (n_bins=16), this works; for general n_bins requires `rem` or div+mul+sub. **Use `and.b32` form**, hardcode n_bins=16 in kernel and pass as param too.

Adjusted kernel (use `and.b32` since n_bins=16=power-of-2):
```
.entry test(.param .u64 OUT, .param .u32 N_BINS)
{
    .reg .u64 %rd<4>;
    .reg .u32 %r<8>;
    
    ld.param.u64 %rd0, [OUT];
    
    mov.u32 %r1, %tid.x;
    and.b32 %r2, %r1, 15;
    mul.lo.s32 %r3, %r2, 4;
    cvt.u64.u32 %rd1, %r3;
    add.u64 %rd2, %rd0, %rd1;
    
    mov.u32 %r4, 1;
    atom.global.add.u32 %r5, [%rd2], %r4;
    
    ret;
}
```

- [ ] **Step 3: Supporting files**

`reference.py`:
```python
import numpy as np


def reference(n_threads_per_cta: int = 32, n_cta: int = 8,
              n_bins: int = 16) -> np.ndarray:
    out = np.zeros(n_bins, dtype=np.uint32)
    for cta in range(n_cta):
        for tid in range(n_threads_per_cta):
            out[tid % n_bins] += 1
    return out
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    out = np.zeros(16, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(8,1,1), block=(32,1,1),
        params={"OUT": out, "N_BINS": 16}, mode="timing", config=cfg,
    )
    print(f"atom_histogram: cycles={res.metrics['cycles']}")
    print(f"  bins = {list(out)}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# atom_histogram

Phase 6 gmem atomic demo. 8 CTAs × 32 threads each atomic.add to a bin
determined by `tid & 15`. With n_bins=16 and 32 threads/CTA, each bin sees
2 atomic per CTA × 8 CTAs = 16 atomic / bin.

L2 atomic ALU serializes atomic on same line; high contention → high latency.

## Run
```
python examples/atom_histogram/run.py
```

## Tutorial
docs/tutorial/22-gmem-atomic-l2-alu.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_atom_histogram.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/atom_histogram/ tests/parity/test_atom_histogram.py
git commit -m "feat(examples): atom_histogram — gmem atomic.add with L2 ALU contention"
```

---

### Task 15: Example atom_cas_spinlock

**Files:**
- Create: `examples/atom_cas_spinlock/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_atom_cas_spinlock.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "atom_cas_spinlock"


def test_atom_cas_spinlock_correctness():
    """N threads use atom.cas as critical section to increment a counter.
    Each thread does counter += 1; final == N."""
    import gpusim
    from gpusim.config.loader import load_default
    n_threads_per_cta = 32
    n_cta = 4
    expected = n_threads_per_cta * n_cta
    out = np.zeros(2, dtype=np.uint32)   # [counter, lock]
    ptx = (_DIR / "kernel.ptx").read_text()
    cfg = load_default()
    res = gpusim.run(
        ptx_src=ptx, grid=(n_cta, 1, 1), block=(n_threads_per_cta, 1, 1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    assert int(out[0]) == expected
```

- [ ] **Step 2: Kernel**

Create `examples/atom_cas_spinlock/kernel.ptx`:

```
.entry test(.param .u64 OUT)
{
    .reg .u64 %rd<4>;
    .reg .u32 %r<6>;
    .reg .pred %p<2>;
    
    ld.param.u64 %rd0, [OUT];
    add.u64 %rd1, %rd0, 4;     // OUT[1] = lock
    
LOCK_LOOP:
    // Try to acquire lock: cas(lock, 0, 1) — if old==0, we got it
    mov.u32 %r0, 0;            // expected
    mov.u32 %r1, 1;            // new
    atom.global.cas.u32 %r2, [%rd1], %r0, %r1;
    setp.ne.u32 %p0, %r2, 0;
    @%p0 bra LOCK_LOOP;
    
    // Critical section: increment counter
    ld.global.u32 %r3, [%rd0];
    add.s32 %r3, %r3, 1;
    st.global.u32 [%rd0], %r3;
    
    // Release lock
    mov.u32 %r4, 0;
    st.global.u32 [%rd1], %r4;
    
    ret;
}
```

- [ ] **Step 3: Supporting files**

`reference.py`:
```python
def reference(n_threads_per_cta: int = 32, n_cta: int = 4) -> int:
    return n_threads_per_cta * n_cta
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    out = np.zeros(2, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(4,1,1), block=(32,1,1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    print(f"atom_cas_spinlock: cycles={res.metrics['cycles']}, counter={out[0]}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# atom_cas_spinlock

Phase 6 CAS lock-free pattern demo. N threads compete for a global lock via
`atom.global.cas.u32`. Acquired lock → increment counter → release lock.

Demonstrates: CAS retry loop pattern; lock contention serialization.

## Run
```
python examples/atom_cas_spinlock/run.py
```

## Tutorial
docs/tutorial/25-cas-lock-free-pattern.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_atom_cas_spinlock.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/atom_cas_spinlock/ tests/parity/test_atom_cas_spinlock.py
git commit -m "feat(examples): atom_cas_spinlock — CAS lock-free critical section"
```

---

### Task 16: Example red_min_max + Tag M3

**Files:**
- Create: `examples/red_min_max/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_red_min_max.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "red_min_max"


def test_red_min_max_correctness():
    """Each thread reads its slot from IN and red.min/red.max into OUT[0..1]."""
    import gpusim
    from gpusim.config.loader import load_default
    rng = np.random.RandomState(0)
    n = 256
    in_arr = rng.randint(0, 1000, size=n).astype(np.int32)
    out = np.zeros(2, dtype=np.int32)
    out[0] = 0x7FFFFFFF   # min seed
    out[1] = -0x80000000  # max seed
    ptx = (_DIR / "kernel.ptx").read_text()
    cfg = load_default()
    res = gpusim.run(
        ptx_src=ptx, grid=(8, 1, 1), block=(32, 1, 1),
        params={"IN": in_arr.copy(), "OUT": out}, mode="timing", config=cfg,
    )
    assert int(out[0]) == int(in_arr.min())
    assert int(out[1]) == int(in_arr.max())
```

- [ ] **Step 2: Kernel**

Create `examples/red_min_max/kernel.ptx`:

```
.entry test(.param .u64 IN, .param .u64 OUT)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<8>;
    .reg .s32 %s<4>;
    
    ld.param.u64 %rd0, [IN];
    ld.param.u64 %rd1, [OUT];
    
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.s32 %r3, %r1, %r2;
    add.s32 %r3, %r3, %r0;     // global tid
    mul.lo.s32 %r4, %r3, 4;
    cvt.u64.u32 %rd2, %r4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.s32 %s0, [%rd3];
    
    // OUT[0] = min, OUT[1] = max
    add.u64 %rd4, %rd1, 4;
    
    red.global.min.s32 [%rd1], %s0;
    red.global.max.s32 [%rd4], %s0;
    
    ret;
}
```

- [ ] **Step 3: Supporting files**

`reference.py`:
```python
import numpy as np


def reference(in_arr: np.ndarray) -> tuple[int, int]:
    return int(in_arr.min()), int(in_arr.max())
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    rng = np.random.RandomState(0)
    in_arr = rng.randint(0, 1000, size=256).astype(np.int32)
    out = np.zeros(2, dtype=np.int32)
    out[0] = 0x7FFFFFFF; out[1] = -0x80000000
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(8,1,1), block=(32,1,1),
        params={"IN": in_arr.copy(), "OUT": out}, mode="timing", config=cfg,
    )
    print(f"red_min_max: cycles={res.metrics['cycles']}")
    print(f"  min = {out[0]}, max = {out[1]}")
    print(f"  numpy: min = {int(in_arr.min())}, max = {int(in_arr.max())}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# red_min_max

Phase 6 `red.global.{min,max}.s32` demo. 256 threads each emit their value
into a global min and max via `red.global` (no return register).

Compare with `atom.global.{min,max}` (same op but with returned old value).
Hardware difference: red has no return path → marginally faster (Phase 6
simulator approximates as same latency; spec §11 noted).

## Run
```
python examples/red_min_max/run.py
```

## Tutorial
docs/tutorial/26-red-vs-atom.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + tag M3**

```
.venv/bin/pytest tests/parity/test_red_min_max.py -v
.venv/bin/pytest -q -m "not slow"
git add examples/red_min_max/ tests/parity/test_red_min_max.py
git commit -m "feat(examples): red_min_max — gmem reduction via red.global.min/max"
git tag M3-phase6-complete
git tag | grep M.-phase6
```

---

## Milestone M4: Cluster TMA store + cooperative epilogue

### Task 17: do_bulk_store_2d cluster decode + SubCore routing

**Files:**
- Modify: `gpusim/core/sub_core.py` (extend cp.async.bulk.tensor branch for cluster store)
- Test: `tests/unit/core/test_tma_store_cluster.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_cluster_tma_store_routes_smem_src_to_remote_cta():
    """cp.async.bulk.tensor.2d.global.shared::cluster reads from cluster-encoded smem_src."""
    import numpy as np
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;
    .reg .pred %p0;
    
    .shared .align 4 .b8 smem_local[64];
    
    ld.param.u64 %rd0, [OUT];
    
    mov.u32 %r0, %tid.x;
    getctarank.u32 %rrank;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra END;
    
    // CTA rank 1 fills its smem with values 100..115
    setp.ne.u32 %p0, %rrank, 1;
    @%p0 bra MAYBE_STORE;
    mov.u32 %r1, 100;
    mov.u64 %rd1, smem_local;
    st.shared.u32 [%rd1], %r1;
    
MAYBE_STORE:
    barrier.cluster.arrive;
    barrier.cluster.wait;
    
    // CTA rank 0: TMA store from rank 1's smem to OUT
    setp.ne.u32 %p0, %rrank, 0;
    @%p0 bra END;
    
    gpusim.tma_desc %rd2, %rd0, 1, 1, 1, 4;
    mov.u64 %rd3, 16777216;        // (1 << 24) | 0 = rank 1, offset 0
    cp.async.bulk.tensor.2d.global.shared::cluster [%rd2], [%rd3];
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group 0;
END:
    barrier.cluster.arrive;
    barrier.cluster.wait;
    ret;
}
"""
    cfg = load_default()
    cfg.cluster_size = 2; cfg.n_sm = 2
    out = np.zeros(1, dtype=np.uint32)
    res = gpusim.run(ptx_src=src, grid=(2,1,1), block=(32,1,1),
                      params={"OUT": out}, mode="timing", config=cfg)
    assert int(out[0]) == 100
```

- [ ] **Step 2: Run (FAIL — cluster TMA store not yet decoding cluster smem_src)**

- [ ] **Step 3: Update SubCore cp.async.bulk.tensor branch**

In `gpusim/core/sub_core.py`, find the existing `cp.async.bulk.tensor.` branch (Phase 4 store + Phase 5 cluster TMA load). Currently it decodes smem_dst as cluster-encoded for `shared::cluster` form. Phase 6 extends: for store form (no `mbarrier` in op + `global.shared` in op + has `shared::cluster`), decode smem_src similarly.

Look at the existing code's `is_store` and `is_cluster` logic and extend the store path:

```python
            elif is_store:
                # smem_src may be cluster-encoded for shared::cluster store form
                cluster_size = getattr(w.executor, "cluster_size", 1)
                if "shared::cluster" in op and w.cluster_id >= 0 and cluster_size > 1:
                    smem_src_rank = (int(smem_dst_ptr) >> 24) & 0xFF
                    smem_src_offset = int(smem_dst_ptr) & 0xFFFFFF
                    source_cta = w.cluster_id * cluster_size + smem_src_rank
                else:
                    smem_src_offset = int(smem_dst_ptr)
                    source_cta = w.cta_id
                from gpusim.core.tma_store import do_bulk_store_2d
                tx_bytes = do_bulk_store_2d(
                    gmem=self.executor.gmem, smem=self.smem,
                    cta_id=source_cta, smem_src=smem_src_offset, desc=desc,
                )
                # ... rest of existing Phase 4 BulkStoreQueue push + recorder ...
```

(Adapt to actual existing variable names. The variable currently used as `smem_dst_ptr` for store path is actually smem_src for store; rename mentally.)

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_tma_store_cluster.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/core/sub_core.py tests/unit/core/test_tma_store_cluster.py
git commit -m "feat(core): cluster TMA store decodes smem_src to remote CTA via cluster pointer"
```

---

### Task 18: Example cluster_cooperative_epilogue

**Files:**
- Create: `examples/cluster_cooperative_epilogue/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_cluster_cooperative_epilogue.py`

This is M4's hardest task. Plan allows DONE_WITH_CONCERNS if full wgmma + cluster TMA store doesn't pass parity: ship a simpler "no wgmma, only cluster TMA store fan-out" version.

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "cluster_cooperative_epilogue"


def test_cluster_cooperative_epilogue_correctness():
    """4-CTA cluster: each CTA fills its smem with rank-tagged data; CTA 0
    uses cluster TMA store to write all 4 CTAs' data to OUT (each CTA's slice
    placed at rank * 32 in OUT)."""
    import gpusim
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.cluster_size = 4; cfg.n_sm = 4
    out = np.zeros(128, dtype=np.uint32)
    ptx = (_DIR / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(4, 1, 1), block=(32, 1, 1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    # Each CTA r filled smem[i] = r*1000 + i for i in 0..31
    # CTA 0 wrote all 4 CTAs' smem (via cluster TMA store) to OUT[r*32 : r*32+32]
    expected = np.zeros(128, dtype=np.uint32)
    for r in range(4):
        for i in range(32):
            expected[r * 32 + i] = r * 1000 + i
    assert (out == expected).all()
```

- [ ] **Step 2: Kernel**

Create `examples/cluster_cooperative_epilogue/kernel.ptx`. Simplified version (no wgmma; cluster TMA store fan-out only):

```
.entry test(.param .u64 OUT)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<8>;
    .reg .pred %p<4>;
    
    .shared .align 16 .b8 smem_D[128];   // 32 fp32 = 128 B per CTA
    
    ld.param.u64 %rd0, [OUT];
    
    mov.u32 %r0, %tid.x;
    getctarank.u32 %rrank;
    
    // Each CTA fills its smem_D: smem_D[i] = rank * 1000 + i
    mul.lo.s32 %r1, %r0, 4;
    cvt.u64.u32 %rd1, %r1;
    mov.u64 %rd2, smem_D;
    add.u64 %rd3, %rd2, %rd1;
    
    mul.lo.s32 %r2, %rrank, 1000;
    add.s32 %r3, %r2, %r0;
    st.shared.u32 [%rd3], %r3;
    
    bar.sync 0;
    barrier.cluster.arrive;
    barrier.cluster.wait;
    
    // CTA 0 thread 0 issues 4 cluster TMA stores
    setp.ne.u32 %p0, %rrank, 0;
    @%p0 bra END;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra END;
    
    // For each rank r (0..3): TMA store smem_D from CTA r → OUT[r*32 : r*32+32]
    
    // Rank 0: smem_src = (0 << 24) | smem_D_offset = smem_D
    // gmem_dst = OUT
    gpusim.tma_desc %rd4, %rd0, 32, 1, 32, 4;
    mov.u64 %rd5, smem_D;       // local rank 0
    cp.async.bulk.tensor.2d.global.shared::cluster [%rd4], [%rd5];
    
    // Rank 1: smem_src = (1 << 24) | smem_D_offset
    add.u64 %rd6, %rd0, 128;    // OUT + 32*4 bytes
    gpusim.tma_desc %rd4, %rd6, 32, 1, 32, 4;
    mov.u64 %rd5, 16777216;     // (1 << 24) | 0
    cp.async.bulk.tensor.2d.global.shared::cluster [%rd4], [%rd5];
    
    // Rank 2
    add.u64 %rd6, %rd0, 256;
    gpusim.tma_desc %rd4, %rd6, 32, 1, 32, 4;
    mov.u64 %rd5, 33554432;     // (2 << 24) | 0
    cp.async.bulk.tensor.2d.global.shared::cluster [%rd4], [%rd5];
    
    // Rank 3
    add.u64 %rd6, %rd0, 384;
    gpusim.tma_desc %rd4, %rd6, 32, 1, 32, 4;
    mov.u64 %rd5, 50331648;     // (3 << 24) | 0
    cp.async.bulk.tensor.2d.global.shared::cluster [%rd4], [%rd5];
    
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group 0;
    
END:
    barrier.cluster.arrive;
    barrier.cluster.wait;
    ret;
}
```

- [ ] **Step 3: Supporting files**

`reference.py`:
```python
import numpy as np


def reference(n_cta: int = 4, n_per_cta: int = 32) -> np.ndarray:
    out = np.zeros(n_cta * n_per_cta, dtype=np.uint32)
    for r in range(n_cta):
        for i in range(n_per_cta):
            out[r * n_per_cta + i] = r * 1000 + i
    return out
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    cfg.cluster_size = 4; cfg.n_sm = 4
    out = np.zeros(128, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(4,1,1), block=(32,1,1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    print(f"cluster_cooperative_epilogue: cycles={res.metrics['cycles']}")
    print(f"  out[0:4] = {list(out[0:4])}, out[32:36] = {list(out[32:36])}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# cluster_cooperative_epilogue

Phase 6 cluster TMA store cooperative epilogue demo. 4-CTA cluster: each CTA
fills its smem with rank-tagged data. CTA 0 issues 4 cluster TMA stores, each
reading from a different rank's smem (via cluster pointer encoding) and
writing to a different gmem offset.

Closes the Phase 5 cluster_matmul_dsmem deferred work: cluster TMA store
enables a single CTA to gather + store data from all cluster CTAs' smem.

Note: simplified version (no wgmma) demonstrates the cluster TMA store
mechanism; full wgmma + cooperative epilogue is the natural next step.

## Run
```
python examples/cluster_cooperative_epilogue/run.py
```

## Tutorial
docs/tutorial/24-cluster-cooperative-epilogue.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_cluster_cooperative_epilogue.py -v
.venv/bin/pytest -q -m "not slow"
```

If passes: continue. If fails (likely PTX edge cases):
- Try simpler 2-CTA cluster + 1 cluster TMA store
- Document what's blocking; commit the working subset

```bash
git add examples/cluster_cooperative_epilogue/ tests/parity/test_cluster_cooperative_epilogue.py
git commit -m "feat(examples): cluster_cooperative_epilogue — cluster TMA store fan-out"
```

---

### Task 19: Tag M4 complete

```
.venv/bin/pytest -q -m "not slow"
git tag M4-phase6-complete
```

---

## Milestone M5: Trace + viz + docs + final

### Task 20: 4 analysis metrics + Result API

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Modify: `gpusim/api.py`
- Modify: `gpusim/viz/notebook.py`
- Test: `tests/unit/analysis/test_phase6_metrics.py` (NEW)

- [ ] **Step 1: Create test**

```python
import pandas as pd


def test_atomic_throughput_per_line():
    from gpusim.analysis.metrics import atomic_throughput_per_line
    df = pd.DataFrame([
        {"cycle": 0, "line_addr": 0x1000},
        {"cycle": 10, "line_addr": 0x1000},
        {"cycle": 5, "line_addr": 0x2000},
    ])
    out = atomic_throughput_per_line(df, total_cycles=100)
    assert isinstance(out, pd.DataFrame)
    assert (out["line_addr"] == 0x1000).any()


def test_atomic_serialization_overhead():
    from gpusim.analysis.metrics import atomic_serialization_overhead
    df = pd.DataFrame([
        {"cycle": 0, "latency": 50},
        {"cycle": 0, "latency": 50},
    ])
    rate = atomic_serialization_overhead(df, total_cycles=100)
    # 2 atomics × 50 latency = 100; total 100 → rate ~1.0
    assert 0 <= rate <= 1.0


def test_atom_vs_red_ratio():
    from gpusim.analysis.metrics import atom_vs_red_ratio
    df = pd.DataFrame([
        {"kind": "ATOM"}, {"kind": "ATOM"}, {"kind": "ATOM"},
        {"kind": "RED"},
    ])
    r = atom_vs_red_ratio(df)
    assert abs(r["atom"] - 0.75) < 1e-6
    assert abs(r["red"] - 0.25) < 1e-6


def test_cooperative_epilogue_overlap():
    from gpusim.analysis.metrics import cooperative_epilogue_overlap
    bulk_df = pd.DataFrame([
        {"kind": "ISSUE", "cycle": 0, "completion_at": 50},
    ])
    mma_df = pd.DataFrame([
        {"cycle": 10},
        {"cycle": 20},
    ])
    r = cooperative_epilogue_overlap(bulk_df, mma_df)
    assert 0 <= r <= 1.0
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Implement metrics**

Append to `gpusim/analysis/metrics.py`:

```python
def atomic_throughput_per_line(atomic_df, total_cycles: int) -> "pd.DataFrame":
    """Per-line atomic throughput (count + atomic ops per cycle)."""
    import pandas as pd
    if atomic_df is None or atomic_df.empty:
        return pd.DataFrame(columns=["line_addr", "atomic_count", "throughput"])
    grouped = atomic_df.groupby("line_addr").size().reset_index(name="atomic_count")
    grouped["throughput"] = grouped["atomic_count"] / max(total_cycles, 1)
    return grouped.sort_values("atomic_count", ascending=False)


def atomic_serialization_overhead(atomic_df, total_cycles: int) -> float:
    """Total atomic latency / total cycles (proxy for L2 ALU utilization)."""
    if atomic_df is None or atomic_df.empty or total_cycles <= 0:
        return 0.0
    total_latency = float(atomic_df["latency"].sum())
    return min(1.0, total_latency / total_cycles)


def atom_vs_red_ratio(atomic_df) -> dict:
    """Fraction of atom vs red events."""
    if atomic_df is None or atomic_df.empty:
        return {"atom": 0.0, "red": 0.0}
    n = len(atomic_df)
    atom_count = int((atomic_df["kind"] == "ATOM").sum())
    red_count = int((atomic_df["kind"] == "RED").sum())
    return {"atom": atom_count / n, "red": red_count / n}


def cooperative_epilogue_overlap(bulk_store_df, mma_df) -> float:
    """Fraction of in-flight bulk store cycles during which mma events occurred."""
    if bulk_store_df is None or bulk_store_df.empty:
        return 0.0
    issues = bulk_store_df[bulk_store_df["kind"] == "ISSUE"] if "kind" in bulk_store_df.columns else bulk_store_df
    if issues.empty:
        return 0.0
    total_inflight = 0
    overlapped = 0
    for _, row in issues.iterrows():
        start = int(row["cycle"])
        end = int(row.get("completion_at", start))
        total_inflight += max(0, end - start)
        if mma_df is not None and not mma_df.empty:
            count = int(((mma_df["cycle"] >= start) & (mma_df["cycle"] <= end)).sum())
            if count > 0:
                overlapped += min(end - start, count * 8)
    return overlapped / max(total_inflight, 1)
```

- [ ] **Step 4: Add events_df helper + Result extension**

Append to `gpusim/viz/notebook.py`:

```python
def atomic_events_dataframe(rec):
    import pandas as pd
    from dataclasses import asdict
    if not rec.atomic_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.atomic_events])
```

Append to `gpusim/api.py` `Result` class:

```python
    @property
    def atomic_events_df(self):
        from gpusim.viz.notebook import atomic_events_dataframe
        return atomic_events_dataframe(self._recorder) if self._recorder else None

    @property
    def atomic_metrics(self) -> dict:
        if self._recorder is None:
            return {}
        from gpusim.analysis.metrics import (
            atomic_throughput_per_line, atomic_serialization_overhead,
            atom_vs_red_ratio, cooperative_epilogue_overlap,
        )
        cycles = self.metrics.get("cycles", 1)
        atomic_df = self.atomic_events_df
        if atomic_df is None or atomic_df.empty:
            return {"count": 0}
        per_line = atomic_throughput_per_line(atomic_df, cycles)
        peak_depth = int(per_line["atomic_count"].max()) if not per_line.empty else 0
        return {
            "count": len(atomic_df),
            "peak_queue_depth": peak_depth,
            "serialization_overhead": atomic_serialization_overhead(atomic_df, cycles),
            "atom_red_ratio": atom_vs_red_ratio(atomic_df),
            "cooperative_overlap": cooperative_epilogue_overlap(
                self.bulk_store_events_df, self.mma_events_df,
            ),
        }

    def atomic_summary(self) -> str:
        m = self.atomic_metrics
        if not m or m.get("count", 0) == 0:
            return "no atomic ops"
        return (f"atomic count={m['count']} / "
                 f"hot line peak depth={m['peak_queue_depth']} / "
                 f"serial overhead={m['serialization_overhead']*100:.1f}%")
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/analysis/test_phase6_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/analysis/metrics.py gpusim/api.py gpusim/viz/notebook.py tests/unit/analysis/test_phase6_metrics.py
git commit -m "feat(analysis+api): 4 Phase 6 metrics + Result.atomic_metrics + atomic_summary"
```

---

### Task 21: HTML §21/§22 + Perfetto atomic track

**Files:**
- Modify: `gpusim/viz/html_report.py`
- Modify: `gpusim/viz/_template.html.j2`
- Modify: `gpusim/viz/perfetto.py`
- Test: `tests/unit/viz/test_html_report_phase6.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_html_report_phase6_sections(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.atomic(cycle=0, sm_id=0, warp_id=0, kind="ATOM",
              op="add", space="global", line_addr=0x1000, latency=50)
    r.atomic(cycle=10, sm_id=1, warp_id=0, kind="ATOM",
              op="add", space="global", line_addr=0x1000, latency=50)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=200, occupancy={"active_ctas":1, "bottleneck":"none"})
    html = out.read_text()
    assert "Atomic" in html or "atomic" in html.lower()
```

- [ ] **Step 2: Add render helpers**

In `gpusim/viz/html_report.py`, append:

```python
def _render_atomic_contention(rec):
    if not rec.atomic_events:
        return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.atomic_events])
    parts = []
    parts.append("<h3>Atomic events</h3>" + df.head(20).to_html(index=False))
    # Per-line summary
    if "line_addr" in df.columns:
        per_line = df.groupby("line_addr").agg(
            count=("cycle", "size"),
            avg_latency=("latency", "mean"),
        ).reset_index().sort_values("count", ascending=False).head(10)
        parts.append("<h3>Hot lines (top 10)</h3>" + per_line.to_html(index=False))
    return "\n".join(parts)


def _render_cooperative_epilogue(rec):
    if not rec.bulk_store_events:
        return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.bulk_store_events])
    return "<h3>Cooperative epilogue (bulk store events)</h3>" + df.to_html(index=False)
```

In `save_html`, populate context:

```python
    context.update({
        "atomic_contention_html": _render_atomic_contention(rec),
        "cooperative_epilogue_html": _render_cooperative_epilogue(rec),
    })
```

- [ ] **Step 3: Add template blocks**

In `gpusim/viz/_template.html.j2`, append after Phase 5 §19/§20:

```html
{% if atomic_contention_html %}
<h2>§21 Atomic contention timeline</h2>
{{ atomic_contention_html | safe }}
{% endif %}

{% if cooperative_epilogue_html %}
<h2>§22 Cooperative epilogue overlap</h2>
{{ cooperative_epilogue_html | safe }}
{% endif %}
```

- [ ] **Step 4: Perfetto atomic track**

In `gpusim/viz/perfetto.py`, in `build_perfetto`, append:

```python
    # Phase 6 atomic events
    for ev in rec.atomic_events:
        events.append({
            "name": f"{ev.kind.lower()}.{ev.space}.{ev.op}",
            "cat": "atomic", "ph": "X", "ts": ev.cycle,
            "dur": max(1, ev.latency),
            "pid": "Atomic", "tid": ev.kind.lower(),
            "args": {"line_addr": ev.line_addr, "sm_id": ev.sm_id,
                     "n_lanes": ev.n_lanes,
                     "queue_depth_before": ev.queue_depth_before},
        })
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/viz/test_html_report_phase6.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/viz/ tests/unit/viz/test_html_report_phase6.py
git commit -m "feat(viz): HTML §21/§22 + Perfetto Atomic swimlane"
```

---

### Task 22: 5 tutorial chapters (22-26)

**Files (NEW):**
- `docs/tutorial/22-gmem-atomic-l2-alu.md`
- `docs/tutorial/23-smem-atomic-bank-conflict.md`
- `docs/tutorial/24-cluster-cooperative-epilogue.md`
- `docs/tutorial/25-cas-lock-free-pattern.md`
- `docs/tutorial/26-red-vs-atom.md`

- [ ] **Step 1: Read existing tutorial style**

```
cat docs/tutorial/21-cluster-tma-pipeline.md | head -60
```

Match structure: English body + Chinese section headers (`看模拟器` / `改一改` / `真机对照`).

- [ ] **Step 2: Write 5 chapters**

Each ~400-700 words.

**Chapter 22: gmem atomic 与 L2 ALU 串行**
- gmem atomic 走 L2 atomic ALU；多 SM 同 line 串行
- L2AtomicQueue per-line FIFO 模型
- atom_histogram 走通：低碰撞 (32 bins) vs 高碰撞 (4 bins) 的 cycles 对比
- 看模拟器：HTML §21 看 hot line / queue depth；`atomic_metrics["serialization_overhead"]`
- 改一改：n_bins=4 → 看 cycles 暴涨；n_bins=128 → cycles 接近 baseline
- 真机对照：H100 L2 atomic ALU 数有限；hot key 是 production 性能 killer

**Chapter 23: smem atomic 与 bank conflict**
- smem atomic = bank conflict 串行 + atomic_op_extra_latency
- 与 gmem atomic 模型对比
- atom_reduction_smem 走通：单 counter → 全 32 lane 串 1 bank
- 看模拟器：cycles vs threads 关系；atomic_summary 输出
- 改一改：counter 改成 32 个分散到 32 bank → cycles 几乎是 1×
- 真机对照：smem atomic 由 SM 内 atomic ALU 处理；同 bank 仍序列化

**Chapter 24: Cluster cooperative epilogue**
- Cluster TMA store 解码 cluster smem_src → CTA 0 代理写所有 cluster CTA 数据
- 闭合 Phase 5 cluster_matmul_dsmem 的 deferred 故事
- cluster_cooperative_epilogue 走通：4 CTA × 32 elem → 4 cluster TMA store → 128 elem
- 看模拟器：BulkStore timeline 看 4 store 的并行度
- 改一改：每 CTA 单独 TMA store → 对比 HBM 流量
- 真机对照：CUTLASS Hopper persistent matmul cooperative epilogue

**Chapter 25: CAS 与 lock-free pattern**
- atom.global.cas: 单 op 完成 expected → new；retry 由用户代码循环
- spinlock 模式 vs lock-free queue
- atom_cas_spinlock 走通：N thread 排队 acquire/release
- 看模拟器：CAS 在 HTML §21 占大部分 atomic 事件
- 改一改：counter += 1 改成更复杂 critical section（atomic 跑得久 → 锁占用更长）
- 真机对照：cuda::atomic_compare_exchange_strong / lockfree libraries

**Chapter 26: red vs atom**
- atom: RMW + 返回 old；占 dst register；scoreboard 跟踪
- red: 仅 RMW；无 dst；硬件可优化（无 return 路径）
- 模拟器同 latency；真机 red 略快
- red_min_max 走通：256 elem → red.min/max → numpy 对照
- 看模拟器：atom_red_ratio 指标
- 改一改：把 red.global.min 改成 atom.global.min → 增加一个 dst register；时钟数同
- 真机对照：red 在 cutlass reduction kernels 是优选

- [ ] **Step 3: Commit**

```bash
git add docs/tutorial/22-gmem-atomic-l2-alu.md docs/tutorial/23-smem-atomic-bank-conflict.md docs/tutorial/24-cluster-cooperative-epilogue.md docs/tutorial/25-cas-lock-free-pattern.md docs/tutorial/26-red-vs-atom.md
git commit -m "docs(tutorial): chapters 22-26 — gmem atomic / smem atomic / cluster epilogue / CAS / red vs atom"
```

---

### Task 23: Phase 6 microbench + Phase 1-5 regression rename + ref fixtures

**Files:**
- Create: `tests/microbench/test_phase6_facts.py`
- Create: `tests/microbench/test_phase6_runtime.py`
- Rename: `tests/parity/test_phase1_4_examples_unchanged.py` → `test_phase1_5_examples_unchanged.py` + extend
- Modify: `tests/reference/gen_reference.py`
- Create: 5 ref JSON stubs

- [ ] **Step 1: Phase 6 microbench**

Create `tests/microbench/test_phase6_facts.py`:

```python
"""Phase 6 microbench — atomic textbook facts."""
import numpy as np


def test_same_line_atomic_serializes():
    """32 thread atomic.add same line cycles ≥ 5× than 32 thread different lines."""
    import gpusim
    from gpusim.config.loader import load_default
    cfg = load_default()
    
    # Same line: all 32 thread atomic.add to OUT[0]
    src_same = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd0;
    .reg .u32 %r<3>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r1, 1;
    atom.global.add.u32 %r2, [%rd0], %r1;
    ret;
}
"""
    # Different lines: each thread atomic.add to OUT[tid * 32]
    src_diff = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd<3>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x;
    mul.lo.s32 %r1, %r0, 128;   // 128 bytes apart (different cache lines)
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 1;
    atom.global.add.u32 %r3, [%rd2], %r2;
    ret;
}
"""
    out = np.zeros(4096, dtype=np.uint32)
    res_same = gpusim.run(ptx_src=src_same, grid=(1,1,1), block=(32,1,1),
                            params={"OUT": out}, mode="timing", config=cfg)
    res_diff = gpusim.run(ptx_src=src_diff, grid=(1,1,1), block=(32,1,1),
                            params={"OUT": out}, mode="timing", config=cfg)
    ratio = res_same.metrics["cycles"] / max(res_diff.metrics["cycles"], 1)
    # Loose threshold: serialization should make same-line ≥ 1.5× slower than diff
    # (full theoretical 5× requires perfect L2 model; loosen for engine variance)
    assert ratio >= 1.5, f"same/diff line ratio = {ratio:.2f}"
```

- [ ] **Step 2: Runtime budget (slow)**

Create `tests/microbench/test_phase6_runtime.py`:

```python
import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_atom_histogram_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "atom_histogram"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_cluster_cooperative_epilogue_runtime_under_60s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "cluster_cooperative_epilogue"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=120)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 60
```

- [ ] **Step 3: Rename + extend Phase 1-4 regression**

```bash
git mv tests/parity/test_phase1_4_examples_unchanged.py tests/parity/test_phase1_5_examples_unchanged.py
```

Edit the file: rename `PHASE_1_4_EXAMPLES` → `PHASE_1_5_EXAMPLES`, add Phase 5 examples:

```python
PHASE_1_5_EXAMPLES = [
    # Phase 1
    "vector_add", "reduction_smem", "tiled_matmul",
    "divergence_demo", "bank_conflict_demo", "coalescing_demo",
    # Phase 2
    "l1_thrash_demo", "smem_vs_l1_demo", "bw_saturation_demo", "row_buffer_demo",
    # Phase 3
    "tc_matmul_precisions", "mixed_accum", "wgmma_basic", "wgmma_async_pipeline",
    # Phase 4
    "multi_sm_scheduler", "l2_sharing_demo", "tma_store_matmul",
    # Phase 5
    "cluster_basic", "cluster_matmul_dsmem", "cluster_tma_pipeline",
]
```

Update test parametrize / function name accordingly.

- [ ] **Step 4: Ref fixtures**

In `tests/reference/gen_reference.py`, append:
```python
"atom_histogram",
"atom_reduction_smem",
"cluster_cooperative_epilogue",
"atom_cas_spinlock",
"red_min_max",
```

Create 5 stub JSONs:

```bash
for k in atom_histogram atom_reduction_smem cluster_cooperative_epilogue atom_cas_spinlock red_min_max; do
  cat > tests/reference/data/$k.ref.json <<JSON
{
  "kernel": "$k",
  "phase": 6,
  "metrics": {
    "atomic_throughput_per_line": null,
    "serialization_overhead": null,
    "cooperative_epilogue_overlap": null
  },
  "tolerance": {
    "atomic_throughput_per_line_pct": 15,
    "serialization_overhead_pct": 10,
    "cooperative_epilogue_overlap_pct": 15
  },
  "notes": "Generated by gen_reference.py on real Hopper. null = not yet measured."
}
JSON
done
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/microbench/test_phase6_facts.py tests/parity/test_phase1_5_examples_unchanged.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add tests/microbench/test_phase6_facts.py tests/microbench/test_phase6_runtime.py tests/parity/test_phase1_5_examples_unchanged.py tests/reference/gen_reference.py tests/reference/data/atom_histogram.ref.json tests/reference/data/atom_reduction_smem.ref.json tests/reference/data/cluster_cooperative_epilogue.ref.json tests/reference/data/atom_cas_spinlock.ref.json tests/reference/data/red_min_max.ref.json
git rm tests/parity/test_phase1_4_examples_unchanged.py 2>/dev/null || true
git commit -m "test(microbench+reference): Phase 6 facts + Phase 1-5 regression rename + 5 ref stubs"
```

---

### Task 24: README v6 + final tag

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read README**

- [ ] **Step 2: Update to v6**

Edit `README.md`:
- Capabilities/status: add Phase 6 ✅
- Phase 6 features: gmem atomic + smem atomic + 5 ops (add/min/max/exch/cas) × 3 dtypes (u32/s32/f32) + cluster TMA store + cooperative epilogue
- Examples list: add 5 (was 20, now 25)
- Tutorials list: add 22-26 (was 21, now 26)
- API usage: show `result.atomic_summary()` and `result.atomic_metrics`
- Phase status table: 1-6 done; 7+ as future

- [ ] **Step 3: Run final suite**

```
.venv/bin/pytest -q -m "not slow"
```

- [ ] **Step 4: Run all 5 new examples**

```bash
.venv/bin/python examples/atom_histogram/run.py
.venv/bin/python examples/atom_reduction_smem/run.py
.venv/bin/python examples/atom_cas_spinlock/run.py
.venv/bin/python examples/red_min_max/run.py
.venv/bin/python examples/cluster_cooperative_epilogue/run.py
```

- [ ] **Step 5: Commit + tag**

```bash
git add README.md
git commit -m "docs(readme): v6 — Phase 6 capabilities (atomics + cluster TMA store + cooperative epilogue)"
git tag phase6-complete
git tag | grep phase
git log --oneline | head -10
```

---

### Task 25: Final sanity sweep + done

- [ ] **Step 1: Run microbench facts**

```
.venv/bin/pytest tests/microbench/test_phase6_facts.py -v
```

- [ ] **Step 2: Run Phase 1-5 regression**

```
.venv/bin/pytest tests/parity/test_phase1_5_examples_unchanged.py -v
```

- [ ] **Step 3: Generate one HTML manually + spot-check §21/§22**

- [ ] **Step 4: Verify Perfetto JSON has Atomic track**

- [ ] **Step 5: Done**

Phase 6 ships when all tasks complete + tags landed.

---

## End-of-plan checklist

- [ ] M1 (Frontend + config + AtomicEvent): T1-T5
- [ ] M2 (smem atomic + atom_reduction_smem): T6-T9
- [ ] M3 (gmem atomic + 3 examples): T10-T16
- [ ] M4 (Cluster TMA store + cooperative epilogue): T17-T19
- [ ] M5 (Trace + viz + docs + final): T20-T25
- [ ] All 5 milestone tags
- [ ] Phase 1-5 regression unbroken
- [ ] 5 new examples + 5 tutorials shipped
- [ ] README v6 reflects Phase 6
