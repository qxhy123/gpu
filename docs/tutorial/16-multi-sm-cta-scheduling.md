# Chapter 16 — Multi-SM 与 CTA 调度

## 从单 SM 到多 SM 的拓扑跳跃

Chapters 1–15 ran every kernel on a single SM: one warp scheduler, one L1/smem, one set of register files. That was sufficient to explore instruction-level behavior — coalescing, bank conflicts, Tensor Cores, TMA. But real GPU workloads launch thousands of CTAs across dozens or hundreds of SMs simultaneously. Chapter 16 introduces the simulator's multi-SM topology and the first question that arises: *which SM gets which CTA, and when?*

The simulator's multi-SM mode models N independent SMs sharing a single L2 cache and HBM. By default N=8, matching a representative slice of a real device. Each SM has its own warp scheduler, register file, shared memory, and L1 cache. The L2 is shared across all SMs, with a configurable number of MSHR slots.

The key new configuration knob:

```python
from gpusim.config.loader import load_default
cfg = load_default()
cfg.n_sm = 8          # number of simulated SMs (default)
cfg.scheduler.cta_policy = "rr"   # or "greedy"
```

## CTA 调度策略：RR vs Greedy

When a kernel launches with `grid=(16,1,1)` and `cfg.n_sm=8`, the simulator must assign 16 CTAs to 8 SMs. Two policies are supported:

**Round-robin (RR)**: CTAs are assigned in order — SM 0 gets CTA 0, SM 1 gets CTA 1, ..., SM 7 gets CTA 7, SM 0 gets CTA 8, and so on. Deterministic and reproducible, but it ignores actual SM load. If CTA 0 takes twice as long as CTA 1, SM 0 still gets its second CTA (CTA 8) only after CTA 0 finishes — even if SM 1 is idle and could take it immediately.

**Greedy**: when a CTA finishes and an SM becomes free, the scheduler immediately assigns the next pending CTA to whichever SM is first available. This is load-balanced: fast SMs get more CTAs, slow SMs never create a queue.

The demo kernel in `examples/multi_sm_scheduler/kernel.ptx` deliberately creates an irregular workload. Even-numbered CTAs (where `cta_id & 1 == 0`) execute an extra 64-iteration busy loop before computing their output. Odd CTAs skip the loop and finish early. This asymmetry makes the two scheduling policies produce measurably different total cycle counts:

```ptx
// Even CTAs run the extra loop:
and.b32 %r5, %r0, 1;
setp.eq.u32 %p0, %r5, 0;
@!%p0 bra SKIP_LOOP;
LOOP:
    setp.ge.u32 %p1, %r6, 64;
    @%p1 bra SKIP_LOOP;
    add.u32 %r6, %r6, 1;
    bra LOOP;
SKIP_LOOP:
```

## 看模拟器

Run the comparison:

```bash
python examples/multi_sm_scheduler/run.py
```

Expected output:

```
# multi_sm_scheduler: RR vs greedy
  rr     : cycles=2048, max diff=0.00e+00
  greedy : cycles=1536, max diff=0.00e+00
```

Both policies produce correct output (max diff = 0), but greedy finishes in fewer total cycles by keeping idle SMs busy with pending work.

Open `report.html` and navigate to section §16 **CTA→SM 派发表** (dispatch table). Each row is a CTA, each column is an SM. The table shows:
- **Start cycle**: when the CTA first became active on its assigned SM.
- **End cycle**: when the CTA retired.
- **Duration**: directly reflects the extra busy loop for even CTAs.

With RR, SM 0 holds CTA 0 (long) and then CTA 8 (long) back-to-back — SM 0 is the critical path. With greedy, after CTA 1 (short) finishes on SM 1, SM 1 immediately picks up CTA 8 rather than waiting for SM 0 to drain.

## 改一改

**Change SM count:**

```python
cfg.n_sm = 4   # half as many SMs
```

With 16 CTAs and 4 SMs, each SM handles 4 CTAs instead of 2. The RR vs greedy gap grows because the scheduling inefficiency compounds over more rounds.

**Swap policy and observe dispatch timing:**

```python
for policy in ("rr", "greedy"):
    for n_sm in (4, 8):
        cfg.n_sm = n_sm
        cfg.scheduler.cta_policy = policy
        # run and record cycles
```

Plot cycles as a function of `(n_sm, policy)`. The greedy advantage is largest when workload variance is high (many even CTAs) and SM count is low (less parallelism to dilute the imbalance).

**Make workload uniform:** Change the busy loop count from 64 to 0 for all CTAs. With a balanced workload, RR and greedy should produce identical cycle counts — the scheduling decision is irrelevant when all CTAs have the same duration.

## 真机対照

A real H100 SXM5 has **132 SMs**. The hardware CTA scheduler is considerably more complex than either RR or greedy:

- **Priority queues**: streams carry priority levels; higher-priority kernels preempt lower-priority CTA assignments.
- **CGA (Cooperative Grid Arrays)**: Hopper introduces cluster-level scheduling where groups of CTAs are co-scheduled onto adjacent SMs to share data via `cluster.shared` memory.
- **Wave quantization**: with 132 SMs and a kernel launching 133 CTAs, one SM gets an extra CTA while all others finish — the "last wave" runs at 1/132 efficiency. Real production kernels are tuned to launch in multiples of 132 (or use persistent kernels that avoid this entirely).
- **SM occupancy limits**: each SM can hold multiple CTAs simultaneously if register/smem usage permits. The simulator models single-CTA occupancy per SM for simplicity.

The simulator's two-policy model is an accurate pedagogical representation of the core scheduling tradeoff. The lesson carries directly to production: greedy dispatch (and its hardware analog) is important whenever CTAs have variable execution time, which happens in attention kernels with variable sequence lengths, sparse operations, or any workload with data-dependent branching.
