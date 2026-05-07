# Chapter 02 — The Warp Scheduler and Latency Hiding

## Latency-bound vs throughput-bound

Different operations in our simulator have very different latencies:

| Operation | Approximate latency |
|-----------|-------------------|
| `add`, `mul`, `mov` (arithmetic) | 4 cycles |
| `ld.shared` / `st.shared` | ~20 cycles |
| `ld.global` / `st.global` | ~400 cycles |
| `bar.sync` | 1 cycle (functional) + warp-sync cost |

A kernel dominated by global loads is *latency-bound* if it doesn't have enough work to overlap with those loads. The warp scheduler's primary job is to fill those ~400-cycle stalls by running other warps.

## Why one warp is not enough

Consider a simple kernel with one warp. The timeline looks like:

```
Cycle 1:   ld.global %f1, [%rd4]  -- issued, latency = 400 cycles
Cycles 2-400: SCOREBOARD stall (waiting for load to complete)
Cycle 401: add.f32 %f3, %f1, %f2  -- issued
...
```

Throughput = 1 useful instruction per 400 cycles ≈ 0.0025 IPC (instructions per cycle). That is terrible.

Now add 7 more warps (8 warps total, each with its own loads and adds). The scheduler can issue each warp's `ld.global` in cycles 1-8, then cycle 401 warp-0's load returns, and we can issue a useful instruction. By the time we've cycled through all 8 warps, most loads have completed and we have a continuous stream of useful work.

In the vector_add example with `block=(128,1,1)` there are `128/32 = 4` warps per CTA, and `8 CTAs` are scheduled on the same SM, giving `32` warps total. The simulator shows **545 cycles** for 1024 elements — about 0.53 cycles per element, much better than the single-warp case.

## LRR vs GTO scheduling policies

The simulator supports two scheduling policies, configurable in `gpusim/config/default_hopper.yaml`:

- **LRR (Least-Recently-Run)** — the scheduler always picks the warp that has been waiting the longest. Fair, predictable, but can waste time switching between warps that are all stalled.
- **GTO (Greedy-Then-Oldest)** — the scheduler keeps executing the *same* warp as long as it has a ready instruction (greedy), and only switches when that warp stalls. This maximizes instruction cache locality but can starve slower warps.

To compare, open `gpusim/config/default_hopper.yaml` and change `scheduler_policy: lrr` to `scheduler_policy: gto`, then rerun vector_add:

```bash
python examples/vector_add/run.py
```

For a memory-bound kernel like vector_add, LRR tends to perform better because it naturally interleaves multiple warps and hides global memory latency. GTO tends to shine on compute-bound kernels where cache reuse matters more than latency hiding.

In the HTML report, look at the **warp timeline** section (if your browser renders it) — each row is a warp, each column a cycle. LRR shows a "striped" pattern; GTO shows longer solid runs per warp.

## Stall taxonomy

Every cycle a warp is not issuing an instruction, it is in a *stall state*. The simulator tracks eight stall reasons:

| Token | Meaning |
|-------|---------|
| `ISSUED` | Instruction successfully issued this cycle (not a stall). |
| `SCOREBOARD` | Waiting for a register to become ready (usually after a load). |
| `MEM_DEP` | A memory instruction is waiting for an earlier memory op to finish (WAR/RAW hazard in the memory pipeline). |
| `BARRIER` | Waiting at `bar.sync` for other warps in the CTA to arrive. |
| `STRUCTURAL` | Functional unit busy (e.g., two warps both want the load unit). |
| `OPERAND` | Register file bank conflict — two operands map to the same bank. |
| `DIVERGENCE_SERIAL` | Warp is executing the "other" divergent path; some lanes are idle. |
| `IDLE` | Warp has no work yet (before first issue) or finished all instructions. |

For vector_add, `SCOREBOARD` dominates because the latency gap between `ld.global` and the dependent `add.f32` is ~400 cycles and only 32 warps are available to fill it.

You can read per-warp stall counts from the `stall_breakdown` table in `report.html`. The "total" row shows the aggregate.

## 改一改 — Use a tiny block to stress stalls

Change the test invocation (in `examples/vector_add/run.py`) to use `block=(32,1,1)` (one warp per CTA) and `grid=(32,1,1)` (32 CTAs, so we still get 1024 elements). Run timing:

```bash
python examples/vector_add/run.py
```

With only one warp per CTA and fewer warps total visible to the scheduler at once, `SCOREBOARD` stalls become a larger fraction of total cycles. The cycle count increases substantially — you are now seeing the cost of not having enough concurrency to hide global load latency.

Contrast this with `block=(256,1,1)` and `grid=(4,1,1)` — same total threads but 8 warps per CTA. The cycle count should decrease as the scheduler has more candidates to pick from.

## 真机对照

Skipped — no reference fixtures committed. On real hardware the pattern is the same: increasing the number of active warps (by increasing block size or having more CTAs per SM) hides memory latency. The "sweet spot" on H100 is typically 2048 active threads per SM (64 warps) for memory-bound kernels.
