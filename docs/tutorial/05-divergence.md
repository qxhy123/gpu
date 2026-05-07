# Chapter 05 — Branch Divergence Deep Dive

## Divergence as the SIMT cost

Chapter 01 introduced the SIMT stack and noted that divergence serializes two paths. This chapter quantifies the cost precisely and explores where divergence actually comes from in real code.

Recall from the timing results in earlier chapters:
- A non-divergent `add.f32` costs 4 cycles throughput, 4 cycles latency.
- A `ld.global` costs ~400 cycles latency.
- A divergent branch with `k` paths each of `n` instructions costs at least `k × n` cycles for the branch body — regardless of how many lanes take each path.

In `divergence_demo` with `block=(32,1,1)`, the branch splits 16 vs 16 lanes. Each path has 1 instruction (`mov.u32`). Total: 2 cycles for the two paths plus SIMT stack overhead. The `DIVERGENCE_SERIAL` stall count in the report captures the cycles where some lanes were idle during the "other" path.

## Walking `divergence_demo` in depth

Run in timing mode:

```bash
python examples/divergence_demo/run.py
```

Open `report.html`. The **stall breakdown** table shows a `DIVERGENCE_SERIAL` row. For this kernel it should be small — the divergent paths are each just one instruction — but it is measurable. The `ISSUED` count reflects instructions that actually issued; `DIVERGENCE_SERIAL` counts cycles where the warp was executing the inactive path.

The **Perfetto trace** (`trace.json`) shows this most clearly. Open [https://ui.perfetto.dev](https://ui.perfetto.dev), drag in `trace.json`. Find the single warp (warp 0). You will see:

1. Several `ISSUED` events for the setup instructions.
2. A `DIV_PUSH` marker at the `@%p1 bra THEN` instruction — this is where the warp forks.
3. Two consecutive runs of instructions — first the else-path (mov 200), then the then-path (mov 100) — each with the complementary active mask.
4. A `DIV_POP` marker at the `DONE:` label — warp reconverges.

The gap between `DIV_PUSH` and `DIV_POP` minus the single-path instruction count is the divergence overhead.

## Where divergence comes from in real code

### 1. Data-dependent branches

The most common source: `if (A[tid] > threshold)`. Each lane has a different value of `A[tid]`, so some lanes take the branch and others do not. This is essentially unavoidable when the branch condition depends on per-thread data.

### 2. Loop trip-count differences

```cuda
for (int i = 0; i < len[tid]; ++i) { ... }
```

If different threads have different `len[tid]`, some lanes exit the loop early while others continue. The warp keeps looping until all active lanes have finished — idle lanes pay full divergence cost.

### 3. Edge-of-grid boundary checks

```cuda
if (tid < N) { ... }
```

The last warp of a CTA may have fewer than 32 valid threads if `N % 32 != 0`. The extra lanes are predicated off, which is a mild divergence cost.

### 4. Irregular data structures

Traversing linked lists, trees, or hash tables where different threads follow different pointer chains leads to highly variable branch counts and is particularly expensive on GPUs.

## Mitigations

### Sort threads by branch decision

If the branch decision can be precomputed (e.g., classify data into two groups), sort the input so that threads 0–15 are one group and threads 16–31 are another. Then the warp either takes the branch for all lanes or none — no divergence. This is called **warp-level sorting** or **ballot-based dispatch**.

### Predicated execution

Instead of a branch, use a predicated instruction that always executes but only writes when the predicate is true:

```ptx
setp.lt.s32 %p1, %r1, 16;
@%p1 mov.u32 %r3, 100;    // always executes, only writes if p1=true
@!%p1 mov.u32 %r3, 200;   // always executes, only writes if p1=false
```

This avoids a branch entirely — both instructions execute in all lanes, but only the appropriate lane writes. This is beneficial when both paths are short (1–2 instructions).

### Warp-level primitives (Phase 2+)

`__ballot_sync`, `__shfl_sync`, and `__reduce_and_sync` can be used to coordinate between lanes without branching. Phase 1 does not model shuffle instructions; Phase 2 will add them.

## 改一改 — Nest two divergent branches; observe stack depth = 3

Add a second branch inside the `THEN` block:

```ptx
// Inside THEN: split lanes 0-7 and 8-15
setp.lt.s32 %p2, %r1, 8;
@%p2 bra INNER_THEN;
mov.u32 %r4, 50;
bra INNER_DONE;
INNER_THEN:
mov.u32 %r4, 25;
INNER_DONE:
```

Now the SIMT stack has depth 3: the outer divergence (all 32 lanes), then within the THEN path the inner divergence (lanes 0–15), then within that the inner-THEN (lanes 0–7) vs inner-else (8–15). The `DIVERGENCE_SERIAL` overhead accumulates: the original 2-path cost plus the nested 2-path cost for the 16-lane subset.

In the Perfetto trace, look for two `DIV_PUSH` events before any `DIV_POP`. The total divergence overhead is approximately:
- Outer THEN path: 1 cycle for the outer `mov.u32 %r3, 100`.
- Inner divergence within THEN: ~2 cycles (inner-else then inner-THEN).
- Outer else path: 1 cycle for the outer `mov.u32 %r3, 200`.
- Total: ~4+ cycles of divergence body vs 1 cycle if there were no branches.

## 真机对照

Skipped — no reference fixtures committed. On a real H100, divergence overhead is structurally identical: the SIMT stack serializes paths. The absolute cycle numbers are different (H100 has a 1.8 GHz clock and can issue 2 warps per cycle per SM), but the *ratio* of divergent cycles to non-divergent cycles is representative.
