# Chapter 01 — The SIMT Execution Model

## From CPU SIMD to GPU SIMT

A CPU SIMD unit processes multiple data elements with a *single instruction*, but each element is independent — two "lanes" of the same SIMD instruction cannot take different code paths.

NVIDIA GPUs use **SIMT** (Single Instruction, Multiple Threads). Like SIMD, all 32 threads in a warp execute the same instruction at the same time. Unlike SIMD, each thread has its own register file and program counter *conceptually* — threads look like independent scalar processors. The hardware reconciles this by maintaining a single *shared* PC for the warp plus a per-lane **active mask** that says which lanes are currently executing.

The key difference from SIMD: threads *can* diverge and take different branches. The hardware serializes the two paths, running one set of active lanes, then the other. This serialization is the cost of SIMT flexibility.

## Anatomy of a warp

A warp is 32 threads numbered 0–31 (their `tid.x` within a 1D block). Threads execute *lockstep*: in each cycle, every active lane executes the same instruction with its own register values.

The **active mask** is a 32-bit integer. Bit `i` is 1 if lane `i` is currently active (should execute the current instruction), 0 if it is idle (predicated off or in a divergent-but-inactive path).

At kernel launch all 32 lanes are active (mask = `0xFFFFFFFF`). The mask changes when:
- A predicated branch (`@%p bra LABEL`) is taken by some lanes but not others.
- A reconvergence point is reached (mask is restored from the SIMT stack).

## The SIMT stack

When a warp encounters a *conditional branch* where some lanes want to jump and others do not, the hardware must execute both paths. It does so with a stack:

1. **IPDOM computation** — at compile time (or in our simulator, at PTX parse time), the *immediate post-dominator* (IPDOM) of every branch is computed. The IPDOM is the earliest point that all paths through the branch must eventually reach.
2. **Diverge** — a `DIV_PUSH` event records (taken-path PC, fallthrough PC, reconvergence PC) on the stack.
3. **First path** — the active mask is set to lanes that took the branch; execution continues from `taken-path PC`.
4. **Reconvergence** — when the first path reaches `rpc`, a `DIV_POP` restores the full mask and starts the second path.
5. **Second path** — executes with the complementary mask.
6. **Merge** — at the post-dominator both paths have been executed; full mask resumes.

This means that if a branch splits 16 lanes each way, the total cost is ≥ 2× the cost of one path — plus stack management overhead.

## Walking `divergence_demo`

Open `examples/divergence_demo/kernel.ptx`. The kernel writes `100` for threads 0–15 and `200` for threads 16–31:

```ptx
setp.lt.s32 %p1, %r1, 16;   // p1 = (tid < 16)
@%p1 bra THEN;               // lanes 0-15 jump to THEN
mov.u32 %r3, 200;            // lanes 16-31 fall through → r3 = 200
bra DONE;
THEN:
mov.u32 %r3, 100;            // lanes 0-15 → r3 = 100
DONE:
st.global.u32 [%rd1], %r3;
```

Run it:

```bash
python examples/divergence_demo/run.py
```

Output:

```
output: [100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 200 200
 200 200 200 200 200 200 200 200 200 200 200 200 200 200]
```

The values are correct. Open `report.html` and look at the stall breakdown. You should see `DIVERGENCE_SERIAL` cycles — these are cycles where half the warp sits idle while the other half's path executes.

In timing mode the divergence event appears as a `DIV_PUSH` in the trace timeline, followed by two serial instruction windows (one per path), then `DIV_POP`.

## Why the stack costs cycles

With a 1-cycle instruction throughput:
- A **non-divergent** warp executes the branch body in N cycles for N instructions.
- A **divergent** warp executes the *taken* path (N₁ instructions), then the *fallthrough* path (N₂ instructions) = N₁ + N₂ cycles total — even if only one instruction differs.

In `divergence_demo`, both paths have one `mov` plus a shared `st.global`. The overhead is one extra `mov` cycle due to serialization, plus the SIMT stack push/pop. Look at the `DIVERGENCE_SERIAL` row in the stall breakdown table: it should be non-zero, representing cycles where 16 lanes waited.

As divergence gets worse — e.g., `tid % 2 == 0` where lanes alternate — the overhead stays the same in this case because there are still only two distinct paths. But *nested* divergence (branch inside a branch) stacks the cost, since each level adds another push/pop.

## 改一改 — Flip the predicate to `tid % 2 == 0`

In `kernel.ptx`, change:

```ptx
setp.lt.s32 %p1, %r1, 16;
```

to:

```ptx
// rem.u32 %r0, %r1, 2;   -- PTX doesn't have rem; use and
and.b32 %r0, %r1, 1;       // r0 = tid & 1
setp.eq.u32 %p1, %r0, 0;   // p1 = (tid is even)
```

Now the mask alternates: lanes 0, 2, 4, … take the THEN path; lanes 1, 3, 5, … take the else. The *two groups* are still serialized, so the total `DIVERGENCE_SERIAL` count stays similar. What changes is *which* lanes are in each group — the mask is `0x55555555` instead of `0x0000FFFF`. Real GPU performance is the same: divergence overhead is about path count, not how the lanes split.

## 真机对照

Skipped — no reference fixtures committed. On a real H100, `divergence_demo` would show the same output array. The cycle count would differ (H100 has much deeper pipelining and a clock near 1.8 GHz), but the `DIVERGENCE_SERIAL`-to-total-cycles ratio would be structurally similar for this micro-kernel.
