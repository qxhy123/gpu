# Chapter 26 — red vs atom

## Two Instructions, One Operation

PTX offers two ways to perform an atomic read-modify-write on global memory:

```ptx
atom.global.min.s32 %rdst, [%raddr], %rsrc;   // atom: returns old value
red.global.min.s32         [%raddr], %rsrc;   // red: no return value
```

Both instructions send a request to the L2 atomic ALU, lock the cache line, compute the new value, and write it back. The hardware behavior at the memory side is identical. The difference is purely about the **return path**:

- `atom` writes the old value into `%rdst`. The issuing warp must wait for the L2 to send that value back — the result register is tracked by the scoreboard, and any subsequent instruction that reads `%rdst` will stall until the round-trip completes.
- `red` discards the old value. The warp does not need to wait for any acknowledgment from L2 beyond the fire-and-forget send. The warp can proceed to its next independent instruction without the scoreboard holding anything.

In the simulator, both instructions model the same latency (the round-trip to L2 is charged either way for the purposes of cycle counting). On real H100/A100 hardware, `red` can be slightly faster in practice because the hardware does not need to allocate return path resources (no response buffer slot, no scoreboard entry for the result). Under high L2 traffic, this resource difference becomes measurable.

## 走通 red_min_max

```bash
python examples/red_min_max/run.py
```

The kernel launches `grid=(8,1,1)`, `block=(32,1,1)` — 256 threads. Each thread reads one element from a 256-element `int32` array and issues two reduction instructions:

```ptx
red.global.min.s32 [%rd1], %s0;     // OUT[0] = global min
red.global.max.s32 [%rd4], %s0;     // OUT[1] = global max
```

The output is verified against NumPy:

```
red_min_max: cycles=<N>
  min = <M>, max = <X>
  numpy: min = <M>, max = <X>
```

Both values will match exactly. The `red` instructions are fire-and-forget: the kernel does not read back the result into any register, so there is no scoreboard stall after the two `red` instructions. The kernel exits immediately after issuing them (the `ret` instruction does not wait for L2 to finish — the TMA-style async completion is implicit).

## 看模拟器

```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default

cfg = load_default()
rng = np.random.RandomState(0)
in_arr = rng.randint(0, 1000, size=256).astype(np.int32)
out = np.zeros(2, dtype=np.int32)
out[0] = 0x7FFFFFFF; out[1] = -0x80000000
ptx = pathlib.Path("examples/red_min_max/kernel.ptx").read_text()
res = gpusim.run(
    ptx_src=ptx, grid=(8,1,1), block=(32,1,1),
    params={"IN": in_arr.copy(), "OUT": out}, mode="timing", config=cfg,
)
print("cycles:", res.metrics["cycles"])
print("atom_red_ratio:", res.atomic_metrics.get("atom_red_ratio", "n/a"))
```

The `atom_red_ratio` metric reports the fraction of all atomic memory events that were `red` instructions (as opposed to `atom`). For `red_min_max` this should be 1.0: every atomic event in the kernel is a `red`. Open `report.html` → **§21 Atomic Events** to see the `red.global.min` and `red.global.max` events separated in the event type breakdown.

The HTML panel also shows the L2 queue depth for `OUT[0]` and `OUT[1]`. With 256 threads all reducing into two locations, you will see queue depths up to 128 per address — the same serialization pattern as Chapter 22's `atom_histogram`, but now with `red` instead of `atom`.

## 改一改

**Convert `red` to `atom`.** Replace both `red.global.min.s32` and `red.global.max.s32` with `atom.global.min.s32` versions that write their return values into temporary registers:

```ptx
.reg .s32 %sold<2>;
atom.global.min.s32 %sold0, [%rd1], %s0;
atom.global.max.s32 %sold1, [%rd4], %s0;
```

Rerun the kernel. The cycle count should be the same or within a few percent — the simulator models identical L2 round-trip latency for both instruction forms. The difference you will notice is:

1. The PTX register file now allocates two additional `.s32` registers (`%sold0`, `%sold1`) that are written but never read. The compiler (on real hardware) might warn about dead stores.
2. The scoreboard now has live entries for `%sold0` and `%sold1` after the two `atom` instructions. If the kernel had any subsequent instruction that accidentally read those registers, it would stall on the round-trip. In this kernel there are none, so the effect is invisible.

On real A100/H100 hardware, you would measure the gap between `red` and `atom` under high L2 traffic: a workload that issues millions of `atom.global.add` per warp, converted to `red.global.add`, typically shows a 2–5% cycle reduction at the kernel level — small, but consistent.

## 真机对照

CUDA's standard `atomicMin` / `atomicMax` intrinsics always use `atom` (they return the old value). If you write the kernel in PTX or inline PTX and genuinely do not need the return value, prefer `red`. CUTLASS's reduction epilogues use `red.global` throughout their store loops for exactly this reason: the accumulator value is already in registers; the reduction to global memory is a one-way write, and there is no reason to wait for the old value to come back.

NVIDIA's PTX ISA guide states: "The `red` instruction is preferred when the return value is not needed, as it may allow the hardware to optimize the operation." On Hopper, this optimization includes coalescing adjacent `red` operations into a single L2 transaction when they target the same cache line and the same operation type — a further reason to prefer `red` in vectorized reduction kernels.

The `atom_red_ratio` metric in the simulator's `atomic_metrics` gives you an immediate signal: a ratio well below 1.0 in a kernel that performs many reductions is a flag to audit whether all those `atom` calls actually use their return values.
