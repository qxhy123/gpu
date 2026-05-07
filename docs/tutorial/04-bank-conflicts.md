# Chapter 04 — Shared Memory Bank Conflicts

## 32 banks, 4-byte stride

Shared memory on NVIDIA GPUs is divided into **32 banks**. Each bank is 4 bytes wide. The bank assignment of a byte address is:

```
bank(addr) = (addr / 4) % 32
```

In other words, words 0, 32, 64, … map to bank 0; words 1, 33, 65, … map to bank 1; and so on.

When 32 threads in a warp all issue `ld.shared` simultaneously, the hardware can serve all 32 lanes in one cycle **if and only if** no two lanes map to the same bank (or all lanes map to the *same* address — a broadcast). If two lanes map to the same bank, those two requests must be serialized, adding one cycle per extra request.

**Conflict degree** = max number of requests targeting the same bank. If conflict degree = k, the `ld.shared` takes k cycles instead of 1.

## Three patterns

### Pattern 1: stride 1 (no conflict)

Lane `i` accesses `smem[i*4]` (each lane reads its own 4-byte word). The addresses are 0, 4, 8, …, 124. The banks are 0, 1, 2, …, 31. Each bank is hit exactly once — **no conflict**.

This is the best case and is what `examples/bank_conflict_demo/kernel.ptx` demonstrates.

### Pattern 2: stride 32 (32-way conflict)

Change the access to `smem[i * 128]` (stride 128 bytes = 32 words). Then:
- Lane 0 → address 0 → bank 0
- Lane 1 → address 128 → bank 0
- Lane 2 → address 256 → bank 0
- ...

All 32 lanes map to bank 0. This is a **32-way conflict**: the hardware must serialize all 32 requests. The `ld.shared` takes 32 cycles instead of 1.

In PTX: change `shl.b32 %r2, %r1, 2` (×4, stride=1) to `shl.b32 %r2, %r1, 7` (×128, stride=32).

### Pattern 3: broadcast (no conflict, single address)

If all 32 lanes read `smem[0]` (the same address), the hardware recognizes this as a **broadcast** and serves all lanes in 1 cycle — bank 0 is read once and the value is broadcast. This is the other conflict-free case.

In PTX: replace the address computation with `mov.u64 %rd2, 0`.

## Walking `bank_conflict_demo`

Run the demo with the default stride-1 kernel:

```bash
python examples/bank_conflict_demo/run.py
```

Output should show `cycles: 341` (or similar) and the output is `[0 1 2 ... 31]`.

Open `report.html` and look at the **bank conflict histogram**. For stride-1:
- All bars are at conflict degree 1 (no conflict).

Now copy `kernel.ptx` to `kernel_stride32.ptx` and change the access pattern to stride 32 (modify the `shl.b32` shift from 2 to 7). Re-run with that kernel. In the report:
- The `st.shared` shows conflict degree 32.
- The `ld.shared` shows conflict degree 32.
- Total cycles increase by approximately 31 × 2 = 62 cycles (the two accesses each pay 31 extra cycles).

The `bank_conflict_hist` in the HTML report shows a histogram: x-axis is conflict degree (1 = no conflict, 2 = 2-way conflict, …, 32 = 32-way), y-axis is number of instructions with that degree.

## Padded layouts

A common technique to eliminate 32-way conflicts with strided access is **padding**: allocate `smem[33][N]` instead of `smem[32][N]`. Then row `r` starts at byte offset `r * 33 * 4`, and consecutive rows are offset by an odd multiple of 4 bytes. Since 33 is not a multiple of 32, consecutive rows land in different banks for column 0, breaking the alignment.

In Phase 1 examples, we do not implement padded layouts, but the principle is straightforward:

```cuda
// In CUDA C++:
__shared__ float sA[16][17];  // pad one column
// sA[row][col] maps to byte (row*17 + col)*4
// Consecutive rows start at different banks
```

For the PTX equivalent, change the stride from `shl.b32 %r5, %r4, 6` (64 bytes = 16 floats per row) to `shl.b32 %r5, %r4, 6; add.s32 %r5, %r5, 4` (64 + 4 = 68 bytes = 17 floats per row).

## 改一改 — Odd stride 33 → 1; even stride 34 → 2

Test these two strides:

**Stride 33** (not a multiple of 32): Lane `i` accesses bank `(i * 33) % 32`. Since `gcd(33, 32) = 1`, all 32 values of `(i * 33) % 32` for i=0..31 are distinct — each bank is hit exactly once. **No conflict.**

**Stride 34** (= 2 × 17, where 17 is odd but 34 = 2 × 32/2 + 2): Lane `i` accesses bank `(i * 34) % 32 = (i * 2) % 32`. Lanes 0, 1, 2, …, 15 hit banks 0, 2, 4, …, 30. Lanes 16, 17, …, 31 also hit banks 0, 2, 4, …, 30. Every even bank is hit twice. **2-way conflict** for all 32 accesses.

The general rule: **stride s has conflict degree `gcd(s, 32)`**. An odd stride always has `gcd = 1` (no conflict). An even stride `s = 2k` has conflict degree at least 2.

## 真机对照

Skipped — no reference fixtures committed. On real H100 the bank layout is the same (32 banks, 4-byte granularity). The penalty formula is the same: k-way conflict costs k cycles per access. The absolute cycle counts differ due to H100's higher clock and pipelining, but the *relative* cost (stride-32 is ~31× slower than stride-1 for shared memory accesses) holds.
