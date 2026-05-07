# Chapter 06 — Occupancy

## Three knobs

GPU occupancy is the ratio of active warps to the SM's maximum warp capacity. Higher occupancy gives the scheduler more candidates to pick from when hiding memory latency — but it is not a goal in itself. The three resources that limit occupancy are:

1. **Warps per CTA** — determined by `block size / 32`. A block of 128 threads has 4 warps. An SM can hold at most 64 warps (on our model SM), so it can hold at most `64 / 4 = 16` such CTAs — already bounded by warp count.

2. **Registers per thread** — the SM has 65536 registers total. A thread using 128 registers means a CTA of 128 threads uses `128 × 128 = 16384` registers. The SM can fit `65536 / 16384 = 4` CTAs.

3. **Shared memory per CTA** — the SM has 49152 bytes (48 KiB) of shared memory. A CTA using 32768 bytes (32 KiB) limits occupancy to `49152 / 32768 = 1` CTA.

The *bottleneck* is the resource that gives the smallest CTA count. The occupancy in warps is `active_ctas × warps_per_cta`.

## The bottleneck classifier

Run `reduction_smem` in timing mode:

```bash
python examples/reduction_smem/run.py
```

The report shows:

```
All metrics: {'cycles': 620, 'occupancy': {'active_ctas': 32, 'bottleneck': 'max_ctas_cap'}}
```

`bottleneck: max_ctas_cap` means the SM could fit more CTAs by register and smem budget, but the hard cap of 32 CTAs per SM (`max_ctas_per_sm=32`) is the actual limit. This happens when the kernel uses few registers and little shared memory — a 32-thread CTA uses so few resources that the cap kicks in first.

The `active_ctas = 32` at 32 threads each gives `32 warps` active. With a 64-warp SM capacity, we're at 50% occupancy by warp count.

## Three occupancy scenarios

Using `compute_occupancy` directly:

```python
from gpusim.core.occupancy import compute_occupancy
from gpusim.config.loader import load_default
cfg = load_default()
```

**Scenario 1 — register-bound (regs_per_thread=128):**

```python
res = compute_occupancy(cfg, threads_per_cta=128, regs_per_thread=128, smem_per_cta=0)
# OccupancyResult(active_ctas=4, warps_per_cta=4, max_by_warps=16, max_by_regs=4,
#                 max_by_smem=49152, bottleneck='regs')
```

`max_by_regs=4` is the binding constraint. With 4 CTAs × 4 warps = 16 warps active, we have 25% of the SM's warp capacity. The scheduler has few candidates, so global memory latency (400 cycles) is rarely hidden.

**Scenario 2 — smem-bound (smem_per_cta=32768 bytes):**

```python
res = compute_occupancy(cfg, threads_per_cta=128, regs_per_thread=32, smem_per_cta=32768)
# OccupancyResult(active_ctas=1, warps_per_cta=4, max_by_warps=16, max_by_regs=16,
#                 max_by_smem=1, bottleneck='smem')
```

Only 1 CTA fits. 4 warps active — the SM is effectively single-threaded from a scheduling perspective. Every global load stall exposes the full ~400-cycle latency.

**Scenario 3 — warps-bound (small CTA, smem_per_cta=0):**

```python
res = compute_occupancy(cfg, threads_per_cta=32, regs_per_thread=32, smem_per_cta=0)
# OccupancyResult(active_ctas=32, warps_per_cta=1, max_by_warps=64, max_by_regs=64,
#                 max_by_smem=49152, bottleneck='max_ctas_cap')
```

With 32-thread CTAs (1 warp each), we hit the `max_ctas_per_sm=32` cap at 32 active warps. That's 50% of the SM's 64-warp capacity — limited by the CTA count cap, not registers or smem.

## Why higher occupancy isn't always better

Consider a kernel that:
- Has no global loads (pure arithmetic on register data).
- Uses 256 registers per thread.

This kernel is **register-bound** at low occupancy, but it may not need high occupancy because there is no latency to hide. The scheduler just keeps issuing from the few active warps without stalls.

In contrast, a memory-bound kernel needs enough warps to keep the memory pipeline busy. The rule of thumb: **you need enough warps to fill the memory latency**. For a 400-cycle global load latency and 4-warp issue width, you need roughly `400 / 4 = 100` warps to achieve theoretical peak throughput. Our SM has 64 warps, so any memory-bound kernel benefits from maximizing occupancy up to 64 warps.

Chapter 02 covered the latency-hiding math in more detail.

## 改一改 — Reduce `regs_per_sm` and observe a new bottleneck

In `gpusim/config/default_hopper.yaml`, change `regs_per_sm: 65536` to `regs_per_sm: 32768`. Now rerun:

```python
from gpusim.core.occupancy import compute_occupancy
from gpusim.config.loader import load_default
cfg = load_default()  # picks up updated YAML
res = compute_occupancy(cfg, threads_per_cta=128, regs_per_thread=32, smem_per_cta=0)
print(res)
```

Expected: `bottleneck='regs'` instead of `'regs'` being non-binding. With half the registers, `max_by_regs = 32768 / (128 × 32) = 8` CTAs instead of 16. The bottleneck shifts from warp-count to registers. Rerun `reduction_smem` to see the cycle count change — fewer active warps means less latency hiding.

Restore `regs_per_sm: 65536` when done.

## 真机对照

Skipped — no reference fixtures committed. On a real H100 SXM5:
- `regs_per_sm` = 65536 (same as our model)
- `max_ctas_per_sm` = 32 (same)
- `warps_per_sm` = 64 (same)
- `smem_per_sm` = 228 KB (much more than our 48 KB)

The occupancy calculations would give different numbers for the smem scenario (more shared memory = more CTAs can fit). The register and warp-count scenarios would match our model closely.
