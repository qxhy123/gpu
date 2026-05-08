# Reference fixtures (real-GPU)

To validate the simulator against real H100 behavior, run `gen_reference.py`
on a CUDA-capable host. It writes JSON files into `tests/reference/data/`
that the simulator's reference tests load when present.

## Layout (`*.ref.json` schema)

```json
{
  "kernel": "vector_add",
  "ptx_path": "examples/vector_add/kernel.ptx",
  "launch": {"grid": [8,1,1], "block": [128,1,1]},
  "device": {"name": "H100 SXM5", "sm_count": 132},
  "inputs_shape": {"A": [1024], "B": [1024], "C": [1024], "N": []},
  "inputs_seed": 42,
  "outputs": {"C": "<base64 npy bytes>"},
  "metrics": {
    "active_warps_per_sm": 64,
    "achieved_occupancy": 1.0,
    "smem_bank_conflicts": 0,
    "gld_efficiency": 1.0
  }
}
```

## Two layers of comparison

1. **Numerical** — `outputs` numpy buffers compared via `assert_allclose(rtol=1e-5)`
2. **Metric** — simulator metrics within tolerance:
   - `active_warps_per_sm` ±5%
   - `smem_bank_conflicts` exact
   - `gld_efficiency` ±10%

`timing` cycles are *not* compared (cycle-approximate ≠ cycle-accurate).

## Skipping when fixtures absent

The reference tests are decorated with `@pytest.mark.reference` and skipped
if the corresponding `.ref.json` does not exist.

---

## Phase 2 kernel schemas

The following four kernels were added in Phase 2. Run
`python tests/reference/gen_reference.py <kernel>` on a CUDA-capable host to
generate the stub JSON, then fill in the `outputs` and `metrics` fields.

### l1_thrash_demo

```json
{
  "kernel": "l1_thrash_demo",
  "ptx_path": "examples/l1_thrash_demo/kernel.ptx",
  "launch": {"grid": [1, 1, 1], "block": [32, 1, 1]},
  "inputs_shape": {"A": [16777216], "OUT": [32], "K": [], "STRIDE": []},
  "inputs_seed": 42,
  "outputs": {"OUT": "<base64 npy bytes>"},
  "metrics": {
    "l1_hit_rate": 0.0,
    "l2_hit_rate": 0.0,
    "row_buffer_hit_rate": 0.94
  }
}
```

Cache metric tolerances for l1_thrash_demo: ± 0.10 (absolute). The hit rates
are 0% in the simulator (no reuse in a single-pass kernel); a real GPU may
show non-zero rates due to prefetching. `row_buffer_hit_rate` is high for the
small-K configurations because all accesses go to the same DRAM row.

### smem_vs_l1_demo

```json
{
  "kernel": "smem_vs_l1_demo",
  "ptx_path": "examples/smem_vs_l1_demo/kernel_smem.ptx",
  "launch": {"grid": [1, 1, 1], "block": [16, 16, 1]},
  "inputs_shape": {"A": [256], "B": [256], "C": [256]},
  "inputs_seed": 0,
  "outputs": {"C": "<base64 npy bytes>"},
  "metrics": {
    "l1_hit_rate_smem_variant": 0.0,
    "l1_hit_rate_no_smem_variant": 0.72
  }
}
```

Tolerance: l1_hit_rate ± 0.15. The smem variant has 0% L1 hit rate (global
loads bypass L1 for smem staging); the no-smem variant relies on L1 reuse and
should have ≥ 0.5 hit rate on both simulator and real hardware.

### bw_saturation_demo

```json
{
  "kernel": "bw_saturation_demo",
  "ptx_path": "examples/bw_saturation_demo/kernel.ptx",
  "launch": {"grid": [64, 1, 1], "block": [32, 1, 1]},
  "inputs_shape": {"A": [2048], "OUT": [2048]},
  "inputs_seed": 42,
  "outputs": {"OUT": "<base64 npy bytes>"},
  "metrics": {
    "channel_utilization_mean": 0.79,
    "hbm_queue_wait_mean": 5.9
  }
}
```

Tolerance: channel_utilization_mean ± 0.15, hbm_queue_wait_mean ± 5.0. Real
GPUs have different channel counts (12–16 vs. 8 in simulator); normalize by
channels when comparing.

### row_buffer_demo

```json
{
  "kernel": "row_buffer_demo",
  "ptx_path": "examples/row_buffer_demo/kernel.ptx",
  "launch": {"grid": [1, 1, 1], "block": [32, 1, 1]},
  "inputs_shape": {"A": [16777216], "OUT": [32], "STRIDE": []},
  "inputs_seed": 42,
  "outputs": {"OUT": "<base64 npy bytes>"},
  "metrics": {
    "row_buffer_hit_rate_stride32": 0.73,
    "row_buffer_hit_rate_stride65568": 0.0
  }
}
```

Tolerance: row_buffer_hit_rate ± 0.20. Real H100 uses XOR address hashing,
so the specific stride value that triggers row misses differs from the
simulator's linear layout. The qualitative trend (small same-row stride = high
hit rate, large cross-row stride = low hit rate) should hold on both.

Note: `_run_nvcc_and_capture_outputs` remains `NotImplementedError` — this
function requires real-GPU execution and is out of scope for the simulator-side
plan.
