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
