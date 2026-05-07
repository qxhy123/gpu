# bank_conflict_demo

Shared-memory 32-bank access pattern comparison. Two PTX files are shipped:

- `kernel.ptx` — stride=1 (no bank conflict): lane i accesses smem[i*4],
  each lane maps to a distinct bank → 1 transaction, fast.
- `kernel_stride32.ptx` — stride=32: lane i accesses smem[i*128],
  all 32 lanes map to bank 0 → 32-way conflict, serialized in 32 steps.

## 预期观察（timing）

Run both in timing mode:

```bash
python scripts/demo_all.py
```

- `kernel.ptx` (stride=1): `bank_conflict_hist` all 1, fast cycle count
- `kernel_stride32.ptx` (stride=32): `conflict_degree=32`, ~31 extra cycles
  compared to stride=1 (the warp serializes 32 bank accesses)
- broadcast (`mov.u32 %r2, 0` for all lanes): `conflict_degree=1` (broadcast)

## 延伸思考

- 把 stride 改成 33 会怎样？（提示：奇 stride 与 32 互质 → 无冲突）
