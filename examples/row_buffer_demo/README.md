# row_buffer_demo

通过 STRIDE 参数演示 DRAM row-buffer locality。

## 关键代码点
- `kernel.ptx` 每个线程读 `addr = base + (tid * STRIDE) * 4`，一次 ld.global → st.global
- 整个 kernel 只做一次访存，便于隔离 HBM 行为

## 运行
```
python examples/row_buffer_demo/run.py
```

## STRIDE 选择与原因

Phase 2 spec §5.2 的地址 layout 把 channel 放在最低位（bits [9:7]），row 在 bits [30:19]。
直接 `stride = row_size = 4 KB` **不会** 触发 row miss——4 KB 增量只让 col-in-row 跳跃。
spec 计算的"理论"row-miss stride 是 512 KB（= 131072 floats），但实测中 stride=512 KB
还会让所有 32 个 lane 映射到 L1 set 0（因为 4096 & 0xFF = 0），引发 L1 thrashing 和模拟器 runaway。

run.py 实际使用的是 spec **作者后续修正**的两个 stride（spec §5.2 末尾有相应说明）：

- **`STRIDE=32`（row hit 基线）**：每个 thread 读 32 个连续 element 的小窗口；
  warp 第一次访问拉 row 进 buffer，后续访问命中同一 row
- **`STRIDE=65568`（= 16384 × 4 + 32 elements，row miss 基线）**：跳过 L1 set 冲突，
  每次访问触发不同 row

## 预期观察（Phase 2 timing mode）
- `STRIDE=32`：`row_buffer_hit_rate ≈ 0.73`，~121 cycles
- `STRIDE=65568`：`row_buffer_hit_rate ≈ 0`，~163 cycles

cycle delta 不大（kernel 只访存一次）；教学信号在 HTML 报告 §8 row buffer locality 节
的 hit rate 数字。

## 局限性
Phase 2 简化了真机 DRAM：channel 完整序列化所有 latency（spec §5.5）。真机有 bank 内
ACT/DATA 重叠和复杂的地址 hash。详见讲义 11。

## 延伸思考
1. 把 STRIDE 设为 1024（= row size 1024 floats = 4 KB）：col cycles 32 times within row 0，
   仍是 row hit。验证 spec 的"4 KB 不触发 row miss"判断
2. 把 STRIDE 设为 33（奇数）：散布到不同 L1 set，不同 col。看 row buffer hit rate 怎么变

## 参考
- `docs/tutorial/11-row-buffer.md` — 完整教学章节
- `docs/superpowers/specs/2026-05-08-gpusim-phase2-design.md` §5.2 — 地址 layout 与 stride 分析
