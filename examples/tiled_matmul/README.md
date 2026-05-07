# tiled_matmul

16×16 矩阵乘法，单 CTA 单 tile。展示数据复用、shared memory tile load、`bar.sync` 同步、k-loop 内的 smem 访问模式。

## 关键代码点
- `kernel.ptx:21-29` 把 A、B 一次性装载进 shared memory（每个线程加载 1 个元素）
- `kernel.ptx:30` `bar.sync` 等 tile 装好
- `kernel.ptx:33-49` k-loop：每 k 读两次 smem，做一次 FMA

## 预期观察（timing mode）
- A 的 `ld.shared` 模式：行内线程访问 (row*16+k) — 同 row 的 16 个线程同时访问 16 个不同 bank → 无冲突
- B 的 `ld.shared` 模式：(k*16+col) — 同 col 的 16 个线程同时访问 16 个不同 bank → 无冲突
- HTML 报告里两次 `ld.shared` 的 bank conflict 直方图都在 1
- 主要时间花在 k-loop 的 16 次迭代

## 延伸思考
1. 把 B 的 layout 转置（`smem_B[col*16+k]`），看 bank conflict 怎么变
2. 把 block 改成 (32,8,1) 跑 32×8 一个 tile，观察 occupancy 与 stall 分布
