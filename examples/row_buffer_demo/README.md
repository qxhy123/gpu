# row_buffer_demo

通过 STRIDE 参数演示 DRAM row-buffer locality。

## 关键代码点
- `kernel.ptx:8-15` 计算 `addr = base + (tid * STRIDE) * 4`
- 整个 kernel 只做一次 ld.global → st.global，便于隔离 HBM 行为

## 运行
```
python examples/row_buffer_demo/run.py
```

## 预期观察（Phase 2 timing mode）
- `STRIDE=1`：32 个 lane 连续读 32 个 4 字节 = 128 B = 1 cache line。所有访问命中 row buffer。`row_buffer_hit_rate ≈ 1.0`
- `STRIDE=131072` (= 512 KB / 4 = 131072 elements)：每个 lane 跳到下一个 row。每次访问 row miss。`row_buffer_hit_rate ≈ 0`
- HTML 报告 §8 (Row buffer locality) pie 图明显切换

## 为什么 stride 是 131072 而不是直觉的 row size (4 KB / 4 = 1024)
Phase 2 的 HBM 地址 layout 把 channel 放在最低位 (bits [9:7])，bank 放在 col-in-row 之上 (bits [18:15])。
要让连续访问命中"不同 row 同 bank 同 channel"，stride 必须跳过 (channel × col-in-row × bank) = 8 × 32 × 16 × 128 B = 524288 B = 131072 floats。
详见 spec §5.2。

## 延伸思考
1. 把 STRIDE 设为 32（spread across 32 cols within row 0）：still all in row 0, channel cycles. row_hit_rate 仍 ≈ 1.0
2. 把 STRIDE 设为 1024 (= row size)：col cycles 32 times within row 0 of bank 0, then bank cycles. Still row 0 in each bank. Still row_hit_rate ≈ 1.0!
