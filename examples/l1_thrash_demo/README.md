# l1_thrash_demo

通过 K（循环次数）和 STRIDE 配置 working set 大小，扫过 L1/L2/HBM 三档。

## 关键代码点
- `kernel.ptx:11-22` 循环 K 次，每次 stride 个 element

## 三个配置（在 run.py 中）
- **A: fits L1**：working set = 32 KB（< L1 = 128 KB）→ L1 hit rate ≈ 100% (除首轮 cold)
- **B: > L1, fits L2**：working set = 1 MB（> L1，< L2 = 4 MB）→ L1 hit < 50%, L2 hit ≈ 100%
- **C: > L2**：working set = 16 MB（> L2）→ L2 hit < 50%, HBM 流量大

## 运行
```
python examples/l1_thrash_demo/run.py
```

## 预期观察
- 三个配置 cycle 数依次显著增大
- HTML 报告 §6 (cache hit rate) 阶跃可见
- §10 eviction heatmap (only for C) 显示密集驱逐

## 延伸思考
1. 配置 D：K=1024, STRIDE=131072（极大 stride）→ row miss 显著
2. 改 `default_hopper.yaml` 的 `l1_size_bytes: 65536`（64 KB tiny L1），看 A 配置是否仍 fit
