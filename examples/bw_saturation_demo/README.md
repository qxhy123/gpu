# bw_saturation_demo

多 CTA 并发流式读 HBM，演示 channel-level bandwidth saturation。

## 关键代码点
- 每 thread 读 1 element 从 HBM 写到 OUT。无算术。
- launch 配置控制并发度

## 运行
```
python examples/bw_saturation_demo/run.py
```

## 预期观察（Phase 2 timing mode）
- 低并发 (2 CTAs, 64 threads)：8 个 channel 都用不满，`channel_utilization` < 0.5
- 高并发 (64 CTAs, 2048 threads)：所有 channel 接近饱和，`channel_utilization` ≈ 1.0；`queue_wait` 分布右偏
- HTML 报告 §7 (HBM channel utilization) 直接看到差距

## 教学讨论点
- 为什么 SM 配置 64 warps 不一定带来 64× memory bandwidth？答：channel 数（8）才是 effective parallelism 上限
- "Memory-bound" kernel 的真实含义：所有 channel 已饱和

## 延伸思考
1. 用 1024 CTAs 看 `queue_wait` 分布会有多偏
2. 改 `default_hopper.yaml` 的 `channels: 16`（双倍 channel），高并发 cycle 数应近乎减半
