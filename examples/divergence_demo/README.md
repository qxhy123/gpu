# divergence_demo

同一 warp 的 32 lane 因 `tid<16` 走两条不同路径，演示 SIMT 序列化。

## 预期观察
- 报告中 `DIV_PUSH` 事件出现一次（在 setp/bra 处）
- `DIVERGENCE_SERIAL` 占总 cycle 的可观察比例
- 两条路径串行执行 → 总 cycle ≈ 两路径独立执行之和

## 延伸思考
1. 把分歧改成 `tid % 2 == 0`，看 `DIVERGENCE_SERIAL` 占比变化
2. 嵌套两层分歧，画 SIMT 栈深度时序
