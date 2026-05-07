# coalescing_demo

通过 `STRIDE` 参数演示 coalesced vs strided global 访问。

## 预期观察（timing）
- stride=1: coalescing_efficiency=1.0, n_transactions=1
- stride=2: 0.5, n_transactions=2
- stride=4: 0.25, n_transactions=4
- stride=8: 0.125, n_transactions=8（依 sector 大小可能合并）

## 延伸思考
- 当 stride * sizeof(type) ≥ sector_bytes 时，每个 lane 都可能落入独立 sector
- 把 dtype 从 u32 改成 u64 时，stride 的影响会怎么变？
