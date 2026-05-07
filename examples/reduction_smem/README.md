# reduction_smem

单 warp 用 shared memory 做 32 元素树形归约。展示 `bar.sync` 节奏与
shared memory 的多次访问模式。

## 关键代码点
- `kernel.ptx:14` 把 gmem → smem 装载（每 lane 一个 dword）
- `kernel.ptx:15` 第一次 `bar.sync`（确保所有 lane 写完 smem 才能开始读）
- `kernel.ptx:18+` stride 16/8/4/2/1 的五次半数归约

## 运行
```
python examples/reduction_smem/run.py
```

## 预期观察
- HTML 报告中 `bar.sync` 占据可见的 cycle 比例
- 各 stride 的 `ld.shared` 没有 bank 冲突（stride 是 4 字节对齐的偶数倍，但落在不同 bank）

## 延伸思考
1. 把 stride 的下一步从 16 改成 17，看 bank conflict 直方图变化
2. 用模拟器验证：去掉中间的 `bar.sync` 会得到错误结果吗？（提示：当前是 functional 模式 ⇒ 不会；timing 模式才能看到序列化）
