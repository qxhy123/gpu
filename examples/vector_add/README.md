# vector_add

最小可运行示例：c[i] = a[i] + b[i]，N=1024。

## 关键代码点
- `kernel.ptx:14` 计算全局线程索引（mad.lo.s32 = ctaid*ntid + tid）
- `kernel.ptx:17` 越界保护（@%p1 bra END）
- `kernel.ptx:19-26` 加载、相加、写回

## 运行
```
python examples/vector_add/run.py
```

## 预期观察
- 模拟器输出 max abs error 应为 0（functional 模式精确等于 a+b）
- Milestone 5 后跑 `gpusim run examples/vector_add/kernel.ptx --grid 8 --block 128 --output report.html`，
  会看到 100% coalesced load、achieved occupancy = 100%、IPC 接近上限。

## 延伸思考
1. 把 block 从 128 改成 32，occupancy 会怎样变化？
2. 把 N 改成 1023（不对齐），尾部 warp 会发生分歧吗？
