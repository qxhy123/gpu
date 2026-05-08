# l1_thrash_demo

通过 K（每个 outer loop 触摸的 cache line 数）、STRIDE（lane 间步长）、OUTER_LOOPS
（外循环次数）配置 working set 大小，扫过 L1 / L2 / HBM 三档。

## 关键代码点
- `kernel.ptx` 外循环 `OUTER:` 重复访问相同的 K 个 cache line —— 这是 cache hit 的来源
- 内循环 `INNER:` 每次触发 32 lane × 1 element = 1 cache line（coalesced，STRIDE=32）

如果没有外循环，每个 inner iter 都触摸新 line，L1 hit rate 永远是 0%（每行都 cold miss）。
外循环让同一组 K 个 cache line 被重复访问 OUTER_LOOPS 次，cache 系统才有发挥空间。

## 三个配置（在 run.py 中）

| 配置 | K | OUTER_LOOPS | Working set | 预期行为 |
|---|---|---|---|---|
| **A** | 256 | 8 | ~32 KB | fit L1 (128 KB)，L1 hit ≈ 87% |
| **B** | 8192 | 4 | ~1 MB | > L1，fits L2 (4 MB)，L1 ≈ 0%，L2 ≈ 75% |
| **C** | 40000 | 2 | ~5 MB | > L2，L1 ≈ 0%，L2 ≈ 0%，HBM 主导 |

## 运行
```bash
python examples/l1_thrash_demo/run.py
```

实际跑出来的输出（参考值，会因模拟器内部时序略有差异）：
```
A: fits L1 (~32 KB):       cycles≈62k,    L1 hit 87.5%, L2 hit  0.0%
B: > L1, fits L2 (~1 MB):  cycles≈983k,   L1 hit  0.0%, L2 hit 75.0%
C: > L2 (~5 MB):           cycles≈2400k,  L1 hit  0.0%, L2 hit  0.0%
```

打开生成的 HTML 报告（M4 之后默认产出，需要在 timing 模式下用 `result.html_report(...)`），
看 §6 cache hierarchy hit rate 节，三档之间的对比一目了然。

## 教学讨论点
- **OUTER_LOOPS 是为什么必不可少**：单次扫过 K 个 cache line 没有重用，每行都 cold miss
- **L1 容量边界**（128 KB）的具体后果：A 配置 32KB 全部 fit；B 配置 1 MB 远超 L1 → L1 thrashing
- **L2 容量边界**（4 MB）：B 配置 1 MB fits L2 → L1 miss 大部分 L2 hit；C 配置 5 MB > L2 → L1+L2 都 miss，HBM 流量主导
- **为什么 B 的 L2 hit 不是 100%**：第一遍 OUTER 是 cold miss，所有 8192 line 从 HBM fetch；后续 OUTER 才能 hit。整体 hit rate ≈ (3/4) = 75%

## 延伸思考
1. 改 OUTER_LOOPS=1：所有配置 L1 + L2 hit rate ≈ 0%（无重用，cold miss only）
2. 改 STRIDE=33（奇数 stride）：看 L1 set 索引分布会怎样变化（避开 set 冲突）
3. 改 `default_hopper.yaml` 的 `l1_size_bytes: 65536`（64 KB tiny L1）：A 配置可能不再 fit
4. 改 `l2_size_bytes: 1048576`（1 MB tiny L2）：B 配置就会变成 C 那种 thrash 状态
