# bank_conflict_demo

shared memory 32-bank 访问模式对比。当前 PTX 是 stride=1（无冲突）。复制 kernel.ptx
为 `kernel_stride32.ptx` 并把 `shl.b32 %r2, %r1, 2;` 改成 `shl.b32 %r2, %r1, 7;`
（×128）即可得到 32-way 冲突版本。

## 预期观察（timing）
- stride=1: bank_conflict_hist 全部为 1
- stride=32: 一次 store 的 conflict_degree=32，cycles 多出约 31 个
- broadcast (`mov.u32 %r2, 0`): conflict_degree=1（broadcast）

## 延伸思考
- 把 stride 改成 33 会怎样？（提示：奇 stride 与 32 互质 → 无冲突）
