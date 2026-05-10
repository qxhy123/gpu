# mempool_train_step — Phase 16

Toy training-step pattern: each iteration allocates an activation buffer (8 KB)
and a gradient buffer (8 KB), uses them, then frees both. After warmup, every
iteration reuses the same two blocks — `high_water_mark` plateaus at 16 KB,
reuse rate approaches 1.0.

## Run
```bash
python run.py
```
