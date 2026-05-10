# mempool_basic — Phase 16

Single stream, single pool. Alloc -> free 4 times of 1024 bytes each.

Demonstrates:
- First malloc grows the pool by 1024 bytes (1 slab).
- Each subsequent malloc reuses the freed block — 0 new growth, reuse rate 75%.
- `high_water_mark` stays at 1024 bytes throughout.

## Run
```bash
python run.py
```
