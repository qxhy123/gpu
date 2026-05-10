# mempool_fragmentation — Phase 16

Allocate 5 blocks of mixed sizes [1024, 2048, 4096, 1024, 2048], free all,
then re-alloc 1024 and 2048 — best-fit picks the right blocks (no growth).

Then `trim_to(0)` releases all 5 fully-free slabs back to the OS.

## Run
```bash
python run.py
```
