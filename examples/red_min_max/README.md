# red_min_max

Phase 6 `red.global.{min,max}.s32` demo. 256 threads each emit their value
into a global min and max via `red.global` (no return register).

Compare with `atom.global.{min,max}` (same op but with returned old value).
Hardware difference: red has no return path → marginally faster (Phase 6
simulator approximates as same latency; spec §11 noted).

## Run
```
python examples/red_min_max/run.py
```

## Tutorial
docs/tutorial/26-red-vs-atom.md
