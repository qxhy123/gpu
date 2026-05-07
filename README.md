# gpusim

Teaching-oriented NVIDIA GPU microarchitecture simulator.

See `docs/superpowers/specs/2026-05-07-gpusim-phase1-design.md` for the design.

## Install (dev)
```
pip install -e ".[dev]"
```

## Quick start
```
gpusim run examples/vector_add/kernel.ptx --grid 8 --block 128 --output report.html
```

## Tests
```
pytest
```
