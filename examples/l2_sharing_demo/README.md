# l2_sharing_demo

Phase 4 demonstrates cross-SM L2 sharing. 8 CTAs read overlapping windows of a
read-only buffer; later SMs hit lines installed by earlier SMs.

## Run
```
python examples/l2_sharing_demo/run.py
```

HTML §17 shows L2 cross-SM hit rate.

## Tutorial
docs/tutorial/17-l2-sharing-cross-sm.md
