# multi_sm_scheduler

Phase 4 Multi-SM CTA scheduler demo. 16 CTAs are dispatched across 8 SMs.
Even-id CTAs run a small extra busy loop creating load imbalance.

Two configurations:
- `cta_policy: rr` — round-robin: deterministic, ignores load
- `cta_policy: greedy` — picks SM with fewest active warps

## Run
```
python examples/multi_sm_scheduler/run.py
```

Look for: greedy total cycles ≤ RR total cycles in steady state.

## Tutorial
docs/tutorial/16-multi-sm-cta-scheduling.md
