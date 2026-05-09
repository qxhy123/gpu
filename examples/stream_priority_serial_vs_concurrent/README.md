# stream_priority_serial_vs_concurrent

Phase 7 demo: same workload (4 vector_add launches) run two ways:
- Serial (4 launches on 1 stream)
- Concurrent (4 launches on 4 streams)

Compares total cycles to demonstrate stream concurrency benefit.

## Run
```
python examples/stream_priority_serial_vs_concurrent/run.py
```

## Tutorial
docs/tutorial/30-scheduler-fairness-streams.md
