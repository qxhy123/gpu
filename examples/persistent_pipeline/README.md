# persistent_pipeline

Phase 14 capstone demo: a 2-stage producer-consumer pipeline using shared WorkQueues.
The producer PersistentKernel writes to buffers; the consumer PersistentKernel
processes the same buffers. Both stages drain via WorkQueue.

## Run
```
python examples/persistent_pipeline/run.py
```

## Tutorial
docs/tutorial/57-persistent-pipeline.md
