# send_recv_pipeline_parallel

Phase 12 demo: 4-GPU pipeline parallelism. Each rank sends activation to next
in a forward-pass chain (rank 0→1→2→3).

## Run
```
python examples/send_recv_pipeline_parallel/run.py
```

## Tutorial
docs/tutorial/49-send-recv-pipeline-parallel.md
