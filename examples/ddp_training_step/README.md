# ddp_training_step

Phase 10 capstone: 4-GPU DDP-style training step.
1. Each rank computes gradients (vec_add)
2. Allreduce gradients across all ranks (ring algorithm)
3. Broadcast updated weights from rank 0

## Run
```
python examples/ddp_training_step/run.py
```

## Tutorial
docs/tutorial/43-ddp-training-pattern.md
