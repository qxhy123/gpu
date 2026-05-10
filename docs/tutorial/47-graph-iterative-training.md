# Chapter 47 — Graph Iterative Training

## The Training Loop Pattern

Deep learning training is the canonical graph-replay workload. Each training step executes the same sequence of operations — forward pass, backward pass, optimizer update — with different data each time. The per-step topology is fixed (same layers, same graph of operations) even though the tensors flowing through it change at every step.

CUDA Graphs match this pattern perfectly. You capture the training step once, then replay the graph for every minibatch. The captured graph holds references to the weight and gradient buffers. Between replays, you update those buffers (load new gradients, apply the optimizer) and the next replay reads the updated values automatically.

Phase 11's `graph_iterative_train_step` example demonstrates this with a simple SGD update kernel.

## The SGD Replay Pattern

```python
from gpusim.api import Stream
from gpusim.config.loader import load_default
import numpy as np

cfg = load_default()
n = 32
weights = np.zeros(n, dtype=np.float32)   # shared weight buffer
grads = np.ones(n, dtype=np.float32)      # gradient buffer (updated each step)

# Capture one SGD update step
s = Stream()
s.begin_capture()
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
          params={"WEIGHTS": weights, "GRADS": grads},
          kernel_name="sgd_update", config=cfg)
g = s.end_capture()
exec = g.instantiate(cfg)

# Training loop: replay the graph for N epochs
for epoch in range(10):
    # Optionally update grads here to simulate new gradients each step
    exec.launch()

print(f"After 10 epochs, weights[0:4]: {list(weights[0:4])}")
```

The SGD update kernel reads `GRADS` and subtracts `lr * grad` from `WEIGHTS`. After 10 replays with `grads = 1.0` everywhere and `lr = 0.01`, `weights[i] = -0.1` for all i. The weight tensor drifts monotonically as expected.

## The graph_iterative_train_step Demo

Run the full example:

```bash
python examples/graph_iterative_train_step/run.py
```

Expected output:
```
After 3 epochs, weights[0:4]: [<values>, <values>, <values>, <values>]
```

The demo runs 3 epochs. Each epoch's `exec.launch()` applies one SGD step, updating `weights` in place. The printed values reflect cumulative weight updates across all 3 replays.

## 看模拟器

**观察 weights 在多次 replay 中的漂移：**

You can observe the weight evolution across replays by sampling the weight buffer between launches:

```python
exec = g.instantiate(cfg)

history = []
for epoch in range(5):
    exec.launch()
    history.append(weights[0])   # sample first weight after each replay

print("Weight[0] trajectory:", history)
# Monotonically decreasing if grads are positive and lr > 0
```

This demonstrates a critical property: **the graph holds a reference to the live weight array, not a snapshot**. After each replay, `weights` contains the updated values. The next replay reads those updated values because the `params` dict stores a reference to the same NumPy array object, not a copy.

To visualize the trajectory:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.plot(range(len(history)), history, marker="o")
plt.xlabel("epoch")
plt.ylabel("weights[0]")
plt.title("SGD weight trajectory across graph replays")
plt.savefig("/tmp/weight_trajectory.png")
```

The plot should show a straight line with slope equal to `-lr * grads[0]` per epoch.

## 改一kasnije

**在相邻 replay 之间修改 grads，模拟真实训练数据变化：**

In a real training loop, gradients change with each minibatch. Modify `grads` between replays to simulate this:

```python
exec = g.instantiate(cfg)
rng = np.random.default_rng(seed=42)

for epoch in range(10):
    # Simulate a new gradient computed from this step's minibatch
    grads[:] = rng.standard_normal(n).astype(np.float32)
    exec.launch()

print(f"Final weights[0:4]: {list(weights[0:4])}")
```

Because `grads` is read at launch time (not at capture time), the graph always uses the most recent gradient values. The captured graph effectively says: "at launch time, read whatever is in the `grads` buffer." This is the correct semantics for training — you don't want the graph to freeze the gradients from the first step.

This contrasts with compile-time constant folding in XLA/JAX: if you `jit` a function with a captured constant tensor, the constant is baked in. CUDA Graphs with device-pointer parameters do not fold constants — the pointer is fixed but the data at that address is live.

## 真机对照

PyTorch provides `make_graphed_callables` as a high-level wrapper:

```python
import torch
import torch.nn as nn

model = nn.Linear(64, 64).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# make_graphed_callables captures forward + backward + optimizer step
graphed_model = torch.cuda.graphs.make_graphed_callables(model, (sample_input,))

# Training loop — graphs replay under the hood
for batch in dataloader:
    x, y = batch
    loss = criterion(graphed_model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

| Behavior | Simulator | PyTorch `make_graphed_callables` |
|---|---|---|
| **Capture target** | SGD kernel only | Forward + backward + optimizer |
| **Gradient update** | Explicit array write before replay | Autograd computes grads each step |
| **Weight update** | SGD kernel in graph | Optimizer step in graph |
| **New data injection** | Modify NumPy arrays in place | Input tensors updated via `copy_` |

The simulator's `graph_iterative_train_step` captures only the optimizer kernel for simplicity. Real PyTorch graph capturing typically covers the forward and backward pass as well, using `torch.cuda.graph` context manager with a captured static graph. The key insight — that tensors are references, not copies — applies equally to both.

## Phase 11 Summary

Chapters 44–47 have covered Phase 11's CUDA Graphs feature set:

- **Chapter 44**: Explicit graph builder — `add_kernel_node`, `add_dependency`, `instantiate`, `launch`.
- **Chapter 45**: Stream capture — `begin_capture`, `end_capture`, implicit dependency from launch order.
- **Chapter 46**: Replay amortization — deterministic cycle counts, `graph_replay_amortization` metric.
- **Chapter 47**: Iterative training — SGD-style weight update via repeated graph replay with live tensor references.

Together these chapters establish the simulator's CUDA Graphs layer. The three graph metrics (`graph_dag_depth`, `graph_node_type_breakdown`, `graph_replay_amortization`) give you quantitative tools to characterize any graph workload. Phase 12 will extend this foundation with multi-stream graph concurrency and graph update (node parameter patching without re-instantiation).
