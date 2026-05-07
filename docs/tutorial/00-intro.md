# Chapter 00 — Introduction

## What this tutorial is

This tutorial teaches GPU microarchitecture by *running* code in a cycle-approximate simulator. Rather than reading about warps, bank conflicts, and coalescing in the abstract, you will execute real PTX kernels, observe how the simulator schedules instructions, and measure the performance effects of your changes.

The simulator (`gpusim`) models a single SM roughly shaped after NVIDIA's Hopper architecture. It is not cycle-accurate — it will not reproduce H100 performance numbers — but it *is* semantically correct: every kernel that passes a parity test against NumPy also produces bit-identical results to real CUDA on compatible inputs. The simulator's value is in exposing the microarchitectural *decisions* that an accurate GPU makes, without requiring GPU hardware.

Each chapter introduces one concept, then asks you to run a kernel, read the report, and modify the source to observe how the metrics change. The modifications are short — typically one or two lines of PTX or one config value — so you can explore at will.

## What the simulator can teach

- **SIMT execution and branch divergence** — how 32 lanes share one program counter, and what the SIMT stack costs when they disagree.
- **Warp scheduling** — LRR (Least-Recently-Run) and GTO (Greedy-Then-Oldest) policies; how multiple warps hide memory latency.
- **Global memory coalescing** — why consecutive threads should access consecutive addresses; how stride affects transaction count.
- **Shared memory bank conflicts** — the 32-bank structure, how access patterns interact with it, and how to avoid serialization.
- **Occupancy** — how registers and shared memory per thread/CTA limit the number of resident warps, and why that matters for latency hiding.
- **Reduction patterns** — tree reduction with shared memory and `bar.sync`.
- **Tiled matrix multiplication** — how to stage data through shared memory for arithmetic intensity.

## What the simulator does NOT model (Phase 1)

Phase 1 is deliberately scoped to the concepts above. The following are **not** modeled and will not appear in reports:

- **Tensor Core / MMA instructions** — no `wmma` or `mma` instructions are parsed.
- **Cache hierarchy** — there is no L1/L2 model; every global load goes directly to "global memory" with a fixed latency.
- **HBM bandwidth and queueing** — memory contention is not modeled.
- **TMA (Tensor Memory Accelerator)** — Hopper-specific async copy engines.
- **Thread-block clusters** — multi-SM coordination.
- **Warp shuffle instructions** — `shfl.sync`, `vote.sync`, etc.
- **Multi-SM, multi-GPU** — a single SM is simulated.
- **ITS (Instruction-level Tracing Spec)** — event semantics are approximate.

These will be addressed in later phases. For the full exclusion list, see the design document at `docs/superpowers/specs/2026-05-07-gpusim-phase1-design.md` section 11.

## Setup

Install the simulator and its development dependencies:

```bash
pip install -e ".[dev]"
```

Then verify all dependencies are present:

```bash
gpusim doctor
```

Expected output:

```
numpy 2.4.4
pandas 3.0.2
pyarrow 24.0.0
plotly 6.7.0
jinja2 3.1.6
pyyaml 6.0.3
OK
```

If any dependency is missing, `pip install` again. Python 3.11 or later is required.

## First run

Run the simplest example — adding two 1024-element float32 vectors:

```bash
python examples/vector_add/run.py
```

Output:

```
max abs error: 0.0
```

That confirms the simulator produced the correct numerical result. The example runs in `timing` mode by default, which models the cycle-stepped pipeline. A `report.html` file is written to the current directory.

Internally, `run.py` calls:

```python
gpusim.run(ptx_src=ptx, grid=(8,1,1), block=(128,1,1),
           params={"A": a, "B": b, "C": c, "N": n}, mode="timing")
```

The grid is `(8,1,1)` CTAs and the block is `(128,1,1)` threads per CTA, giving `8 × 128 = 1024` threads total — one per element. The kernel runs for **545 cycles** on our model SM.

## Reading a report

Open `report.html` in any browser. You will see several sections:

**Metrics summary** — a table with cycle count, occupancy bottleneck, and the active CTA count. For vector_add with `block=(128,1,1)` the occupancy is limited by `warps` (maximum warp count per SM), meaning more CTAs are resident than the SM can fully schedule but is allowed by the resource limits.

**Stall breakdown** — a bar chart showing how many cycles each warp spent in each stall state (ISSUED, SCOREBOARD, MEM_DEP, BARRIER, DIVERGENCE_SERIAL, etc.). For vector_add the dominant stall is `SCOREBOARD` — waiting for the global load latency to resolve before the addition can issue.

**Source-line attribution** — each PTX instruction is annotated with stall counts so you can see which lines are hot. In vector_add the `ld.global` and subsequent `add` are the hot lines.

**Bank conflict histogram** — shows how many shared memory accesses had each conflict degree. For vector_add there is no shared memory, so this chart is empty.

**Coalescing report** — shows the number of 128-byte cache-line transactions per global memory access. For vector_add with `tid` = consecutive integers, all 128 threads access consecutive addresses → 1 transaction per 32-thread warp → perfect coalescing.

Perfetto traces are also exported to `trace.json`. Drag this file into [https://ui.perfetto.dev](https://ui.perfetto.dev) for an interactive timeline showing instruction issue events per warp per cycle.

## Where to go next

The chapters proceed roughly in order of concept complexity:

- **Chapter 01** — SIMT model: the warp, the active mask, the SIMT stack.
- **Chapter 02** — Warp scheduler and latency hiding.
- **Chapter 03** — Global memory coalescing in detail.
- **Chapter 04** — Shared memory bank conflicts.
- **Chapter 05** — Branch divergence deep dive.
- **Chapter 06** — Occupancy analysis.
- **Chapter 07** — Tiled matrix multiplication: putting it all together.

Each chapter has a "改一改" (try it yourself) section with short modifications you can make to the PTX or config, and a "真机对照" (real-GPU comparison) section that notes what real hardware would show — or references a reference fixture if one is available.
