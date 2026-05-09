# Chapter 40 — Multi-GPU System and NVLink Fabric

## From One GPU to Many

A single H100 SXM5 delivers 3.35 TFLOPS of FP64 and 80 GB of HBM3. For training large language models those numbers are real but insufficient — GPT-4 class models require hundreds of GPUs working in concert. The simulator models this scale-out through two classes introduced in Phase 10: `MultiGpuSystem` and `NvlinkFabric`.

Phase 10 adds `cfg.n_gpus` to `DeviceConfig`. Setting this field to N causes `MultiGpuSystem.from_config(cfg)` to instantiate N `GPU` objects and wire them into an all-to-all NVLink fabric:

```python
cfg = load_default()
cfg.n_gpus = 4
sys = MultiGpuSystem.from_config(cfg)
```

After this call, `sys.gpus` is a list of four GPU objects, each with its own scheduler, cache hierarchy, and HBM model. `sys.nvlink_fabric` is an `NvlinkFabric` instance with 4×3 = 12 unidirectional `NvlinkLink` objects covering every ordered GPU pair.

## Topology: All-to-All and NVSwitch

The default topology (`"all_to_all"`) gives every GPU a direct lane to every other GPU. This corresponds to the NVSwitch-based fabric used in DGX H100 and HGX H100 nodes. NVSwitch is a high-radix crossbar that eliminates multi-hop routing: GPU 0 communicates with GPU 7 in a single hop through the switch silicon, just as it communicates with GPU 1.

H100 NVLink 4 provides 900 GB/s aggregate bidirectional bandwidth per GPU (18 links × 50 GB/s each). The simulator encodes this as `bandwidth_gbps = 900.0` in `NvlinkConfig`. Each call to `NvlinkFabric.transfer(src, dst, n_bytes, arrival_cycle)` occupies the link for `ceil(n_bytes / bandwidth_gbps)` cycles at a 1 GHz clock, plus a fixed `latency_cycles = 100` cycle latency.

## Running the Multi-GPU Setup Demo

```bash
python examples/multi_gpu_setup/run.py
```

The demo creates a 2-GPU system and launches a vector-add kernel on each GPU independently, then reports fabric topology information:

```python
cfg = load_default()
cfg.n_gpus = 2
sys = MultiGpuSystem.from_config(cfg)
# ... launch kernels per GPU ...
print(f"NVLink topology: {len(sys.nvlink_fabric.links)} links across {sys.nvlink_fabric.n_gpus} GPUs")
```

Output shows 2 GPUs and 2 unidirectional links (GPU0→GPU1 and GPU1→GPU0). Each link carries independent traffic; there is no shared-link serialization between the two directions.

## 看模拟器

**读取 nvlink_bandwidth_utilization 指标：**

After running a collective, inspect bandwidth utilization by comparing the bytes transferred to the theoretical maximum:

```python
from gpusim.comm.system import MultiGpuSystem
from gpusim.comm.nvlink import NvlinkFabric
from gpusim.config.loader import load_default

cfg = load_default()
cfg.n_gpus = 4
sys = MultiGpuSystem.from_config(cfg)

# Perform a direct transfer on one link
link = sys.nvlink_fabric.links[(0, 1)]
n_bytes = 1024 * 1024  # 1 MB
end_cycle = sys.nvlink_fabric.transfer(
    src_gpu=0, dst_gpu=1, n_bytes=n_bytes, arrival_cycle=0
)

transfer_cycles = end_cycle - link.latency_cycles
theoretical_min = n_bytes / link.bandwidth_gbps  # at peak BW
utilization = theoretical_min / transfer_cycles
print(f"Utilization: {utilization:.2%}")  # approaches 1.0 for large transfers
```

For large transfers the link is nearly saturated (utilization close to 100 %). For tiny transfers the fixed `latency_cycles` overhead dominates and utilization drops below 1 %. This mirrors the real NVLink behavior where small messages are latency-bound.

## 改一改

**把 n_gpus 从 2 改到 8，观察 fabric 链路数的变化：**

```python
for n in [2, 4, 8]:
    cfg = load_default()
    cfg.n_gpus = n
    sys = MultiGpuSystem.from_config(cfg)
    n_links = len(sys.nvlink_fabric.links)
    print(f"n_gpus={n}: {n_links} links  (expected {n*(n-1)})")
```

The link count grows as N×(N-1): 2, 12, 56 for N=2, 4, 8 respectively. On a real DGX H100 8-GPU node, there are 8 NVSwitch chips and 72 NVLink 4 ports per GPU, so every GPU can reach all others at full bandwidth simultaneously. In the simulator, the all-to-all topology approximates this by providing a direct link for every pair.

Try setting `cfg.nvlink.bandwidth_gbps = 450.0` (half the default) and measure how allreduce latency scales — it should double for bandwidth-bound transfers but remain the same for tiny messages where latency dominates.

## 真机对照

On real hardware, you query NVLink topology with `nvidia-smi topo -m`. A DGX H100 system prints a matrix showing `NVL` connections (NVLink) versus `SYS` (PCIe across NUMA domains) between every GPU pair.

| Connection type | Simulator | H100 DGX |
|---|---|---|
| **GPU-GPU BW** | `bandwidth_gbps` (default 900) | 900 GB/s bidirectional per GPU |
| **Topology** | `all_to_all` | NVSwitch crossbar, fully non-blocking |
| **Latency** | `latency_cycles` (default 100) | ~1–2 µs end-to-end |
| **Link count** | N×(N-1) per direction | 18 NVLink 4 per GPU |

H100 NVLink 4 vs. A100 NVLink 3: the per-GPU aggregate bandwidth doubled from 600 GB/s (A100) to 900 GB/s (H100), achieved by increasing from 12 to 18 links and raising per-link speed from 50 to 50 GB/s. NVSwitch 3 (used in H100 DGX) handles 3.6 TB/s of total switch bandwidth across all 8 GPUs — more than enough to keep all 8 GPUs simultaneously saturating their NVLink ports.

The simulator models the bandwidth accurately but abstracts away the internal switch routing, NVSwitch chip-to-chip wiring, and PCIe fallback paths. For topology-sensitive collective algorithms (e.g., ring vs. double-binary-tree), the all-to-all fabric means every algorithm sees the same per-hop latency regardless of neighbor distance.
