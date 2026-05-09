"""Phase 10: NVLink fabric — point-to-point links between GPUs."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class NvlinkLink:
    src_gpu: int
    dst_gpu: int
    bandwidth_gbps: float = 900.0
    latency_cycles: int = 100
    busy_until: int = 0


@dataclass
class NvlinkFabric:
    n_gpus: int
    links: dict = field(default_factory=dict)
    topology: str = "all_to_all"

    @classmethod
    def from_config(cls, cfg, n_gpus: int) -> "NvlinkFabric":
        nv_cfg = getattr(cfg, "nvlink", None)
        bw = getattr(nv_cfg, "bandwidth_gbps", 900.0) if nv_cfg else 900.0
        lat = getattr(nv_cfg, "latency_cycles", 100) if nv_cfg else 100
        topo = getattr(nv_cfg, "topology", "all_to_all") if nv_cfg else "all_to_all"
        links = {}
        for src in range(n_gpus):
            for dst in range(n_gpus):
                if src != dst:
                    links[(src, dst)] = NvlinkLink(
                        src_gpu=src, dst_gpu=dst,
                        bandwidth_gbps=bw, latency_cycles=lat,
                    )
        return cls(n_gpus=n_gpus, links=links, topology=topo)

    def transfer(self, src_gpu: int, dst_gpu: int, n_bytes: int,
                   arrival_cycle: int, *, recorder=None,
                   rank: int = -1, op_name: str = "") -> int:
        """Transfer n_bytes over (src->dst) NVLink. Returns completion cycle."""
        link = self.links.get((src_gpu, dst_gpu))
        if link is None:
            raise KeyError(f"no link {src_gpu}->{dst_gpu}")
        start = max(arrival_cycle, link.busy_until)
        bytes_per_cycle = link.bandwidth_gbps   # simplified: 1 GHz clock
        transfer_cycles = max(1, int(n_bytes / bytes_per_cycle))
        completion = start + link.latency_cycles + transfer_cycles
        link.busy_until = completion
        if recorder is not None:
            recorder.nvlink_transfer(
                src_gpu=src_gpu, dst_gpu=dst_gpu, n_bytes=n_bytes,
                start_cycle=start, end_cycle=completion,
                rank=rank, op_name=op_name,
            )
        return completion
