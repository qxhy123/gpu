from __future__ import annotations
from dataclasses import dataclass


@dataclass
class KernelNodeArgs:
    ptx_src: str
    grid: tuple
    block: tuple
    params: dict          # ndarray references; replay reads at launch time
    kernel_name: str = "<unnamed>"


@dataclass
class MemcpyNodeArgs:
    src: object           # ndarray
    dst: object           # ndarray
    n_bytes: int


@dataclass
class EventNodeArgs:
    event: object         # gpusim.api.Event (Phase 8)
    op: str               # "record" | "wait"


@dataclass
class GraphNode:
    node_id: int
    type: str             # "kernel" | "memcpy" | "event"
    kernel_args: KernelNodeArgs | None = None
    memcpy_args: MemcpyNodeArgs | None = None
    event_args: EventNodeArgs | None = None
