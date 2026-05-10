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
class MemsetNodeArgs:
    buf: object
    value: int
    n_bytes: int


@dataclass
class ChildGraphNodeArgs:
    graph: object


@dataclass
class ConditionalNodeArgs:
    cond_fn: object       # Callable[[], bool], evaluated at exec time
    true_graph: object    # Graph
    false_graph: object   # Graph (may be empty)


@dataclass
class WhileNodeArgs:
    cond_fn: object       # Callable[[], bool], re-evaluated each iteration
    body_graph: object    # Graph
    max_iterations: int = 1000


@dataclass
class GraphNode:
    node_id: int
    type: str             # "kernel" | "memcpy" | "event" | "memset" | "child_graph" | "conditional" | "while"
    kernel_args: KernelNodeArgs | None = None
    memcpy_args: MemcpyNodeArgs | None = None
    event_args: EventNodeArgs | None = None
    memset_args: MemsetNodeArgs | None = None    # NEW Phase 13
    child_graph_args: ChildGraphNodeArgs | None = None    # NEW Phase 13
    conditional_args: ConditionalNodeArgs | None = None    # NEW Phase 15
    while_args: WhileNodeArgs | None = None                # NEW Phase 15
