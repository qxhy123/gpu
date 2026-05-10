from __future__ import annotations
from dataclasses import dataclass, field
from gpusim.graph.node import (
    GraphNode, KernelNodeArgs, MemcpyNodeArgs, EventNodeArgs,
    ConditionalNodeArgs, WhileNodeArgs,
)


@dataclass
class Graph:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)   # [(parent_id, child_id), ...]
    _next_id: int = 0
    is_captured: bool = False    # NEW Phase 15 — True iff produced by Stream.end_capture

    def add_kernel_node(self, *, ptx_src: str, grid: tuple, block: tuple,
                        params: dict, kernel_name: str = "<unnamed>") -> int:
        node_id = self._next_id
        self._next_id += 1
        args = KernelNodeArgs(ptx_src=ptx_src, grid=grid, block=block,
                              params=params, kernel_name=kernel_name)
        self.nodes.append(GraphNode(node_id=node_id, type="kernel", kernel_args=args))
        return node_id

    def add_memcpy_node(self, *, src, dst, n_bytes: int) -> int:
        node_id = self._next_id
        self._next_id += 1
        args = MemcpyNodeArgs(src=src, dst=dst, n_bytes=n_bytes)
        self.nodes.append(GraphNode(node_id=node_id, type="memcpy", memcpy_args=args))
        return node_id

    def add_event_node(self, *, event, op: str) -> int:
        node_id = self._next_id
        self._next_id += 1
        args = EventNodeArgs(event=event, op=op)
        self.nodes.append(GraphNode(node_id=node_id, type="event", event_args=args))
        return node_id

    def add_memset_node(self, *, buf, value: int, n_bytes: int) -> int:
        from gpusim.graph.node import MemsetNodeArgs
        nid = self._next_id; self._next_id += 1
        args = MemsetNodeArgs(buf=buf, value=value, n_bytes=n_bytes)
        self.nodes.append(GraphNode(node_id=nid, type="memset", memset_args=args))
        return nid

    def add_child_graph_node(self, *, graph: "Graph") -> int:
        from gpusim.graph.node import ChildGraphNodeArgs
        nid = self._next_id; self._next_id += 1
        args = ChildGraphNodeArgs(graph=graph)
        self.nodes.append(GraphNode(node_id=nid, type="child_graph",
                                       child_graph_args=args))
        return nid

    def add_conditional_node(self, *, cond_fn, true_graph: "Graph",
                                false_graph: "Graph") -> int:
        nid = self._next_id; self._next_id += 1
        args = ConditionalNodeArgs(cond_fn=cond_fn, true_graph=true_graph,
                                      false_graph=false_graph)
        self.nodes.append(GraphNode(node_id=nid, type="conditional",
                                       conditional_args=args))
        return nid

    def add_while_node(self, *, cond_fn, body_graph: "Graph",
                          max_iterations: int = 1000) -> int:
        nid = self._next_id; self._next_id += 1
        args = WhileNodeArgs(cond_fn=cond_fn, body_graph=body_graph,
                                max_iterations=max_iterations)
        self.nodes.append(GraphNode(node_id=nid, type="while",
                                       while_args=args))
        return nid

    def add_dependency(self, parent_id: int, child_id: int) -> None:
        if parent_id == child_id:
            raise ValueError("self-dependency not allowed")
        self.edges.append((parent_id, child_id))

    def instantiate(self, config) -> "GraphExec":
        from gpusim.graph.exec import GraphExec
        return GraphExec.from_graph(self, config)
