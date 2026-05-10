from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass


def _topological_sort(nodes, edges) -> list:
    """Kahn's algorithm. Raises ValueError if graph has cycles."""
    in_degree = defaultdict(int)
    children = defaultdict(list)
    node_ids = {n.node_id for n in nodes}
    for parent, child in edges:
        in_degree[child] += 1
        children[parent].append(child)
    queue = deque([nid for nid in node_ids if in_degree[nid] == 0])
    out = []
    while queue:
        nid = queue.popleft()
        out.append(nid)
        for c in children[nid]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    if len(out) != len(node_ids):
        raise ValueError("graph has a cycle")
    return out


@dataclass
class GraphExec:
    graph: object          # Graph
    topo_order: list       # list of node_ids in valid execution order
    config: object
    _recorder: object | None = None
    _graph_id: int = 0
    _launch_count: int = 0

    @classmethod
    def from_graph(cls, graph, config) -> "GraphExec":
        topo = _topological_sort(graph.nodes, graph.edges)
        return cls(graph=graph, topo_order=topo, config=config)

    def launch(self) -> int:
        """Execute all nodes in topological order. Returns total cycles."""
        from gpusim.api import Stream, synchronize
        total_cycles = 0
        for node_id in self.topo_order:
            node = next(n for n in self.graph.nodes if n.node_id == node_id)
            if node.type == "kernel":
                s = Stream()
                args = node.kernel_args
                s.launch(ptx_src=args.ptx_src, grid=args.grid, block=args.block,
                         params=args.params, kernel_name=args.kernel_name,
                         config=self.config)
                res = synchronize(streams=[s], config=self.config)
                total_cycles += res.streams[s.stream_id][0].metrics.get("cycles", 0)
            elif node.type == "memcpy":
                args = node.memcpy_args
                args.dst[:] = args.src    # functional memcpy
                total_cycles += 100        # fixed overhead per memcpy
            elif node.type == "event":
                # Event nodes: ordering enforced via topo order; no cycle cost
                pass
        if self._recorder is not None:
            self._recorder.graph_launch(
                graph_id=self._graph_id,
                n_nodes=len(self.graph.nodes),
                n_edges=len(self.graph.edges),
                launch_index=self._launch_count,
                start_cycle=0,
                end_cycle=total_cycles,
            )
        self._launch_count += 1
        return total_cycles
