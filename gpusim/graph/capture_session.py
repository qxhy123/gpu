from __future__ import annotations
from dataclasses import dataclass, field
from gpusim.graph.graph import Graph


@dataclass
class CaptureSession:
    """Shared capture state for multiple Streams capturing into one Graph (Phase 15)."""

    graph: Graph = field(default_factory=Graph)
    streams: list = field(default_factory=list)
    _event_source_node: dict = field(default_factory=dict)    # event_id -> node_id

    def __post_init__(self):
        self.graph.is_captured = True

    def attach(self, stream) -> None:
        """Attach a Stream to this session. The Stream now captures into self.graph."""
        if stream in self.streams:
            raise RuntimeError(
                f"stream {stream.stream_id} already attached to this CaptureSession"
            )
        if stream._captured_graph is not None:
            raise RuntimeError(
                f"stream {stream.stream_id} is already capturing standalone"
            )
        self.streams.append(stream)
        stream._captured_graph = self.graph
        stream._capture_last_node = None
        stream._capture_session = self

    def end(self) -> Graph:
        """End the session, detach all streams, return the shared graph."""
        for s in self.streams:
            s._captured_graph = None
            s._capture_last_node = None
            s._capture_session = None
        self.streams = []
        self._event_source_node = {}
        return self.graph

    def register_event_source(self, event_id: int, node_id: int) -> None:
        self._event_source_node[event_id] = node_id

    def lookup_event_source(self, event_id: int):
        return self._event_source_node.get(event_id)
