def test_all_events_have_stream_id_default_zero():
    from gpusim.trace.events import (
        InstrIssue, MemoryAccess, BarrierEvent, MmaEvent, BulkLoadEvent,
        BulkStoreEvent, ClusterDispatch, ClusterBarrier, CtaDispatch,
        L2MshrEvent, AtomicEvent,
    )
    # Each event class must accept stream_id kwarg, default 0
    for cls in [InstrIssue, MemoryAccess, BarrierEvent, MmaEvent, BulkLoadEvent,
                BulkStoreEvent, ClusterDispatch, ClusterBarrier, CtaDispatch,
                L2MshrEvent, AtomicEvent]:
        # Must have stream_id field with default 0
        assert "stream_id" in cls.__dataclass_fields__, f"{cls.__name__} missing stream_id"
        assert cls.__dataclass_fields__["stream_id"].default == 0, \
            f"{cls.__name__}.stream_id default must be 0"


def test_atomic_event_accepts_stream_id_explicit():
    from gpusim.trace.events import AtomicEvent
    e = AtomicEvent(cycle=0, sm_id=0, warp_id=0, kind="ATOM",
                     op="add", space="global", line_addr=0,
                     latency=10, stream_id=3)
    assert e.stream_id == 3
