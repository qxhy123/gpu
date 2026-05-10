def test_recorder_pool_allocate_appends():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=64,
                       ptr_id=10, reused=False, cycle=0)
    assert len(rec.pool_allocate_events) == 1
    ev = rec.pool_allocate_events[0]
    assert ev.pool_id == 1
    assert ev.stream_id == 0
    assert ev.n_bytes == 64
    assert ev.ptr_id == 10
    assert ev.reused is False


def test_recorder_pool_free_appends():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_free(pool_id=1, stream_id=0, ptr_id=10, n_bytes=64, cycle=0)
    assert len(rec.pool_free_events) == 1
    ev = rec.pool_free_events[0]
    assert ev.pool_id == 1
    assert ev.ptr_id == 10
    assert ev.n_bytes == 64


def test_recorder_pool_grow_appends():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_grow(pool_id=1, n_bytes_added=128, cycle=0)
    assert len(rec.pool_grow_events) == 1
    assert rec.pool_grow_events[0].n_bytes_added == 128


def test_recorder_pool_trim_appends():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_trim(pool_id=1, n_bytes_released=256, cycle=0)
    assert len(rec.pool_trim_events) == 1
    assert rec.pool_trim_events[0].n_bytes_released == 256
