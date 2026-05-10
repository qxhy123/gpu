def test_pool_high_water_mark_zero_for_empty_recorder():
    from gpusim.analysis.metrics import pool_high_water_mark
    from gpusim.trace.recorder import Recorder
    assert pool_high_water_mark(Recorder(), pool_id=1) == 0


def test_pool_high_water_mark_tracks_peak():
    from gpusim.analysis.metrics import pool_high_water_mark
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=100, ptr_id=0,
                       reused=False, cycle=0)
    rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=200, ptr_id=1,
                       reused=False, cycle=10)   # peak = 300
    rec.pool_free(pool_id=1, stream_id=0, ptr_id=0, n_bytes=100, cycle=20)
    rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=50, ptr_id=2,
                       reused=False, cycle=30)   # in_flight = 250 (still < 300)
    assert pool_high_water_mark(rec, pool_id=1) == 300


def test_pool_high_water_mark_per_pool():
    from gpusim.analysis.metrics import pool_high_water_mark
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=100, ptr_id=0,
                       reused=False, cycle=0)
    rec.pool_allocate(pool_id=2, stream_id=0, n_bytes=500, ptr_id=1,
                       reused=False, cycle=0)
    assert pool_high_water_mark(rec, pool_id=1) == 100
    assert pool_high_water_mark(rec, pool_id=2) == 500


def test_pool_reuse_rate_zero_when_no_allocs():
    from gpusim.analysis.metrics import pool_reuse_rate
    from gpusim.trace.recorder import Recorder
    assert pool_reuse_rate(Recorder(), pool_id=1) == 0.0


def test_pool_reuse_rate_correctly_reports_fraction():
    from gpusim.analysis.metrics import pool_reuse_rate
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    for i in range(4):
        rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=64, ptr_id=i,
                           reused=(i > 0), cycle=i*10)
    assert pool_reuse_rate(rec, pool_id=1) == 0.75


def test_pool_alloc_count():
    from gpusim.analysis.metrics import pool_alloc_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    for i in range(5):
        rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=64, ptr_id=i,
                           reused=False, cycle=0)
    assert pool_alloc_count(rec, pool_id=1) == 5


def test_pool_release_total_bytes_sums_trim_events():
    from gpusim.analysis.metrics import pool_release_total_bytes
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_trim(pool_id=1, n_bytes_released=1024, cycle=0)
    rec.pool_trim(pool_id=1, n_bytes_released=2048, cycle=10)
    rec.pool_trim(pool_id=2, n_bytes_released=500, cycle=20)
    assert pool_release_total_bytes(rec, pool_id=1) == 3072
    assert pool_release_total_bytes(rec, pool_id=2) == 500


def test_pool_release_total_bytes_zero_when_no_trim():
    from gpusim.analysis.metrics import pool_release_total_bytes
    from gpusim.trace.recorder import Recorder
    assert pool_release_total_bytes(Recorder(), pool_id=1) == 0
