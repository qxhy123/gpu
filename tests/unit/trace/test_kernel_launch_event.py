def test_recorder_records_kernel_launch():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="vec_add", grid=(8,1,1),
                     block=(32,1,1), launch_cycle=10, complete_cycle=200, n_ctas=8)
    assert len(r.kernel_launch_events) == 1
    e = r.kernel_launch_events[0]
    assert e.kernel_name == "vec_add"
    assert e.launch_cycle == 10
    assert e.complete_cycle == 200


def test_recorder_writes_kernel_launch_parquet(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.kernel_launch(stream_id=1, kernel_name="k", grid=(1,1,1),
                     block=(32,1,1), launch_cycle=0, complete_cycle=100, n_ctas=1)
    write_parquet(r, tmp_path)
    assert (tmp_path / "kernel_launch.parquet").exists()
