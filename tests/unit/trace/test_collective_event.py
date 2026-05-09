def test_recorder_records_nvlink_transfer():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.nvlink_transfer(src_gpu=0, dst_gpu=1, n_bytes=1024,
                        start_cycle=0, end_cycle=100, rank=0, op_name="allreduce")
    assert len(r.nvlink_transfer_events) == 1
    e = r.nvlink_transfer_events[0]
    assert e.src_gpu == 0
    assert e.n_bytes == 1024


def test_recorder_records_collective_op():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.collective(op_name="allreduce", algorithm="ring", n_bytes=4096,
                   world_size=4, start_cycle=0, end_cycle=300, n_steps=6)
    assert len(r.collective_events) == 1
    e = r.collective_events[0]
    assert e.algorithm == "ring"


def test_recorder_writes_collective_parquets(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.nvlink_transfer(src_gpu=0, dst_gpu=1, n_bytes=64,
                        start_cycle=0, end_cycle=10)
    r.collective(op_name="broadcast", algorithm="linear", n_bytes=64,
                   world_size=4, start_cycle=0, end_cycle=10, n_steps=3)
    write_parquet(r, tmp_path)
    assert (tmp_path / "nvlink_transfer.parquet").exists()
    assert (tmp_path / "collective.parquet").exists()
