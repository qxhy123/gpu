def test_kernel_launch_default_parent_kernel_id():
    from gpusim.trace.events import KernelLaunch
    e = KernelLaunch(stream_id=0, kernel_name="k", grid=(1,1,1), block=(32,1,1),
                       launch_cycle=0, complete_cycle=100, n_ctas=1)
    assert e.parent_kernel_id == -1
    assert e.is_persistent is False


def test_kernel_launch_with_parent_id_set():
    from gpusim.trace.events import KernelLaunch
    e = KernelLaunch(stream_id=1, kernel_name="child", grid=(1,1,1), block=(32,1,1),
                       launch_cycle=10, complete_cycle=50, n_ctas=1,
                       parent_kernel_id=0, is_persistent=False)
    assert e.parent_kernel_id == 0


def test_recorder_kernel_launch_accepts_phase14_fields():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="k", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=0, complete_cycle=100, n_ctas=1,
                     parent_kernel_id=-1, is_persistent=True)
    e = r.kernel_launch_events[-1]
    assert e.is_persistent is True
