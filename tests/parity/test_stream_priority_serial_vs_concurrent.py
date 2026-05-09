import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "stream_priority_serial_vs_concurrent"


def test_stream_priority_serial_vs_concurrent():
    """Same total work: 4 vec_add grids serial (1 stream) vs concurrent (4 streams).
    Concurrent should not be dramatically slower."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (_DIR / "kernel.ptx").read_text()

    # Serial: 4 launches on 1 stream
    _reset_stream_id_counter()
    s_serial = Stream()
    outs_serial = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i in range(4):
        s_serial.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                         params={"A": A, "B": B, "OUT": outs_serial[i]},
                         kernel_name=f"k{i}", config=cfg)
    res_serial = gpusim.synchronize(streams=[s_serial], config=cfg)
    serial_cycles = res_serial.total_cycles

    # Concurrent: 4 streams each 1 launch
    _reset_stream_id_counter()
    streams = [Stream() for _ in range(4)]
    outs_conc = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i, s in enumerate(streams):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": outs_conc[i]},
                  kernel_name=f"k{i}", config=cfg)
    res_conc = gpusim.synchronize(streams=streams, config=cfg)
    conc_cycles = res_conc.total_cycles

    # All outputs correct
    for o in outs_serial: np.testing.assert_array_equal(o, A + B)
    for o in outs_conc: np.testing.assert_array_equal(o, A + B)

    # Concurrent should NOT take longer than 1.5x serial (Phase 7 sequential drain)
    assert conc_cycles <= serial_cycles * 1.5
