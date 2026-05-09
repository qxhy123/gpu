import numpy as np, pathlib, gpusim
from gpusim.api import Stream, _reset_stream_id_counter
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()

    _reset_stream_id_counter()
    s_serial = Stream()
    outs = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i in range(4):
        s_serial.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                         params={"A": A, "B": B, "OUT": outs[i]},
                         kernel_name=f"k{i}", config=cfg)
    rs = gpusim.synchronize(streams=[s_serial], config=cfg)

    _reset_stream_id_counter()
    streams = [Stream() for _ in range(4)]
    outs2 = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i, s in enumerate(streams):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": outs2[i]},
                  kernel_name=f"k{i}", config=cfg)
    rc = gpusim.synchronize(streams=streams, config=cfg)

    print(f"Serial:     {rs.total_cycles} cycles (1 stream, 4 launches)")
    print(f"Concurrent: {rc.total_cycles} cycles (4 streams, 1 launch each)")
    print(f"Speedup:    {rs.total_cycles / max(rc.total_cycles, 1):.2f}x")


if __name__ == "__main__":
    main()
