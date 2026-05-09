import numpy as np, pathlib, gpusim
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    C = np.zeros(n, dtype=np.float32)
    D = np.arange(n, dtype=np.float32) * 5
    E = np.arange(n, dtype=np.float32) * 3
    F = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    cfg.n_sm = 8
    here = pathlib.Path(__file__).parent
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=(here / "kernel_compute.ptx").read_text(),
              grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": C},
              kernel_name="compute_heavy", config=cfg)
    s1.launch(ptx_src=(here / "kernel_memory.ptx").read_text(),
              grid=(1,1,1), block=(32,1,1),
              params={"A": D, "B": E, "OUT": F},
              kernel_name="memory_heavy", config=cfg)
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    print(multi_res.stream_summary())
    s0_cycles = multi_res.streams[s0.stream_id][0].metrics["cycles"]
    s1_cycles = multi_res.streams[s1.stream_id][0].metrics["cycles"]
    print(f"Per-launch cycles: s0={s0_cycles}, s1={s1_cycles}")
    print(f"Total cycles: {multi_res.total_cycles} (max-not-sum demonstrates Phase 9 overlap awareness)")
    print(f"Cross-stream concurrency gain: {multi_res.cross_stream_concurrency_gain():.2f}x")


if __name__ == "__main__":
    main()
