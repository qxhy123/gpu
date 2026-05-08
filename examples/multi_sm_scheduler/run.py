import numpy as np
import pathlib
import sys
import gpusim
from gpusim.config.loader import load_default

_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_DIR.parent.parent))


def main():
    rng = np.random.RandomState(0)
    n_cta = 16
    base = (rng.rand(n_cta * 32) * 100).astype(np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    print("# multi_sm_scheduler: RR vs greedy")
    for policy in ("rr", "greedy"):
        cfg = load_default()
        cfg.n_sm = 8
        cfg.scheduler.cta_policy = policy
        out = np.zeros(n_cta * 32, dtype=np.float32)
        res = gpusim.run(
            ptx_src=ptx, grid=(n_cta, 1, 1), block=(32, 1, 1),
            params={"BASE": base.copy(), "OUT": out},
            mode="timing", config=cfg,
        )
        from examples.multi_sm_scheduler.reference import reference
        expected = reference(base)
        diff = float(np.max(np.abs(out - expected)))
        print(f"  {policy:<7}: cycles={res.metrics['cycles']}, max diff={diff:.2e}")


if __name__ == "__main__":
    main()
