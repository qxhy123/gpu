"""Phase 5 microbench — cluster textbook facts."""
import numpy as np


def test_cluster_size_2_overhead_small():
    """cluster_size=2 vs cluster_size=1 on simple kernel — overhead < 50%."""
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd<3>;
    .reg .u32 %r<3>;
    .reg .pred %p0;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %ctaid.x;
    mul.lo.s32 %r1, %r0, 4;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, %tid.x;
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra END;
    st.global.u32 [%rd2], %r0;
END:
    ret;
}
"""
    cfg1 = load_default(); cfg1.cluster_size = 1; cfg1.n_sm = 2
    cfg2 = load_default(); cfg2.cluster_size = 2; cfg2.n_sm = 2
    out1 = np.zeros(2, dtype=np.uint32); out2 = np.zeros(2, dtype=np.uint32)
    res1 = gpusim.run(ptx_src=src, grid=(2,1,1), block=(32,1,1),
                       params={"OUT": out1}, mode="timing", config=cfg1)
    res2 = gpusim.run(ptx_src=src, grid=(2,1,1), block=(32,1,1),
                       params={"OUT": out2}, mode="timing", config=cfg2)
    ratio = res2.metrics["cycles"] / max(res1.metrics["cycles"], 1)
    assert ratio <= 1.5, f"cluster_size=2 / =1 cycle ratio = {ratio:.2f}"


def test_cluster_basic_runs_with_correct_output():
    """End-to-end smoke test for cluster_basic example."""
    import gpusim, pathlib
    from gpusim.config.loader import load_default
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "cluster_basic"
    cfg = load_default(); cfg.cluster_size = 2
    out = np.zeros(2, dtype=np.uint32)
    ptx = (base / "kernel.ptx").read_text()
    res = gpusim.run(ptx_src=ptx, grid=(2,1,1), block=(32,1,1),
                      params={"OUT": out}, mode="timing", config=cfg)
    assert (out == np.array([0, 1], dtype=np.uint32)).all()
