import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.schema import SMConfig
from gpusim.core.sm import SM


def test_wgmma_issues_when_all_4_warps_arrive():
    """Setup: 128-thread block (4 warps in one warp-group). Single wgmma.
    All 4 warps must arrive at the wgmma PC before issue."""
    src = """
.entry test()
{
    .reg .u64 %rd0;
    .reg .f32 %d<64>;
    .reg .f32 %c<64>;
    wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
        {%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7,
         %d8, %d9, %d10, %d11, %d12, %d13, %d14, %d15,
         %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23,
         %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31,
         %d32, %d33, %d34, %d35, %d36, %d37, %d38, %d39,
         %d40, %d41, %d42, %d43, %d44, %d45, %d46, %d47,
         %d48, %d49, %d50, %d51, %d52, %d53, %d54, %d55,
         %d56, %d57, %d58, %d59, %d60, %d61, %d62, %d63},
        %rd0,
        %rd0;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
}
"""
    k = parse(src, "<test>")
    cfg = SMConfig()
    sm = SM(cfg)
    res = sm.run(kernel=k, grid=(1, 1, 1), block=(128, 1, 1), params={})
    assert res.cycles > 0
    assert res.cycles < 10_000
