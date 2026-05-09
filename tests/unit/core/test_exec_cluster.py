def test_mapa_encodes_rank_and_offset():
    """mapa.shared::cluster encodes (rank << 24) | offset."""
    from gpusim.frontend.parser import parse
    from gpusim.core.exec import (
        InstrExecutor, GlobalMemory, SharedMemory, ParamSpace, WarpFnState,
    )
    src = """
.entry test() {
    .reg .u64 %rd<3>;
    .reg .u32 %r<2>;
    mov.u64 %rd1, 100;
    mov.u32 %r0, 3;
    mapa.shared::cluster %rd2, %rd1, %r0;
}
"""
    k = parse(src, "<t>")
    ex = InstrExecutor(kernel=k, gmem=GlobalMemory(), smem=SharedMemory(),
                        params=ParamSpace({}), cta_id=0,
                        ctaid=(0,0,0), nctaid=(1,1,1), ntid=(1,1,1))
    ex.cluster_id = 0; ex.cluster_rank = 0; ex.cluster_size = 4
    w = WarpFnState(warp_size=1, tids=(0,))
    for instr in k.instrs:
        ex.execute(w, instr)
    encoded = w.threads[0].get_u64("rd2")
    assert encoded == ((3 << 24) | 100)


def test_dsmem_ld_st_routes_to_target_cta():
    """ld.shared::cluster.f32 routes to remote CTA's smem."""
    from gpusim.frontend.parser import parse
    from gpusim.core.exec import (
        InstrExecutor, GlobalMemory, SharedMemory, ParamSpace, WarpFnState,
    )
    smem = SharedMemory(size_bytes=8192)
    smem.allocate_cta(0, 1024); smem.allocate_cta(1, 1024)
    smem.store_f32(1, 16, 42.0)   # CTA 1's smem at offset 16

    src = """
.entry test() {
    .reg .u64 %rd<2>;
    .reg .f32 %f0;
    mov.u64 %rd0, 16777232;
    ld.shared::cluster.f32 %f0, [%rd0];
}
"""
    # 16777232 = (1 << 24) | 16
    k = parse(src, "<t>")
    ex = InstrExecutor(kernel=k, gmem=GlobalMemory(), smem=smem,
                        params=ParamSpace({}), cta_id=0,
                        ctaid=(0,0,0), nctaid=(1,1,1), ntid=(1,1,1))
    ex.cluster_id = 0; ex.cluster_rank = 0; ex.cluster_size = 2
    w = WarpFnState(warp_size=1, tids=(0,))
    for instr in k.instrs:
        ex.execute(w, instr)
    assert w.threads[0].get_f32("f0") == 42.0


def test_dsmem_st_routes_to_target_cta():
    """st.shared::cluster.f32 stores to remote CTA's smem."""
    from gpusim.frontend.parser import parse
    from gpusim.core.exec import (
        InstrExecutor, GlobalMemory, SharedMemory, ParamSpace, WarpFnState,
    )
    smem = SharedMemory(size_bytes=8192)
    smem.allocate_cta(0, 1024); smem.allocate_cta(1, 1024)

    src = """
.entry test() {
    .reg .u64 %rd<2>;
    .reg .f32 %f0;
    mov.u64 %rd0, 16777248;
    st.shared::cluster.f32 [%rd0], %f0;
}
"""
    # 16777248 = (1 << 24) | 32
    k = parse(src, "<t>")
    ex = InstrExecutor(kernel=k, gmem=GlobalMemory(), smem=smem,
                        params=ParamSpace({}), cta_id=0,
                        ctaid=(0,0,0), nctaid=(1,1,1), ntid=(1,1,1))
    ex.cluster_id = 0; ex.cluster_rank = 0; ex.cluster_size = 2
    w = WarpFnState(warp_size=1, tids=(0,))
    # Pre-load %f0 = 60.0 (0f42700000 literal not supported by parser)
    w.threads[0].set_f32("f0", 60.0)
    for instr in k.instrs:
        ex.execute(w, instr)
    assert smem.load_f32(1, 32) == 60.0


def test_getctarank_returns_cluster_rank():
    from gpusim.frontend.parser import parse
    from gpusim.core.exec import (
        InstrExecutor, GlobalMemory, SharedMemory, ParamSpace, WarpFnState,
    )
    src = """
.entry test() {
    .reg .u32 %r0;
    getctarank.u32 %r0;
}
"""
    k = parse(src, "<t>")
    ex = InstrExecutor(kernel=k, gmem=GlobalMemory(), smem=SharedMemory(),
                        params=ParamSpace({}), cta_id=0,
                        ctaid=(0,0,0), nctaid=(1,1,1), ntid=(1,1,1))
    ex.cluster_id = 0; ex.cluster_rank = 5; ex.cluster_size = 8
    w = WarpFnState(warp_size=1, tids=(0,))
    for instr in k.instrs:
        ex.execute(w, instr)
    assert w.threads[0].get_u32("r0") == 5
