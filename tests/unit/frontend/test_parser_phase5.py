def test_parser_mapa_shared_cluster():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd<3>;
    .reg .u32 %r<2>;
    mapa.shared::cluster %rd0, %rd1, %r0;
}
"""
    k = parse(src, "<test>")
    assert k.instrs[0].op == "mapa.shared::cluster"
    assert len(k.instrs[0].dst) == 1 and len(k.instrs[0].src) == 2


def test_parser_ld_st_shared_cluster():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .f32 %f0;
    ld.shared::cluster.f32 %f0, [%rd0];
    st.shared::cluster.f32 [%rd0], %f0;
}
"""
    k = parse(src, "<test>")
    assert k.instrs[0].op == "ld.shared::cluster.f32"
    assert k.instrs[1].op == "st.shared::cluster.f32"


def test_parser_barrier_cluster():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    barrier.cluster.arrive;
    barrier.cluster.wait;
}
"""
    k = parse(src, "<test>")
    assert len(k.instrs) == 2
    assert k.instrs[0].op == "barrier.cluster.arrive"
    assert k.instrs[1].op == "barrier.cluster.wait"


def test_parser_mbarrier_shared_cluster():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .pred %p0;
    mbarrier.init.shared::cluster [%rd0], 4;
    mbarrier.arrive.shared::cluster [%rd0];
    mbarrier.try_wait.parity.shared::cluster %p0, [%rd0], 0;
}
"""
    k = parse(src, "<test>")
    assert k.instrs[0].op == "mbarrier.init.shared::cluster"
    assert k.instrs[1].op == "mbarrier.arrive.shared::cluster"
    assert k.instrs[2].op == "mbarrier.try_wait.parity.shared::cluster"


def test_parser_getctarank():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u32 %r0;
    getctarank.u32 %r0;
}
"""
    k = parse(src, "<test>")
    assert k.instrs[0].op == "getctarank.u32"
