def test_parser_atom_global_add():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .u32 %r<3>;
    atom.global.add.u32 %r0, [%rd0], %r1;
}
"""
    k = parse(src, "<t>")
    assert k.instrs[0].op == "atom.global.add.u32"
    assert len(k.instrs[0].dst) == 1
    assert len(k.instrs[0].src) == 2


def test_parser_atom_global_cas_3_src():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .u32 %r<4>;
    atom.global.cas.u32 %r0, [%rd0], %r1, %r2;
}
"""
    k = parse(src, "<t>")
    assert k.instrs[0].op == "atom.global.cas.u32"
    assert len(k.instrs[0].src) == 3


def test_parser_atom_shared_min():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .s32 %r<3>;
    atom.shared.min.s32 %r0, [%rd0], %r1;
}
"""
    k = parse(src, "<t>")
    assert k.instrs[0].op == "atom.shared.min.s32"


def test_parser_red_global_add_no_dst():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .f32 %f0;
    red.global.add.f32 [%rd0], %f0;
}
"""
    k = parse(src, "<t>")
    assert k.instrs[0].op == "red.global.add.f32"
    assert len(k.instrs[0].dst) == 0
    assert len(k.instrs[0].src) == 2


def test_parser_red_shared_max():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .u32 %r0;
    red.shared.max.u32 [%rd0], %r0;
}
"""
    k = parse(src, "<t>")
    assert k.instrs[0].op == "red.shared.max.u32"
    assert len(k.instrs[0].dst) == 0
