from gpusim.frontend.parser import parse
from gpusim.frontend.ir import PtxType

def test_parse_minimal_kernel():
    src = """
    .visible .entry vec_add(
        .param .u64 A,
        .param .u64 B,
        .param .u32 N
    )
    {
        .reg .u32 %r<4>;
        .reg .u64 %rd<3>;
        .reg .pred %p<2>;
        .reg .f32 %f<2>;
    }
    """
    k = parse(src, "<t>")
    assert k.name == "vec_add"
    assert [p.name for p in k.params] == ["A", "B", "N"]
    assert [p.type for p in k.params] == [PtxType.u64, PtxType.u64, PtxType.u32]
    assert k.regs.u32 == 4
    assert k.regs.u64 == 3
    assert k.regs.pred == 2
    assert k.regs.f32 == 2
    assert k.instrs == ()

def test_parse_no_params():
    src = ".visible .entry empty() { .reg .u32 %r<1>; }"
    k = parse(src, "<t>")
    assert k.name == "empty"
    assert k.params == ()
