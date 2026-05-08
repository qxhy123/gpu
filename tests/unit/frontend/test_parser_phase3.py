from gpusim.frontend.parser import parse


def _wrap_kernel(body: str) -> str:
    return f"""
.entry test() {{
    .reg .f16 %h<8>;
    {body}
}}
"""


def test_parser_brace_list_two_regs():
    from gpusim.frontend.parser import _Parser
    src = "{%h0, %h1}"
    p = _Parser(src, "<test>")
    op = p._parse_brace_list()
    from gpusim.frontend.ir import RegGroup
    assert isinstance(op, RegGroup)
    assert len(op.regs) == 2
    assert op.regs[0].name == "h0"
    assert op.regs[1].name == "h1"


def test_parser_brace_list_eight_regs():
    from gpusim.frontend.parser import _Parser
    src = "{%h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7}"
    p = _Parser(src, "<test>")
    op = p._parse_brace_list()
    assert len(op.regs) == 8


def test_parser_tma_desc_pseudo_instr():
    src = """
.entry test()
{
    .reg .u64 %rd<2>;
    gpusim.tma_desc %rd0, %rd1, 128, 64, 512, 2;
}
"""
    k = parse(src, "<test>")
    assert len(k.instrs) == 1
    instr = k.instrs[0]
    assert instr.op == "gpusim.tma_desc"
    # dst[0] = handle reg, src[0] = gmem_base_reg, src[1..4] = dim_x, dim_y, stride_y, elem_bytes
    from gpusim.frontend.ir import Reg, Imm
    assert isinstance(instr.dst[0], Reg) and instr.dst[0].name == "rd0"
    assert isinstance(instr.src[0], Reg) and instr.src[0].name == "rd1"
    assert isinstance(instr.src[1], Imm) and instr.src[1].value == 128
    assert isinstance(instr.src[2], Imm) and instr.src[2].value == 64
    assert isinstance(instr.src[3], Imm) and instr.src[3].value == 512
    assert isinstance(instr.src[4], Imm) and instr.src[4].value == 2


def test_parser_sync_mma_with_brace_lists():
    src = """
.entry test()
{
    .reg .f32 %d<4>;
    .reg .f16 %a<8>;
    .reg .f16 %b<4>;
    .reg .f32 %c<4>;
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {%d0, %d1, %d2, %d3},
        {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7},
        {%b0, %b1, %b2, %b3},
        {%c0, %c1, %c2, %c3};
}
"""
    k = parse(src, "<test>")
    assert len(k.instrs) == 1
    i = k.instrs[0]
    assert i.op.startswith("mma.sync.aligned.m16n8k16")
    from gpusim.frontend.ir import RegGroup
    assert len(i.dst) == 1 and isinstance(i.dst[0], RegGroup)
    assert len(i.dst[0].regs) == 4
    assert len(i.src) == 3
    assert all(isinstance(s, RegGroup) for s in i.src)
    assert len(i.src[0].regs) == 8
    assert len(i.src[1].regs) == 4
    assert len(i.src[2].regs) == 4


def test_parser_wgmma_compute():
    src = """
.entry test()
{
    .reg .f32 %d<64>;
    .reg .u64 %rd<2>;
    wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
        {%d0, %d1, %d2, %d3},
        %rd0,
        %rd1;
}
"""
    k = parse(src, "<test>")
    i = k.instrs[0]
    assert i.op == "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16"


def test_parser_wgmma_fence_commit_wait():
    src = """
.entry test()
{
    wgmma.fence.sync.aligned;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
}
"""
    k = parse(src, "<test>")
    assert len(k.instrs) == 3
    assert k.instrs[0].op == "wgmma.fence.sync.aligned"
    assert k.instrs[1].op == "wgmma.commit_group.sync.aligned"
    assert k.instrs[2].op == "wgmma.wait_group.sync.aligned"
    from gpusim.frontend.ir import Imm
    assert isinstance(k.instrs[2].src[0], Imm) and k.instrs[2].src[0].value == 0


def test_parser_cp_async_bulk_tensor_2d():
    src = """
.entry test()
{
    .reg .u64 %rd<3>;
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
        [%rd0], [%rd1], [%rd2];
}
"""
    k = parse(src, "<test>")
    i = k.instrs[0]
    assert i.op == "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    assert len(i.src) == 3


def test_parser_mbarrier_ops():
    src = """
.entry test()
{
    .reg .u64 %rd0;
    .reg .pred %p0;
    mbarrier.init.shared::cta [%rd0], 4;
    mbarrier.arrive.shared::cta [%rd0];
    mbarrier.try_wait.parity.shared::cta %p0, [%rd0], 0;
}
"""
    k = parse(src, "<test>")
    assert len(k.instrs) == 3
    assert k.instrs[0].op == "mbarrier.init.shared::cta"
    assert k.instrs[1].op == "mbarrier.arrive.shared::cta"
    assert k.instrs[2].op == "mbarrier.try_wait.parity.shared::cta"


def test_parser_cp_async_bulk_store():
    from gpusim.frontend.parser import parse
    src = """
.entry test() {
    .reg .u64 %rd<3>;
    cp.async.bulk.tensor.2d.global.shared::cta [%rd0], [%rd1];
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group 0;
}
"""
    k = parse(src, "<test>")
    assert len(k.instrs) == 3
    assert k.instrs[0].op == "cp.async.bulk.tensor.2d.global.shared::cta"
    assert k.instrs[1].op == "cp.async.bulk.commit_group"
    assert k.instrs[2].op == "cp.async.bulk.wait_group"
    from gpusim.frontend.ir import Imm
    assert isinstance(k.instrs[2].src[0], Imm) and k.instrs[2].src[0].value == 0
