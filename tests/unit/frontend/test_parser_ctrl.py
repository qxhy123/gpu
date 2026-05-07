from gpusim.frontend.parser import parse
from gpusim.frontend.ir import PtxType

KT = """
.visible .entry k() {{
    .reg .u32 %r<4>;
    .reg .pred %p<2>;
    {body}
}}
"""

def test_setp_lt():
    ker = parse(KT.format(body="setp.lt.s32 %p1, %r1, 8;"), "<t>")
    inst = ker.instrs[0]
    assert inst.op == "setp.lt.s32"
    assert inst.type is PtxType.s32

def test_predicated_branch_to_label():
    ker = parse(KT.format(body="L1:\n@%p1 bra L1;\nbra L2;\nL2: bar.sync 0;"), "<t>")
    assert ker.labels["L1"] == 0
    assert ker.labels["L2"] == 2
    assert ker.instrs[0].op == "bra"
    assert ker.instrs[1].op == "bra"
    assert ker.instrs[2].op == "bar.sync"

def test_bar_sync_with_id():
    ker = parse(KT.format(body="bar.sync 0;"), "<t>")
    assert ker.instrs[0].op == "bar.sync"

def test_membar_cta():
    ker = parse(KT.format(body="membar.cta;"), "<t>")
    assert ker.instrs[0].op == "membar.cta"
