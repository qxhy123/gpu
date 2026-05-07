from gpusim.frontend.parser import parse

KT = """
.visible .entry k() {{
    .reg .u32 %r<4>;
    .reg .pred %p<2>;
    {body}
}}
"""

def test_simple_if_then():
    # if (p) goto L1; r1 = r2; L1: r1 = r3;
    body = """
        @%p1 bra L1;
        add.s32 %r1, %r2, %r2;
        L1: add.s32 %r1, %r3, %r3;
    """
    k = parse(KT.format(body=body), "<t>")
    # bra is at pc=0; ipdom of pc=0 should be pc=2 (label L1)
    assert k.ipdom[0] == 2

def test_if_else():
    body = """
        @%p1 bra L1;
        add.s32 %r1, %r2, %r2;
        bra L2;
        L1: add.s32 %r1, %r3, %r3;
        L2: add.s32 %r1, %r1, %r1;
    """
    k = parse(KT.format(body=body), "<t>")
    # both branches reconverge at L2 (pc=4)
    assert k.ipdom[0] == 4
    assert k.ipdom[2] == 4

def test_no_branch_no_ipdom():
    body = "add.s32 %r1, %r2, %r3;"
    k = parse(KT.format(body=body), "<t>")
    assert k.ipdom == {}
