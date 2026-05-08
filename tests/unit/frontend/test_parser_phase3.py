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
