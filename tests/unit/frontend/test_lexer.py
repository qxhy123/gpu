from gpusim.frontend.lexer import tokenize, Tok

def t(types_values, src):
    toks = tokenize(src, "<test>")
    got = [(tk.kind, tk.value) for tk in toks if tk.kind != "NL" and tk.kind != "EOF"]
    assert got == types_values

def test_identifier_and_number():
    t([("IDENT","foo"), ("NUM","42")], "foo 42")

def test_register_token():
    t([("REG","r1"), ("REG","p2")], "%r1 %p2")

def test_special_register():
    t([("SREG","tid.x"), ("SREG","ntid.x"), ("SREG","ctaid.x"), ("SREG","nctaid.x")],
      "%tid.x %ntid.x %ctaid.x %nctaid.x")

def test_directive_and_punct():
    t([("DOT","."), ("IDENT","entry"), ("LBRACE","{"), ("RBRACE","}"),
       ("LBRACK","["), ("RBRACK","]"), ("COMMA",","), ("SEMI",";"),
       ("AT","@"), ("BANG","!"), ("COLON",":")],
      ".entry { } [ ] , ; @ ! :")

def test_op_dotted():
    # 'add.s32' should be one IDENT then DOT then IDENT — handled in parser; lexer sees them
    toks = [tk for tk in tokenize("add.s32 r1, r2, r3;", "<t>") if tk.kind not in ("NL","EOF")]
    assert toks[0].kind == "IDENT" and toks[0].value == "add"
    assert toks[1].kind == "DOT"
    assert toks[2].kind == "IDENT" and toks[2].value == "s32"

def test_string_after_comment_skipped():
    toks = [tk for tk in tokenize("// comment\nfoo", "<t>") if tk.kind not in ("NL","EOF")]
    assert toks == [Tok("IDENT","foo",2,1,"<t>")]

def test_block_comment_skipped():
    toks = [tk for tk in tokenize("/* skip\nme */ bar", "<t>") if tk.kind not in ("NL","EOF")]
    assert toks[0].value == "bar"

def test_negative_number():
    toks = [tk for tk in tokenize("-5", "<t>") if tk.kind not in ("NL","EOF")]
    assert toks[0].kind == "NUM" and toks[0].value == "-5"

def test_hex_number():
    toks = [tk for tk in tokenize("0xFF", "<t>") if tk.kind not in ("NL","EOF")]
    assert toks[0].kind == "NUM" and toks[0].value == "0xFF"

def test_lexer_coloncolon():
    from gpusim.frontend.lexer import tokenize
    toks = [t for t in tokenize("shared::cluster", "<test>") if t.kind != "EOF"]
    kinds = [t.kind for t in toks]
    assert kinds == ["IDENT", "COLONCOLON", "IDENT"]
    assert toks[1].value == "::"

def test_lexer_single_colon_still_works():
    from gpusim.frontend.lexer import tokenize
    toks = [t for t in tokenize("L1:", "<test>") if t.kind != "EOF"]
    kinds = [t.kind for t in toks]
    assert kinds == ["IDENT", "COLON"]
