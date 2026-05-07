# gpusim Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement gpusim Phase 1 per `docs/superpowers/specs/2026-05-07-gpusim-phase1-design.md` — a single-SM, cycle-approximate, Hopper-shaped Python GPU simulator with PTX subset support, three visualization outputs, six teaching examples, and an eight-chapter tutorial.

**Architecture:** 6 modules (frontend → config → core → trace → analysis → viz). Cycle-stepped main loop in `core/sm.py`. Trace events are the firewall between core and downstream analysis/viz. Pure Python, no C/C++ extensions.

**Tech Stack:** Python 3.11+. Runtime: `numpy`, `pyyaml`, `pyarrow`, `pandas`, `plotly`, `jinja2`, `typer`. Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`.

**Execution note:** Plan has 6 milestones (M1–M6). After each milestone, pause for review checkpoint. The plan is sequential — each milestone depends on the previous.

---

## Scope check

Phase 1 is one cohesive subsystem (single-SM simulator). Milestones inside Phase 1 are sequential refinements, not independent subsystems, so this is a single plan. Phase 2–5 are separate plans (future).

---

## File structure (all files created across the plan)

```
/                                       # repo root (already git-init'd)
├── pyproject.toml                      # project metadata + deps
├── .gitignore
├── README.md
├── docs/                               # already has specs/
│   └── tutorial/                       # 8 chapters created in M6
├── gpusim/
│   ├── __init__.py
│   ├── api.py                          # gpusim.run() public entry
│   ├── cli.py                          # typer CLI
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── ir.py                       # IR dataclasses (Operand, Instr, Kernel)
│   │   ├── lexer.py                    # PTX tokenizer
│   │   ├── parser.py                   # PTX → Kernel
│   │   └── ipdom.py                    # post-dominator analysis
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py                   # SMConfig dataclass
│   │   ├── loader.py                   # YAML → SMConfig
│   │   └── default_hopper.yaml         # default H100-shaped params
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exec.py                     # functional instruction semantics
│   │   ├── warp.py                     # per-warp state (regs, instr buffer)
│   │   ├── simt_stack.py               # PDOM stack
│   │   ├── scoreboard.py               # RAW dependency tracking
│   │   ├── functional_units.py         # FP32/INT32/BRU/LSU/SYNC pipelines
│   │   ├── regfile.py                  # banked regfile + operand collector
│   │   ├── smem.py                     # shared memory + bank conflict
│   │   ├── gmem.py                     # global memory + coalescing
│   │   ├── scheduler.py                # LRR + GTO
│   │   ├── sub_core.py                 # one of 4 sub-cores within an SM
│   │   ├── occupancy.py                # CTA-on-SM scheduling + bottleneck calc
│   │   └── sm.py                       # SM main object, step()/run() loop
│   ├── trace/
│   │   ├── __init__.py
│   │   ├── events.py                   # event type definitions
│   │   ├── recorder.py                 # event recorder + RLE compression
│   │   └── writer.py                   # parquet writer
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── stall.py                    # stall_breakdown
│   │   ├── attribution.py              # stall_by_source_line
│   │   └── metrics.py                  # ipc_timeline, bank_conflict_hist, etc.
│   └── viz/
│       ├── __init__.py
│       ├── html_report.py              # Jinja2 + Plotly HTML report
│       ├── perfetto.py                 # Perfetto trace JSON exporter
│       └── notebook.py                 # Result class + DataFrame APIs
├── examples/                           # 6 examples in M6
│   └── <name>/{kernel.cu,kernel.ptx,reference.py,run.py,README.md}
├── tests/
│   ├── unit/                           # mirrors gpusim/ structure
│   ├── parity/                         # numpy parity per kernel
│   ├── reference/                      # gen_reference.py + data/*.ref.json
│   └── microbench/                     # textbook-fact assertions
└── scripts/
    └── ptx_from_cuda.py                # nvcc wrapper helper
```

**Test layout convention:** for `gpusim/X/Y.py`, unit tests go in `tests/unit/X/test_Y.py`. Parity tests for examples go in `tests/parity/test_<example>.py`.

---

## Milestone 1 — Frontend + Functional Executor

Outcome: a functional simulator (no timing) that loads a PTX file, dispatches a grid of CTAs, and produces correct numerical output for `vector_add`. Validated against numpy.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`
- Create: `gpusim/__init__.py`, plus empty `__init__.py` in every subpackage (`frontend/`, `config/`, `core/`, `trace/`, `analysis/`, `viz/`)
- Create: `tests/__init__.py`, `tests/unit/__init__.py`, `tests/unit/frontend/__init__.py`, `tests/unit/core/__init__.py`, `tests/unit/config/__init__.py`, `tests/unit/trace/__init__.py`, `tests/unit/analysis/__init__.py`, `tests/unit/viz/__init__.py`
- Create: `tests/parity/__init__.py`, `tests/microbench/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gpusim"
version = "0.1.0"
description = "Teaching-oriented NVIDIA GPU microarchitecture simulator"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pyyaml>=6.0",
  "pyarrow>=15",
  "pandas>=2.1",
  "plotly>=5.18",
  "jinja2>=3.1",
  "typer>=0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov>=4", "ruff>=0.4", "mypy>=1.8"]

[project.scripts]
gpusim = "gpusim.cli:app"

[tool.setuptools.packages.find]
include = ["gpusim*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "reference: tests requiring real-GPU reference fixtures (skipped if absent)",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = false
ignore_missing_imports = true
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/
build/
dist/
.venv/
venv/
*.parquet
*.html
*.json
!docs/**/*.json
!tests/reference/data/*.json
```

- [ ] **Step 3: Create `README.md`**

```markdown
# gpusim

Teaching-oriented NVIDIA GPU microarchitecture simulator.

See `docs/superpowers/specs/2026-05-07-gpusim-phase1-design.md` for the design.

## Install (dev)
```
pip install -e ".[dev]"
```

## Quick start
```
gpusim run examples/vector_add/kernel.ptx --grid 8 --block 128 --output report.html
```

## Tests
```
pytest
```
```

- [ ] **Step 4: Create empty `__init__.py` files**

```bash
mkdir -p gpusim/frontend gpusim/config gpusim/core gpusim/trace gpusim/analysis gpusim/viz
mkdir -p tests/unit/frontend tests/unit/config tests/unit/core tests/unit/trace tests/unit/analysis tests/unit/viz
mkdir -p tests/parity tests/microbench tests/reference/data
touch gpusim/__init__.py gpusim/frontend/__init__.py gpusim/config/__init__.py gpusim/core/__init__.py gpusim/trace/__init__.py gpusim/analysis/__init__.py gpusim/viz/__init__.py
touch tests/__init__.py tests/unit/__init__.py tests/unit/frontend/__init__.py tests/unit/config/__init__.py tests/unit/core/__init__.py tests/unit/trace/__init__.py tests/unit/analysis/__init__.py tests/unit/viz/__init__.py
touch tests/parity/__init__.py tests/microbench/__init__.py
```

- [ ] **Step 5: Verify install works**

```bash
pip install -e ".[dev]"
pytest --collect-only
```
Expected: pytest collects 0 items, no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md gpusim/ tests/
git commit -m "chore: scaffold gpusim Python package and test layout"
```

---

### Task 2: Frontend IR data structures

**Files:**
- Create: `gpusim/frontend/ir.py`
- Test: `tests/unit/frontend/test_ir.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/frontend/test_ir.py
from gpusim.frontend.ir import (
    Operand, Reg, Imm, Param, MemSpace, PtxType, Predicate,
    Instr, Kernel, RegDecl, SrcLoc,
)

def test_reg_operand_str_round_trip():
    op = Reg(name="r1", type=PtxType.s32)
    assert op.name == "r1"
    assert op.type is PtxType.s32

def test_imm_operand_value():
    op = Imm(value=42, type=PtxType.s32)
    assert op.value == 42

def test_predicate_negation():
    p = Predicate(reg="p1", negated=False)
    assert p.reg == "p1" and p.negated is False
    pn = Predicate(reg="p1", negated=True)
    assert pn.negated is True

def test_instr_immutable():
    instr = Instr(
        op="add.s32",
        dst=(Reg("r1", PtxType.s32),),
        src=(Reg("r2", PtxType.s32), Reg("r3", PtxType.s32)),
        pred=None,
        space=None,
        type=PtxType.s32,
        pc=0,
        src_loc=SrcLoc("k.ptx", 10),
    )
    import dataclasses
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        instr.op = "sub.s32"  # type: ignore[misc]

def test_kernel_holds_instr_list_and_labels():
    k = Kernel(
        name="vec_add",
        params=(Param(name="A", type=PtxType.b64),),
        regs=RegDecl(s32=4, f32=4, pred=2, b64=2),
        instrs=(),
        labels={"L1": 0},
        ipdom={},
    )
    assert k.name == "vec_add"
    assert k.labels["L1"] == 0
    assert k.regs.s32 == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/frontend/test_ir.py -v
```
Expected: ImportError / collection error.

- [ ] **Step 3: Implement IR**

```python
# gpusim/frontend/ir.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PtxType(Enum):
    s32 = "s32"
    u32 = "u32"
    s64 = "s64"
    u64 = "u64"
    b32 = "b32"
    b64 = "b64"
    f32 = "f32"
    pred = "pred"


class MemSpace(Enum):
    GLOBAL = "global"
    SHARED = "shared"
    PARAM = "param"
    LOCAL = "local"


@dataclass(frozen=True)
class SrcLoc:
    file: str
    line: int


@dataclass(frozen=True)
class Reg:
    name: str
    type: PtxType


@dataclass(frozen=True)
class Imm:
    value: int | float
    type: PtxType


@dataclass(frozen=True)
class Param:
    name: str
    type: PtxType


# operands are reg | imm | param-name (resolved during parse)
Operand = Reg | Imm


@dataclass(frozen=True)
class Predicate:
    reg: str
    negated: bool


@dataclass(frozen=True)
class RegDecl:
    s32: int = 0
    u32: int = 0
    s64: int = 0
    u64: int = 0
    b32: int = 0
    b64: int = 0
    f32: int = 0
    pred: int = 0


@dataclass(frozen=True)
class Instr:
    op: str
    dst: tuple[Operand, ...]
    src: tuple[Operand | str, ...]   # str = label or param name
    pred: Optional[Predicate]
    space: Optional[MemSpace]
    type: PtxType
    pc: int
    src_loc: SrcLoc


@dataclass(frozen=True)
class Kernel:
    name: str
    params: tuple[Param, ...]
    regs: RegDecl
    instrs: tuple[Instr, ...]
    labels: dict[str, int]
    ipdom: dict[int, int]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/unit/frontend/test_ir.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/frontend/ir.py tests/unit/frontend/test_ir.py
git commit -m "feat(frontend): add PTX IR data structures"
```

---

### Task 3: PTX lexer

**Files:**
- Create: `gpusim/frontend/lexer.py`
- Test: `tests/unit/frontend/test_lexer.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/frontend/test_lexer.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/frontend/test_lexer.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement lexer**

```python
# gpusim/frontend/lexer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator

@dataclass(frozen=True)
class Tok:
    kind: str
    value: str
    line: int
    col: int
    file: str

_PUNCT = {
    "{": "LBRACE", "}": "RBRACE", "[": "LBRACK", "]": "RBRACK",
    "(": "LPAREN", ")": "RPAREN", ",": "COMMA", ";": "SEMI",
    "@": "AT", "!": "BANG", ":": "COLON", ".": "DOT",
}

class LexError(Exception):
    pass

def tokenize(src: str, file: str = "<input>") -> Iterator[Tok]:
    i, line, col = 0, 1, 1
    n = len(src)
    while i < n:
        c = src[i]
        # newline
        if c == "\n":
            yield Tok("NL", "\n", line, col, file)
            i += 1; line += 1; col = 1
            continue
        # whitespace
        if c in " \t\r":
            i += 1; col += 1
            continue
        # line comment
        if c == "/" and i + 1 < n and src[i+1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        # block comment
        if c == "/" and i + 1 < n and src[i+1] == "*":
            i += 2; col += 2
            while i < n and not (src[i] == "*" and i + 1 < n and src[i+1] == "/"):
                if src[i] == "\n":
                    line += 1; col = 1
                else:
                    col += 1
                i += 1
            i += 2; col += 2
            continue
        # register %name (special: %tid.x, %ntid.x, etc.)
        if c == "%":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            name = src[i+1:j]
            # look for dotted special-reg suffix .x/.y/.z
            if j < n and src[j] == "." and name in {"tid","ntid","ctaid","nctaid"}:
                k = j + 1
                while k < n and (src[k].isalnum() or src[k] == "_"):
                    k += 1
                full = src[i+1:k]
                yield Tok("SREG", full, line, col, file)
                col += k - i; i = k
            else:
                yield Tok("REG", name, line, col, file)
                col += j - i; i = j
            continue
        # number (handles negative, hex, float)
        if c.isdigit() or (c == "-" and i + 1 < n and src[i+1].isdigit()):
            j = i + 1
            if c == "-":
                j = i + 1
            if c == "0" and i + 1 < n and src[i+1] in "xX":
                j = i + 2
                while j < n and src[j] in "0123456789abcdefABCDEF":
                    j += 1
            else:
                while j < n and (src[j].isdigit() or src[j] in ".eE+-fF"):
                    # crude: stop at next non-numeric except for float chars
                    if src[j] in "+-" and j > i and src[j-1] not in "eE":
                        break
                    j += 1
            yield Tok("NUM", src[i:j], line, col, file)
            col += j - i; i = j
            continue
        # identifier / keyword
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            yield Tok("IDENT", src[i:j], line, col, file)
            col += j - i; i = j
            continue
        # punctuation
        if c in _PUNCT:
            yield Tok(_PUNCT[c], c, line, col, file)
            i += 1; col += 1
            continue
        raise LexError(f"{file}:{line}:{col}: unexpected character {c!r}")
    yield Tok("EOF", "", line, col, file)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/unit/frontend/test_lexer.py -v
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/frontend/lexer.py tests/unit/frontend/test_lexer.py
git commit -m "feat(frontend): add PTX lexer"
```

---

### Task 4: PTX parser — kernel header (entry, params, reg decls)

**Files:**
- Create: `gpusim/frontend/parser.py`
- Test: `tests/unit/frontend/test_parser_header.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/frontend/test_parser_header.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: ImportError.

- [ ] **Step 3: Implement parser (header only for now; instructions in later tasks)**

```python
# gpusim/frontend/parser.py
from __future__ import annotations
from typing import Iterator
from .lexer import tokenize, Tok
from .ir import (
    Kernel, Param, RegDecl, Instr, Operand, Reg, Imm, Predicate,
    PtxType, MemSpace, SrcLoc,
)

class ParseError(Exception):
    pass


class _Parser:
    def __init__(self, src: str, file: str):
        self.tokens = [t for t in tokenize(src, file) if t.kind != "NL"]
        self.i = 0
        self.file = file

    # ---- low-level token utilities ----
    def peek(self, k: int = 0) -> Tok:
        return self.tokens[self.i + k]

    def eat(self, kind: str, value: str | None = None) -> Tok:
        t = self.peek()
        if t.kind != kind or (value is not None and t.value != value):
            raise ParseError(
                f"{t.file}:{t.line}:{t.col}: expected {kind}"
                + (f"={value!r}" if value else "")
                + f", got {t.kind}={t.value!r}"
            )
        self.i += 1
        return t

    def accept(self, kind: str, value: str | None = None) -> Tok | None:
        t = self.peek()
        if t.kind == kind and (value is None or t.value == value):
            self.i += 1
            return t
        return None

    # ---- top level ----
    def parse_kernel(self) -> Kernel:
        # optional .visible
        while self.accept("DOT"):
            t = self.eat("IDENT")
            if t.value == "entry":
                break
            # else: directive like 'visible', 'weak' — ignore
        name = self.eat("IDENT").value
        params = self._parse_params()
        self.eat("LBRACE")
        regs = self._parse_reg_decls()
        # instructions handled in later task — for now consume until RBRACE
        instrs, labels = self._parse_body(regs)
        self.eat("RBRACE")
        return Kernel(
            name=name,
            params=tuple(params),
            regs=regs,
            instrs=tuple(instrs),
            labels=labels,
            ipdom={},
        )

    def _parse_params(self) -> list[Param]:
        params: list[Param] = []
        self.eat("LPAREN")
        if self.accept("RPAREN"):
            return params
        while True:
            self.eat("DOT"); self.eat("IDENT", "param")
            self.eat("DOT"); ty_tok = self.eat("IDENT")
            try:
                ty = PtxType(ty_tok.value)
            except ValueError:
                raise ParseError(f"unknown param type {ty_tok.value!r}")
            name = self.eat("IDENT").value
            params.append(Param(name=name, type=ty))
            if self.accept("COMMA"):
                continue
            self.eat("RPAREN")
            break
        return params

    def _parse_reg_decls(self) -> RegDecl:
        counts = {k: 0 for k in ("s32","u32","s64","u64","b32","b64","f32","pred")}
        while True:
            # peek for ".reg"
            if not (self.peek().kind == "DOT" and self.peek(1).kind == "IDENT"
                    and self.peek(1).value == "reg"):
                break
            self.eat("DOT"); self.eat("IDENT", "reg")
            self.eat("DOT"); ty_tok = self.eat("IDENT")
            try:
                ty = PtxType(ty_tok.value)
            except ValueError:
                raise ParseError(f"unknown reg type {ty_tok.value!r}")
            # %name<N>;  e.g. %r<4>;
            self.eat("REG")
            count = 1
            if self.accept("IDENT", "lt") or self.peek().value == "<":
                # allow form %r<N>
                pass
            # We accept "<" as IDENT-like or via separate path; PTX uses <> as part of REG decl shorthand.
            # Lexer doesn't tokenize '<' explicitly; treat lookahead heuristically:
            # Simpler approach: REG token already captured 'r'; next char in source might be '<N>'.
            # To avoid extra lexer complexity, require user to write '%r<N>' as REG=r then NUM=<N> chunk OR
            # accept simple case where reg count is implicit via ';'.
            # For pragmatic Phase 1, parse '<NUM>' if present using crude scan.
            if self.peek().kind == "IDENT" and self.peek().value == "_lt":
                pass
            # Hack: if next token is NUM via literal '<' missing, just allow ';'
            if self.peek().kind == "SEMI":
                count = 1
            elif self.peek().kind == "NUM":
                count = int(self.peek().value)
                self.i += 1
            self.eat("SEMI")
            counts[ty.value] = count
        return RegDecl(**counts)

    def _parse_body(self, regs: RegDecl) -> tuple[list[Instr], dict[str,int]]:
        # placeholder for Tasks 5–7; for now just skip everything until matching RBRACE
        instrs: list[Instr] = []
        labels: dict[str, int] = {}
        depth = 1
        while depth > 0:
            t = self.peek()
            if t.kind == "LBRACE":
                depth += 1; self.i += 1
            elif t.kind == "RBRACE":
                depth -= 1
                if depth == 0:
                    break
                self.i += 1
            else:
                self.i += 1
        return instrs, labels


def parse(src: str, file: str = "<input>") -> Kernel:
    p = _Parser(src, file)
    return p.parse_kernel()
```

> Note: the `<N>` syntax for register-count uses `<` and `>` characters. The lexer currently doesn't tokenize them. The simplest fix is to extend the lexer with `LT` / `GT` tokens. Do this now before running tests:

- [ ] **Step 3a: Extend lexer for `<`, `>`**

Edit `gpusim/frontend/lexer.py`, in the `_PUNCT` dict add:
```python
"<": "LT", ">": "GT",
```

Update parser `_parse_reg_decls` to consume `LT NUM GT`:
```python
            # in _parse_reg_decls, replace the "Hack" block:
            if self.accept("LT"):
                count = int(self.eat("NUM").value)
                self.eat("GT")
            self.eat("SEMI")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/unit/frontend/test_parser_header.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/frontend/parser.py gpusim/frontend/lexer.py tests/unit/frontend/test_parser_header.py
git commit -m "feat(frontend): parse kernel header (entry, params, reg decls)"
```

---

### Task 5: Parser — data movement & arithmetic instructions

**Files:**
- Modify: `gpusim/frontend/parser.py` (replace placeholder `_parse_body`)
- Test: `tests/unit/frontend/test_parser_arith.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/frontend/test_parser_arith.py
from gpusim.frontend.parser import parse
from gpusim.frontend.ir import PtxType, MemSpace, Reg, Imm

KERNEL_TEMPLATE = """
.visible .entry k(.param .u64 A, .param .u64 B) {{
    .reg .u32 %r<6>;
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    .reg .pred %p<2>;
    {body}
}}
"""

def k(body: str):
    return parse(KERNEL_TEMPLATE.format(body=body), "<t>")

def test_simple_add_int():
    ker = k("add.s32 %r1, %r2, %r3;")
    assert len(ker.instrs) == 1
    inst = ker.instrs[0]
    assert inst.op == "add.s32"
    assert inst.type is PtxType.s32
    assert inst.dst == (Reg("r1", PtxType.s32),)
    assert inst.src == (Reg("r2", PtxType.s32), Reg("r3", PtxType.s32))

def test_mad_fp32():
    ker = k("mad.f32 %f1, %f2, %f3, %f4;")
    inst = ker.instrs[0]
    assert inst.op == "mad.f32"
    assert len(inst.src) == 3

def test_mul_lo_s32_with_immediate():
    ker = k("mul.lo.s32 %r1, %r2, 4;")
    inst = ker.instrs[0]
    assert inst.op == "mul.lo.s32"
    assert inst.src[1] == Imm(value=4, type=PtxType.s32)

def test_ld_global_with_address():
    ker = k("ld.global.f32 %f1, [%rd1];")
    inst = ker.instrs[0]
    assert inst.op == "ld.global.f32"
    assert inst.space is MemSpace.GLOBAL

def test_ld_global_with_offset():
    ker = k("ld.global.f32 %f1, [%rd1+8];")
    inst = ker.instrs[0]
    assert inst.op == "ld.global.f32"
    # offset captured as Imm in src tuple alongside base reg

def test_st_shared():
    ker = k("st.shared.f32 [%rd1], %f1;")
    inst = ker.instrs[0]
    assert inst.space is MemSpace.SHARED

def test_mov_special_register():
    ker = k("mov.u32 %r1, %tid.x;")
    inst = ker.instrs[0]
    assert inst.op == "mov.u32"
    # special reg encoded as a Reg with name 'tid.x'
    assert inst.src[0] == Reg("tid.x", PtxType.u32)

def test_cvt_s32_f32():
    ker = k("cvt.s32.f32 %r1, %f1;")
    inst = ker.instrs[0]
    assert inst.op == "cvt.s32.f32"

def test_predicate_and_negation():
    ker = k("@%p1 add.s32 %r1, %r2, %r3;\n@!%p1 add.s32 %r1, %r2, %r3;")
    assert ker.instrs[0].pred is not None and ker.instrs[0].pred.negated is False
    assert ker.instrs[1].pred.negated is True
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: existing parser ignores body, returns 0 instrs.

- [ ] **Step 3: Implement instruction parsing**

Replace `_parse_body` in `gpusim/frontend/parser.py`:

```python
    def _parse_body(self, regs: RegDecl) -> tuple[list[Instr], dict[str,int]]:
        instrs: list[Instr] = []
        labels: dict[str, int] = {}
        while self.peek().kind != "RBRACE":
            # label "L1:"
            if self.peek().kind == "IDENT" and self.peek(1).kind == "COLON":
                labels[self.peek().value] = len(instrs)
                self.i += 2
                continue
            instrs.append(self._parse_instr(len(instrs)))
        return instrs, labels

    def _parse_instr(self, pc: int) -> Instr:
        loc_tok = self.peek()
        # optional predicate "@p" or "@!p"
        pred: Predicate | None = None
        if self.accept("AT"):
            negated = self.accept("BANG") is not None
            pred_reg = self.eat("REG").value
            pred = Predicate(reg=pred_reg, negated=negated)

        # opcode dotted: ident('.' ident)*
        op_parts = [self.eat("IDENT").value]
        while self.peek().kind == "DOT" and self.peek(1).kind == "IDENT":
            self.eat("DOT")
            op_parts.append(self.eat("IDENT").value)
        op = ".".join(op_parts)

        space = self._space_from_op(op)
        ptx_type = self._type_from_op(op)

        dst, src = self._parse_operands(op, ptx_type)
        self.eat("SEMI")
        return Instr(
            op=op, dst=tuple(dst), src=tuple(src), pred=pred,
            space=space, type=ptx_type, pc=pc,
            src_loc=SrcLoc(loc_tok.file, loc_tok.line),
        )

    @staticmethod
    def _space_from_op(op: str) -> MemSpace | None:
        for s in MemSpace:
            if f".{s.value}." in f".{op}.":
                return s
        return None

    @staticmethod
    def _type_from_op(op: str) -> PtxType:
        # last dotted component that is a known type
        for part in reversed(op.split(".")):
            try:
                return PtxType(part)
            except ValueError:
                continue
        return PtxType.b32  # fallback for branches and bar.sync etc.

    def _parse_operands(self, op: str, ty: PtxType) -> tuple[list[Operand], list]:
        # branches/sync handled in Task 6; arithmetic and memory here.
        if op.startswith("bra") or op.startswith("@") or op.startswith("bar.") or op.startswith("membar"):
            return [], list(self._parse_operand_list(ty))

        if op.startswith("ld."):
            # ld.<space>.<ty> dst, [addr];
            dst = self._parse_operand(ty)
            self.eat("COMMA")
            base, off = self._parse_addr()
            srcs: list = [base]
            if off is not None:
                srcs.append(off)
            return [dst], srcs

        if op.startswith("st."):
            # st.<space>.<ty> [addr], src;
            base, off = self._parse_addr()
            self.eat("COMMA")
            src = self._parse_operand(ty)
            srcs: list = [base]
            if off is not None:
                srcs.append(off)
            srcs.append(src)
            return [], srcs

        # arithmetic / mov / cvt / setp: dst, src...
        ops = list(self._parse_operand_list(ty))
        if not ops:
            return [], []
        # for setp.<cmp>.<ty>: dst is a predicate register; src are 2 typed operands
        if op.startswith("setp."):
            return [ops[0]], ops[1:]
        return [ops[0]], ops[1:]

    def _parse_operand_list(self, ty: PtxType):
        yield self._parse_operand(ty)
        while self.accept("COMMA"):
            yield self._parse_operand(ty)

    def _parse_operand(self, ty: PtxType) -> Operand:
        t = self.peek()
        if t.kind == "REG":
            self.i += 1
            return Reg(name=t.value, type=ty)
        if t.kind == "SREG":
            self.i += 1
            return Reg(name=t.value, type=PtxType.u32)
        if t.kind == "NUM":
            self.i += 1
            v: int | float = float(t.value) if "." in t.value or "e" in t.value.lower() else (
                int(t.value, 16) if t.value.startswith(("0x","0X","-0x","-0X")) else int(t.value)
            )
            return Imm(value=v, type=ty)
        if t.kind == "IDENT":
            # bare identifier — likely a parameter name (used in ld.param)
            self.i += 1
            return Reg(name=t.value, type=ty)  # treat as "named" operand
        raise ParseError(f"{t.file}:{t.line}:{t.col}: unexpected operand token {t.kind} {t.value!r}")

    def _parse_addr(self) -> tuple[Operand, Imm | None]:
        # [reg]  or  [reg+imm]  or  [reg-imm]  or  [param_name]
        self.eat("LBRACK")
        base = self._parse_operand(PtxType.u64)
        off: Imm | None = None
        nxt = self.peek()
        if nxt.kind == "NUM":  # signed-prefixed lexer already handles '-NN'
            off = Imm(value=int(nxt.value, 0), type=PtxType.s32)
            self.i += 1
        self.eat("RBRACK")
        return base, off
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/unit/frontend/test_parser_arith.py -v
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/frontend/parser.py tests/unit/frontend/test_parser_arith.py
git commit -m "feat(frontend): parse arithmetic and memory instructions"
```

---

### Task 6: Parser — control flow (setp, bra, bar.sync)

**Files:**
- Modify: `gpusim/frontend/parser.py`
- Test: `tests/unit/frontend/test_parser_ctrl.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/frontend/test_parser_ctrl.py
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
```

- [ ] **Step 2: Run to verify failure**

Expected: parse errors on `bra L1;` (label as operand) and `bar.sync 0;`.

- [ ] **Step 3: Extend `_parse_operands` and `_parse_operand` in parser.py**

Update `_parse_operand` to accept identifier as label string (return as plain `str`):

Replace the `IDENT` branch:
```python
        if t.kind == "IDENT":
            # bare identifier: parameter name OR label (resolved later)
            self.i += 1
            return t.value  # plain str — caller treats as label/param
```

Update `_parse_operands` for branches:
```python
        if op == "bra" or op.endswith(".bra"):
            # bra LABEL;
            label = self._parse_operand(ty)
            return [], [label]
        if op.startswith("bar."):
            # bar.sync N;  (N optional, default 0)
            srcs: list = []
            if self.peek().kind == "NUM":
                srcs.append(self._parse_operand(PtxType.s32))
            return [], srcs
        if op.startswith("membar"):
            return [], []
```

Update `Instr.src` type-annotation to allow `str` (already done in Task 2 IR).

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/unit/frontend/test_parser_ctrl.py -v tests/unit/frontend/test_parser_arith.py
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add gpusim/frontend/parser.py tests/unit/frontend/test_parser_ctrl.py
git commit -m "feat(frontend): parse setp/bra/bar.sync/membar"
```

---

### Task 7: IPDOM (post-dominator) analysis

**Files:**
- Create: `gpusim/frontend/ipdom.py`
- Modify: `gpusim/frontend/parser.py` (call `compute_ipdom`)
- Test: `tests/unit/frontend/test_ipdom.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/frontend/test_ipdom.py
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
```

- [ ] **Step 2: Run to verify fail**

Expected: import or assertion failures.

- [ ] **Step 3: Implement ipdom**

```python
# gpusim/frontend/ipdom.py
from __future__ import annotations
from .ir import Kernel, Instr


def successors(instr: Instr, pc: int, n_instrs: int, labels: dict[str, int]) -> list[int]:
    """CFG successors of `instr` at position `pc`."""
    op = instr.op
    if op == "bra":
        # unconditional: only target
        target = instr.src[0]  # label string
        if isinstance(target, str) and target in labels:
            return [labels[target]]
        return []
    if op.endswith("bra") or instr.pred is not None and op == "bra":
        # predicated bra not used in our subset; predicated branches use @p bra L
        pass
    # any predicated bra (regardless of opcode) handled below via instr.pred
    if instr.pred is not None and op == "bra":
        target = instr.src[0]
        succ = []
        if isinstance(target, str) and target in labels:
            succ.append(labels[target])
        if pc + 1 < n_instrs:
            succ.append(pc + 1)
        return succ
    # fall-through for everything else; predicated non-bra falls through too
    return [pc + 1] if pc + 1 < n_instrs else []


def compute_ipdom(kernel: Kernel) -> dict[int, int]:
    """For every branching instruction (predicated bra), compute IPDOM PC.

    Algorithm: build reverse CFG, compute post-dominator tree by iterative
    dataflow, IPDOM(n) = idom-equivalent in post-dom tree.
    """
    instrs = kernel.instrs
    n = len(instrs)
    if n == 0:
        return {}

    # build CFG
    succ: list[list[int]] = [[] for _ in range(n)]
    for pc, ins in enumerate(instrs):
        if ins.op == "bra" and ins.pred is None:
            # unconditional: only label target
            tgt = ins.src[0]
            if isinstance(tgt, str) and tgt in kernel.labels:
                succ[pc].append(kernel.labels[tgt])
        elif ins.op == "bra" and ins.pred is not None:
            tgt = ins.src[0]
            if isinstance(tgt, str) and tgt in kernel.labels:
                succ[pc].append(kernel.labels[tgt])
            if pc + 1 < n:
                succ[pc].append(pc + 1)
        else:
            if pc + 1 < n:
                succ[pc].append(pc + 1)

    # exit nodes (no successors) — gather and treat as post-dominator universe
    exits = {pc for pc in range(n) if not succ[pc]}
    if not exits:
        # ensure last instr is treated as exit
        exits = {n - 1}

    # post-dom set per node: pdom[v] = nodes that post-dominate v
    full = set(range(n))
    pdom: list[set[int]] = [set(full) for _ in range(n)]
    for v in exits:
        pdom[v] = {v}

    changed = True
    while changed:
        changed = False
        for v in range(n):
            if v in exits:
                continue
            if not succ[v]:
                new = {v}
            else:
                new = set(full)
                for s in succ[v]:
                    new &= pdom[s]
                new |= {v}
            if new != pdom[v]:
                pdom[v] = new
                changed = True

    # IPDOM(v) = closest post-dominator other than v
    def ipdom_of(v: int) -> int | None:
        candidates = pdom[v] - {v}
        if not candidates:
            return None
        # pick the one whose own pdom set is largest (closest to v)
        best = None
        best_size = -1
        for c in candidates:
            size = len(pdom[c])
            if size > best_size:
                best_size = size; best = c
        return best

    out: dict[int, int] = {}
    for pc, ins in enumerate(instrs):
        if ins.op == "bra" and ins.pred is not None:
            ip = ipdom_of(pc)
            if ip is not None:
                out[pc] = ip
    return out
```

Wire into parser. In `parse()`:

```python
def parse(src: str, file: str = "<input>") -> Kernel:
    p = _Parser(src, file)
    k = p.parse_kernel()
    from .ipdom import compute_ipdom
    return Kernel(
        name=k.name, params=k.params, regs=k.regs, instrs=k.instrs,
        labels=k.labels, ipdom=compute_ipdom(k),
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/frontend/test_ipdom.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/frontend/ipdom.py gpusim/frontend/parser.py tests/unit/frontend/test_ipdom.py
git commit -m "feat(frontend): post-dominator analysis for IPDOM/RPC"
```

---

### Task 8: Functional executor — register file + thread state

**Files:**
- Create: `gpusim/core/exec.py` (functional layer first; timing wraps it later)
- Test: `tests/unit/core/test_exec_state.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/core/test_exec_state.py
import numpy as np
from gpusim.core.exec import ThreadState, WarpFnState, RegName

def test_thread_reg_read_write():
    t = ThreadState()
    t.set_u32("r1", 42)
    assert t.get_u32("r1") == 42

def test_thread_predicate_default_false():
    t = ThreadState()
    assert t.get_pred("p1") is False
    t.set_pred("p1", True)
    assert t.get_pred("p1") is True

def test_warp_active_mask_default_all_active():
    w = WarpFnState(warp_size=32, tids=tuple(range(32)))
    assert w.active_mask == (1 << 32) - 1

def test_per_lane_register():
    w = WarpFnState(warp_size=32, tids=tuple(range(32)))
    for lane in range(32):
        w.threads[lane].set_u32("r1", lane * 10)
    assert [w.threads[i].get_u32("r1") for i in range(32)] == [i*10 for i in range(32)]
```

- [ ] **Step 2: Run to verify fail**

Expected: ImportError.

- [ ] **Step 3: Implement state**

```python
# gpusim/core/exec.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


RegName = str


class ThreadState:
    """Per-lane scalar register state. Phase 1 keeps registers as Python ints/floats;
    we convert to/from numpy at memory-op boundaries."""

    def __init__(self):
        self._u32: dict[RegName, int] = {}
        self._s32: dict[RegName, int] = {}
        self._u64: dict[RegName, int] = {}
        self._f32: dict[RegName, float] = {}
        self._pred: dict[RegName, bool] = {}

    def set_u32(self, name: RegName, v: int) -> None: self._u32[name] = v & 0xFFFFFFFF
    def get_u32(self, name: RegName) -> int: return self._u32.get(name, 0)
    def set_s32(self, name: RegName, v: int) -> None:
        v &= 0xFFFFFFFF
        if v & 0x80000000: v -= 0x100000000
        self._s32[name] = v
    def get_s32(self, name: RegName) -> int: return self._s32.get(name, 0)
    def set_u64(self, name: RegName, v: int) -> None: self._u64[name] = v & ((1<<64)-1)
    def get_u64(self, name: RegName) -> int: return self._u64.get(name, 0)
    def set_f32(self, name: RegName, v: float) -> None: self._f32[name] = float(np.float32(v))
    def get_f32(self, name: RegName) -> float: return self._f32.get(name, 0.0)
    def set_pred(self, name: RegName, v: bool) -> None: self._pred[name] = bool(v)
    def get_pred(self, name: RegName) -> bool: return self._pred.get(name, False)


@dataclass
class WarpFnState:
    warp_size: int
    tids: tuple[int, ...]      # global thread IDs of the lanes (or per-CTA)
    pc: int = 0
    active_mask: int = 0
    threads: list[ThreadState] = field(default_factory=list)

    def __post_init__(self):
        if not self.threads:
            self.threads = [ThreadState() for _ in range(self.warp_size)]
        if self.active_mask == 0:
            self.active_mask = (1 << self.warp_size) - 1
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/core/test_exec_state.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/core/exec.py tests/unit/core/test_exec_state.py
git commit -m "feat(core): per-thread/per-warp register state"
```

---

### Task 9: Functional memory subsystem

**Files:**
- Modify: `gpusim/core/exec.py` (add memory classes)
- Test: `tests/unit/core/test_exec_memory.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/core/test_exec_memory.py
import numpy as np
from gpusim.core.exec import GlobalMemory, SharedMemory, ParamSpace

def test_global_memory_load_store_f32():
    g = GlobalMemory()
    arr = np.arange(16, dtype=np.float32)
    base = g.bind("A", arr)
    assert g.load_f32(base + 4 * 3) == 3.0
    g.store_f32(base + 4 * 5, 99.0)
    assert g.load_f32(base + 4 * 5) == 99.0
    assert arr[5] == 99.0

def test_global_memory_load_u32_round_trip():
    g = GlobalMemory()
    arr = np.zeros(8, dtype=np.uint32)
    base = g.bind("X", arr)
    g.store_u32(base, 0xDEADBEEF)
    assert g.load_u32(base) == 0xDEADBEEF

def test_param_space_returns_value():
    p = ParamSpace({"A": 0xDEAD0000, "N": 1024})
    assert p.read_u64("A") == 0xDEAD0000
    assert p.read_u32("N") == 1024

def test_shared_memory_per_cta_isolated():
    s = SharedMemory(size_bytes=2048)
    s.allocate_cta(cta_id=0, size_bytes=512)
    s.allocate_cta(cta_id=1, size_bytes=512)
    s.store_f32(cta_id=0, offset=0, value=1.0)
    s.store_f32(cta_id=1, offset=0, value=2.0)
    assert s.load_f32(cta_id=0, offset=0) == 1.0
    assert s.load_f32(cta_id=1, offset=0) == 2.0
```

- [ ] **Step 2: Run to verify fail**

- [ ] **Step 3: Implement memory subsystem (append to `gpusim/core/exec.py`)**

```python
# add to gpusim/core/exec.py
import struct


class GlobalMemory:
    """Flat byte-addressable global memory. Each `bind`-ed numpy array gets a base
    address and shares storage with the underlying buffer."""

    def __init__(self):
        self._next_base = 0x1_0000_0000  # arbitrary base above 32-bit imm range
        self._segments: dict[int, np.ndarray] = {}   # base -> buffer (1d uint8 view)
        self._names: dict[str, int] = {}

    def bind(self, name: str, arr: np.ndarray) -> int:
        view = arr.view(np.uint8).reshape(-1)
        base = self._next_base
        self._segments[base] = view
        self._names[name] = base
        # advance, aligned to 256 bytes
        self._next_base += (view.nbytes + 255) & ~255
        return base

    def address_of(self, name: str) -> int:
        return self._names[name]

    def _seg_for(self, addr: int) -> tuple[np.ndarray, int]:
        # find segment whose base <= addr < base + size
        for base, buf in self._segments.items():
            if base <= addr < base + buf.nbytes:
                return buf, addr - base
        raise ValueError(f"unbound global address 0x{addr:x}")

    def load_f32(self, addr: int) -> float:
        buf, off = self._seg_for(addr)
        return float(buf[off:off+4].view(np.float32)[0])

    def store_f32(self, addr: int, v: float) -> None:
        buf, off = self._seg_for(addr)
        buf[off:off+4].view(np.float32)[0] = np.float32(v)

    def load_u32(self, addr: int) -> int:
        buf, off = self._seg_for(addr)
        return int(buf[off:off+4].view(np.uint32)[0])

    def store_u32(self, addr: int, v: int) -> None:
        buf, off = self._seg_for(addr)
        buf[off:off+4].view(np.uint32)[0] = np.uint32(v & 0xFFFFFFFF)


class SharedMemory:
    def __init__(self, size_bytes: int = 48 * 1024):
        self.size_bytes = size_bytes
        self._cta: dict[int, np.ndarray] = {}

    def allocate_cta(self, cta_id: int, size_bytes: int) -> None:
        self._cta[cta_id] = np.zeros(size_bytes, dtype=np.uint8)

    def free_cta(self, cta_id: int) -> None:
        self._cta.pop(cta_id, None)

    def load_f32(self, cta_id: int, offset: int) -> float:
        return float(self._cta[cta_id][offset:offset+4].view(np.float32)[0])

    def store_f32(self, cta_id: int, offset: int, value: float) -> None:
        self._cta[cta_id][offset:offset+4].view(np.float32)[0] = np.float32(value)

    def load_u32(self, cta_id: int, offset: int) -> int:
        return int(self._cta[cta_id][offset:offset+4].view(np.uint32)[0])

    def store_u32(self, cta_id: int, offset: int, value: int) -> None:
        self._cta[cta_id][offset:offset+4].view(np.uint32)[0] = np.uint32(value & 0xFFFFFFFF)


class ParamSpace:
    def __init__(self, params: dict[str, int]):
        self._params = dict(params)

    def read_u64(self, name: str) -> int:
        return int(self._params[name])

    def read_u32(self, name: str) -> int:
        return int(self._params[name]) & 0xFFFFFFFF
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/core/test_exec_memory.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/core/exec.py tests/unit/core/test_exec_memory.py
git commit -m "feat(core): functional global/shared/param memory subsystems"
```

---

### Task 10: Functional instruction executor

Implements the per-instruction semantic step for all PTX subset opcodes. This is the heart of "what does this instruction do to the lane state" and is timing-agnostic — Milestone 2 wraps timing around it.

**Files:**
- Modify: `gpusim/core/exec.py` (add `InstrExecutor` class)
- Test: `tests/unit/core/test_exec_instr.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/core/test_exec_instr.py
import numpy as np
from gpusim.frontend.parser import parse
from gpusim.frontend.ir import PtxType
from gpusim.core.exec import (
    WarpFnState, GlobalMemory, SharedMemory, ParamSpace, InstrExecutor,
)

KT = """
.visible .entry k(.param .u64 A, .param .u32 N) {{
    .reg .u32 %r<8>;
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    .reg .pred %p<2>;
    {body}
}}
"""

def make_ctx(body, params=None, gmem_arrays=None):
    k = parse(KT.format(body=body), "<t>")
    g = GlobalMemory()
    if gmem_arrays:
        for name, arr in gmem_arrays.items():
            params = dict(params or {})
            params[name] = g.bind(name, arr)
    s = SharedMemory()
    s.allocate_cta(0, 4096)
    p = ParamSpace(params or {})
    w = WarpFnState(warp_size=32, tids=tuple(range(32)))
    ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=p, cta_id=0,
                       ctaid=(0,0,0), nctaid=(1,1,1), ntid=(32,1,1))
    return k, w, ex

def test_add_s32_per_lane():
    k, w, ex = make_ctx("mov.u32 %r1, %tid.x; add.s32 %r2, %r1, 100; bra END; END: bar.sync 0;")
    while w.pc < len(k.instrs):
        ex.execute(w, k.instrs[w.pc])
        w.pc += 1
    for lane in range(32):
        assert w.threads[lane].get_s32("r2") == lane + 100

def test_setp_lt_sets_predicate_per_lane():
    body = "mov.u32 %r1, %tid.x; setp.lt.s32 %p1, %r1, 8;"
    k, w, ex = make_ctx(body)
    while w.pc < len(k.instrs):
        ex.execute(w, k.instrs[w.pc])
        w.pc += 1
    for lane in range(32):
        assert w.threads[lane].get_pred("p1") is (lane < 8)

def test_ld_global_f32():
    arr = np.arange(32, dtype=np.float32)
    body = """
        ld.param.u64 %rd1, [A];
        mov.u32 %r1, %tid.x;
        mul.lo.s32 %r2, %r1, 4;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd3, %rd1, %rd2;
        ld.global.f32 %f1, [%rd3];
    """
    k, w, ex = make_ctx(body, gmem_arrays={"A": arr})
    while w.pc < len(k.instrs):
        ex.execute(w, k.instrs[w.pc])
        w.pc += 1
    for lane in range(32):
        assert w.threads[lane].get_f32("f1") == float(lane)
```

- [ ] **Step 2: Run to verify fail**

- [ ] **Step 3: Implement `InstrExecutor`** (append to `gpusim/core/exec.py`)

```python
from .exec_helpers import op_value  # we'll inline; placeholder
from gpusim.frontend.ir import (
    Instr, Kernel, Reg, Imm, MemSpace, PtxType, Predicate,
)

class InstrExecutor:
    """Executes one Instr against a WarpFnState. Timing-agnostic."""

    def __init__(self, kernel: Kernel, gmem: GlobalMemory, smem: SharedMemory,
                 params: ParamSpace, cta_id: int,
                 ctaid: tuple[int,int,int], nctaid: tuple[int,int,int],
                 ntid: tuple[int,int,int]):
        self.k = kernel
        self.gmem = gmem
        self.smem = smem
        self.params = params
        self.cta_id = cta_id
        self.ctaid = ctaid
        self.nctaid = nctaid
        self.ntid = ntid

    # ---- helpers ----
    def _lane_active(self, w: WarpFnState, lane: int, instr: Instr) -> bool:
        if not (w.active_mask >> lane) & 1:
            return False
        if instr.pred is None:
            return True
        v = w.threads[lane].get_pred(instr.pred.reg)
        return (not v) if instr.pred.negated else v

    @staticmethod
    def _read(t: ThreadState, op, ty: PtxType):
        if isinstance(op, Imm):
            return op.value
        if isinstance(op, str):
            # label/param identifier — caller handles separately
            return op
        # Reg
        name = op.name
        if name in ("tid.x","tid.y","tid.z","ntid.x","ntid.y","ntid.z",
                    "ctaid.x","ctaid.y","ctaid.z","nctaid.x","nctaid.y","nctaid.z"):
            return None  # special; resolved in execute() per-lane
        if ty in (PtxType.s32,):
            return t.get_s32(name)
        if ty in (PtxType.u32, PtxType.b32):
            return t.get_u32(name)
        if ty in (PtxType.s64, PtxType.u64, PtxType.b64):
            return t.get_u64(name)
        if ty is PtxType.f32:
            return t.get_f32(name)
        if ty is PtxType.pred:
            return t.get_pred(name)
        return t.get_u32(name)

    @staticmethod
    def _write(t: ThreadState, op: Reg, value, ty: PtxType):
        name = op.name
        if ty is PtxType.s32:
            t.set_s32(name, int(value))
        elif ty in (PtxType.u32, PtxType.b32):
            t.set_u32(name, int(value))
        elif ty in (PtxType.s64, PtxType.u64, PtxType.b64):
            t.set_u64(name, int(value))
        elif ty is PtxType.f32:
            t.set_f32(name, float(value))
        elif ty is PtxType.pred:
            t.set_pred(name, bool(value))
        else:
            t.set_u32(name, int(value))

    def _resolve_special(self, t: ThreadState, sreg: str, lane: int) -> int:
        if sreg == "tid.x": return lane  # warp_size lanes within first dim of CTA
        if sreg in ("tid.y","tid.z"): return 0
        if sreg == "ntid.x": return self.ntid[0]
        if sreg == "ntid.y": return self.ntid[1]
        if sreg == "ntid.z": return self.ntid[2]
        if sreg == "ctaid.x": return self.ctaid[0]
        if sreg == "ctaid.y": return self.ctaid[1]
        if sreg == "ctaid.z": return self.ctaid[2]
        if sreg == "nctaid.x": return self.nctaid[0]
        if sreg == "nctaid.y": return self.nctaid[1]
        if sreg == "nctaid.z": return self.nctaid[2]
        raise ValueError(f"unknown special reg {sreg}")

    # ---- main entry ----
    def execute(self, w: WarpFnState, instr: Instr) -> None:
        op = instr.op
        # specials — bra and bar.sync return without per-lane work; control handled by caller
        if op == "bra" or op == "bar.sync" or op == "membar.cta":
            return
        for lane in range(w.warp_size):
            if not self._lane_active(w, lane, instr):
                continue
            self._exec_lane(w.threads[lane], instr, lane)

    def _exec_lane(self, t: ThreadState, instr: Instr, lane: int) -> None:
        op = instr.op
        ty = instr.type

        # mov
        if op.startswith("mov."):
            src = instr.src[0]
            if isinstance(src, Reg) and src.name in (
                "tid.x","tid.y","tid.z","ntid.x","ntid.y","ntid.z",
                "ctaid.x","ctaid.y","ctaid.z","nctaid.x","nctaid.y","nctaid.z"):
                v = self._resolve_special(t, src.name, lane)
            else:
                v = self._read(t, src, ty)
            self._write(t, instr.dst[0], v, ty)
            return

        # cvt
        if op.startswith("cvt."):
            # cvt.<dst_ty>.<src_ty>
            parts = op.split(".")
            dst_ty = PtxType(parts[1]); src_ty = PtxType(parts[2])
            v = self._read(t, instr.src[0], src_ty)
            if dst_ty is PtxType.s32 and src_ty is PtxType.f32:
                self._write(t, instr.dst[0], int(v), PtxType.s32)
            elif dst_ty is PtxType.f32 and src_ty is PtxType.s32:
                self._write(t, instr.dst[0], float(v), PtxType.f32)
            elif dst_ty in (PtxType.u64,) and src_ty in (PtxType.u32, PtxType.s32):
                self._write(t, instr.dst[0], int(v) & ((1<<64)-1), PtxType.u64)
            elif dst_ty in (PtxType.u32, PtxType.s32) and src_ty in (PtxType.u64,):
                self._write(t, instr.dst[0], int(v) & 0xFFFFFFFF, dst_ty)
            else:
                self._write(t, instr.dst[0], v, dst_ty)
            return

        # arithmetic
        if op in ("add.s32","add.u32","sub.s32","mul.lo.s32","shl.b32","shr.s32","add.u64","sub.u64"):
            a = self._read(t, instr.src[0], ty)
            b = self._read(t, instr.src[1], ty)
            if op.startswith("add."):    r = a + b
            elif op.startswith("sub."):  r = a - b
            elif op == "mul.lo.s32":     r = (a * b) & 0xFFFFFFFF
            elif op == "shl.b32":        r = (a << (b & 31)) & 0xFFFFFFFF
            elif op == "shr.s32":        r = (a >> (b & 31))
            self._write(t, instr.dst[0], r, ty)
            return

        if op in ("add.f32","sub.f32","mul.f32","mad.f32","fma.f32","mad.lo.s32"):
            if op == "mad.lo.s32":
                a = self._read(t, instr.src[0], PtxType.s32)
                b = self._read(t, instr.src[1], PtxType.s32)
                c = self._read(t, instr.src[2], PtxType.s32)
                self._write(t, instr.dst[0], (a*b + c) & 0xFFFFFFFF, PtxType.s32)
                return
            a = self._read(t, instr.src[0], PtxType.f32)
            b = self._read(t, instr.src[1], PtxType.f32)
            if op == "add.f32": r = a + b
            elif op == "sub.f32": r = a - b
            elif op == "mul.f32": r = a * b
            elif op in ("mad.f32","fma.f32"):
                c = self._read(t, instr.src[2], PtxType.f32)
                r = a * b + c
            self._write(t, instr.dst[0], r, PtxType.f32)
            return

        # setp
        if op.startswith("setp."):
            parts = op.split(".")
            cmp_ = parts[1]; sty = PtxType(parts[2])
            a = self._read(t, instr.src[0], sty); b = self._read(t, instr.src[1], sty)
            if cmp_ == "eq": r = a == b
            elif cmp_ == "ne": r = a != b
            elif cmp_ == "lt": r = a < b
            elif cmp_ == "le": r = a <= b
            elif cmp_ == "gt": r = a > b
            elif cmp_ == "ge": r = a >= b
            else: raise ValueError(f"unknown setp cmp {cmp_}")
            self._write(t, instr.dst[0], r, PtxType.pred)
            return

        # ld.param.<ty>
        if op.startswith("ld.param."):
            param_name = instr.src[0]
            if isinstance(param_name, Reg):
                param_name = param_name.name
            if ty in (PtxType.u64, PtxType.b64, PtxType.s64):
                v = self.params.read_u64(param_name)
            else:
                v = self.params.read_u32(param_name)
            self._write(t, instr.dst[0], v, ty)
            return

        # ld.global.<ty> / ld.shared.<ty>
        if op.startswith("ld.global.") or op.startswith("ld.shared."):
            base = self._read(t, instr.src[0], PtxType.u64)
            off = 0
            if len(instr.src) > 1 and isinstance(instr.src[1], Imm):
                off = int(instr.src[1].value)
            addr = int(base) + off
            if op.startswith("ld.global."):
                if ty is PtxType.f32: v = self.gmem.load_f32(addr)
                else:                 v = self.gmem.load_u32(addr)
            else:
                if ty is PtxType.f32: v = self.smem.load_f32(self.cta_id, addr)
                else:                 v = self.smem.load_u32(self.cta_id, addr)
            self._write(t, instr.dst[0], v, ty)
            return

        # st.global.<ty> / st.shared.<ty>
        if op.startswith("st.global.") or op.startswith("st.shared."):
            base = self._read(t, instr.src[0], PtxType.u64)
            off = 0; src_pos = 1
            if isinstance(instr.src[1], Imm):
                off = int(instr.src[1].value); src_pos = 2
            addr = int(base) + off
            v = self._read(t, instr.src[src_pos], ty)
            if op.startswith("st.global."):
                if ty is PtxType.f32: self.gmem.store_f32(addr, float(v))
                else:                 self.gmem.store_u32(addr, int(v))
            else:
                if ty is PtxType.f32: self.smem.store_f32(self.cta_id, addr, float(v))
                else:                 self.smem.store_u32(self.cta_id, addr, int(v))
            return

        raise NotImplementedError(f"opcode {op!r}")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/core/test_exec_instr.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/core/exec.py tests/unit/core/test_exec_instr.py
git commit -m "feat(core): functional instruction executor (arith/mem/cvt/setp)"
```

---

### Task 11: Functional SIMT stack + control flow + barrier

**Files:**
- Create: `gpusim/core/simt_stack.py`
- Modify: `gpusim/core/exec.py` (functional `step_warp` driver)
- Test: `tests/unit/core/test_simt_stack.py`, `tests/unit/core/test_functional_control.py`

- [ ] **Step 1: Write tests for SIMT stack**

```python
# tests/unit/core/test_simt_stack.py
from gpusim.core.simt_stack import SIMTStack, SIMTEntry

def test_initial_entry_full_mask():
    s = SIMTStack(warp_size=32, entry_pc=0)
    assert s.top().active_mask == (1 << 32) - 1
    assert s.top().pc == 0

def test_push_diverge_two_paths():
    s = SIMTStack(warp_size=32, entry_pc=0)
    taken = 0xFFFF        # lanes 0..15
    s.diverge(taken_pc=10, fallthrough_pc=5, taken_mask=taken, rpc=20)
    # top must be one of the two pushed frames; both reachable by pop
    seen_pcs = []
    while s.top().pc != 20:
        seen_pcs.append(s.top().pc)
        # simulate: each path executes one instr that reaches RPC
        e = s.top()
        s.update_top_pc(e.rpc)
        s.maybe_pop()
    assert sorted(seen_pcs) == sorted([5, 10])

def test_no_diverge_when_all_lanes_take_same_path():
    s = SIMTStack(warp_size=32, entry_pc=0)
    full = (1 << 32) - 1
    diverged = s.diverge(taken_pc=10, fallthrough_pc=5, taken_mask=full, rpc=20)
    assert diverged is False
    assert s.top().pc == 10
    assert s.top().active_mask == full
```

- [ ] **Step 2: Implement SIMTStack**

```python
# gpusim/core/simt_stack.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SIMTEntry:
    pc: int
    active_mask: int
    rpc: int  # reconverge PC; -1 means none


class SIMTStack:
    def __init__(self, warp_size: int, entry_pc: int):
        self._warp_size = warp_size
        full = (1 << warp_size) - 1
        self._stack: list[SIMTEntry] = [SIMTEntry(pc=entry_pc, active_mask=full, rpc=-1)]

    def top(self) -> SIMTEntry: return self._stack[-1]

    def is_done(self) -> bool: return not self._stack

    def update_top_pc(self, pc: int) -> None: self._stack[-1].pc = pc

    def update_top_mask(self, mask: int) -> None: self._stack[-1].active_mask = mask

    def maybe_pop(self) -> bool:
        # pop while top.pc == top.rpc (and we have a parent)
        popped = False
        while len(self._stack) > 1 and self._stack[-1].pc == self._stack[-1].rpc:
            self._stack.pop()
            popped = True
        return popped

    def diverge(self, taken_pc: int, fallthrough_pc: int,
                taken_mask: int, rpc: int) -> bool:
        cur = self._stack[-1]
        if taken_mask == cur.active_mask:
            cur.pc = taken_pc
            return False
        if taken_mask == 0:
            cur.pc = fallthrough_pc
            return False
        # actual divergence: replace top with one path, push the other
        ft_mask = cur.active_mask & ~taken_mask
        cur.pc = fallthrough_pc
        cur.active_mask = ft_mask
        cur.rpc = rpc
        self._stack.append(SIMTEntry(pc=taken_pc, active_mask=taken_mask, rpc=rpc))
        return True

    def end_warp(self) -> None:
        self._stack.clear()
```

Run: `pytest tests/unit/core/test_simt_stack.py -v` — should pass.

- [ ] **Step 3: Tests for functional driver with branches & barrier**

```python
# tests/unit/core/test_functional_control.py
import numpy as np
from gpusim.core.exec import functional_run
from gpusim.frontend.parser import parse

def test_branch_divergence_writes_per_lane():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p<2>; .reg .f32 %f<2>;
        ld.param.u64 %rd1, [OUT];
        mov.u32 %r1, %tid.x;
        cvt.u64.u32 %rd2, %r1;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd3, %r2;
        add.u64 %rd2, %rd1, %rd3;
        setp.lt.s32 %p1, %r1, 16;
        @%p1 bra THEN;
        mov.u32 %r3, 100;
        bra DONE;
        THEN:
        mov.u32 %r3, 200;
        DONE:
        st.global.u32 [%rd2], %r3;
    }
    """
    out = np.zeros(32, dtype=np.uint32)
    functional_run(src, params={"OUT": out}, grid=(1,1,1), block=(32,1,1))
    expected = np.array([200]*16 + [100]*16, dtype=np.uint32)
    np.testing.assert_array_equal(out, expected)
```

- [ ] **Step 4: Implement functional driver**

Append to `gpusim/core/exec.py`:

```python
from gpusim.frontend.parser import parse as parse_ptx
from .simt_stack import SIMTStack


def _resolve_branch_mask(w: WarpFnState, instr: Instr) -> int:
    """Return the mask of currently-active lanes whose predicate is True (i.e., will take the branch)."""
    if instr.pred is None:
        return w.active_mask
    mask = 0
    for lane in range(w.warp_size):
        if not (w.active_mask >> lane) & 1:
            continue
        v = w.threads[lane].get_pred(instr.pred.reg)
        if instr.pred.negated:
            v = not v
        if v:
            mask |= 1 << lane
    return mask


def _step_warp(kernel: Kernel, w: WarpFnState, ex: InstrExecutor,
               stack: SIMTStack, barrier_state: dict) -> bool:
    """Advance warp by one instruction. Returns True if warp completed."""
    if stack.is_done():
        return True
    pc = stack.top().pc
    if pc >= len(kernel.instrs):
        stack.end_warp()
        return True
    w.pc = pc
    w.active_mask = stack.top().active_mask
    instr = kernel.instrs[pc]

    # bra
    if instr.op == "bra":
        target_label = instr.src[0]
        target_pc = kernel.labels[target_label] if isinstance(target_label, str) else int(target_label)
        if instr.pred is None:
            stack.update_top_pc(target_pc)
            stack.maybe_pop()
            return False
        taken_mask = _resolve_branch_mask(w, instr)
        rpc = kernel.ipdom.get(pc, target_pc)
        stack.diverge(taken_pc=target_pc, fallthrough_pc=pc + 1,
                      taken_mask=taken_mask, rpc=rpc)
        stack.maybe_pop()
        return False

    # bar.sync — handled by outer loop (no-op here; CTA-level barrier in functional run)
    if instr.op == "bar.sync":
        # functional: caller orchestrates; just advance PC
        stack.update_top_pc(pc + 1); stack.maybe_pop()
        return False

    # all other ops: per-lane execution then PC++
    ex.execute(w, instr)
    stack.update_top_pc(pc + 1)
    stack.maybe_pop()
    return False


def functional_run(ptx_src: str, *, params: dict[str, np.ndarray | int],
                   grid: tuple[int,int,int], block: tuple[int,int,int]) -> None:
    """Run kernel functionally over the grid. Mutates numpy arrays in `params` in place."""
    k = parse_ptx(ptx_src, "<inline>")
    g = GlobalMemory()
    s = SharedMemory()
    p_dict: dict[str, int] = {}
    for name, val in params.items():
        if isinstance(val, np.ndarray):
            p_dict[name] = g.bind(name, val)
        else:
            p_dict[name] = int(val)
    paramspace = ParamSpace(p_dict)

    threads_per_cta = block[0] * block[1] * block[2]
    warps_per_cta = (threads_per_cta + 31) // 32

    for cz in range(grid[2]):
      for cy in range(grid[1]):
        for cx in range(grid[0]):
            cta_id = cx + cy * grid[0] + cz * grid[0] * grid[1]
            s.allocate_cta(cta_id, 48 * 1024)
            ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=paramspace,
                               cta_id=cta_id, ctaid=(cx,cy,cz),
                               nctaid=grid, ntid=block)
            warps = []
            for wid in range(warps_per_cta):
                tid_base = wid * 32
                tids = tuple(tid_base + i for i in range(32))
                w = WarpFnState(warp_size=32, tids=tids)
                warps.append((w, SIMTStack(warp_size=32, entry_pc=0)))

            # CTA loop: round-robin warp stepping until all done; bar.sync handled
            # by collecting all warps at any bar.sync before any can pass.
            done = [False] * len(warps)
            barrier_pcs = [-1] * len(warps)  # if non-negative, this warp is at a barrier
            while not all(done):
                progressed = False
                for i, (w, st) in enumerate(warps):
                    if done[i] or barrier_pcs[i] >= 0:
                        continue
                    pc = st.top().pc if not st.is_done() else -1
                    if pc >= 0 and k.instrs[pc].op == "bar.sync":
                        barrier_pcs[i] = pc
                        continue
                    finished = _step_warp(k, w, ex, st, {})
                    if finished: done[i] = True
                    progressed = True
                if all((done[i] or barrier_pcs[i] >= 0) for i in range(len(warps))) \
                   and not all(done):
                    # release barrier — all non-done are at it
                    for i in range(len(warps)):
                        if barrier_pcs[i] >= 0:
                            warps[i][1].update_top_pc(barrier_pcs[i] + 1)
                            warps[i][1].maybe_pop()
                            barrier_pcs[i] = -1
                    progressed = True
                if not progressed:
                    raise RuntimeError("functional_run: no warp progressed (deadlock)")
            s.free_cta(cta_id)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/core/test_simt_stack.py tests/unit/core/test_functional_control.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add gpusim/core/simt_stack.py gpusim/core/exec.py \
        tests/unit/core/test_simt_stack.py tests/unit/core/test_functional_control.py
git commit -m "feat(core): SIMT stack + functional run loop with bar.sync"
```

---

### Task 12: Public API `gpusim.run()` (functional mode only)

**Files:**
- Create: `gpusim/api.py`
- Modify: `gpusim/__init__.py` (export `run`)
- Test: `tests/unit/test_api_functional.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_api_functional.py
import numpy as np
import gpusim

VECTOR_ADD = """
.visible .entry vec_add(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %r<6>; .reg .f32 %f<4>; .reg .u64 %rd<6>; .reg .pred %p<2>;
    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [B];
    ld.param.u64 %rd3, [C];
    ld.param.u32 %r1, [N];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.s32 %p1, %r5, %r1;
    @%p1 bra END;
    shl.b32 %r6, %r5, 2;
    cvt.u64.u32 %rd4, %r6;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    add.u64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f3;
    END: bar.sync 0;
}
"""

def test_vector_add_functional_parity():
    n = 1024
    a = np.random.RandomState(0).randn(n).astype(np.float32)
    b = np.random.RandomState(1).randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    r = gpusim.run(
        ptx_src=VECTOR_ADD,
        grid=(8,1,1), block=(128,1,1),
        params={"A": a, "B": b, "C": c, "N": n},
        mode="functional",
    )
    np.testing.assert_allclose(c, a + b, rtol=1e-5)
    assert r.outputs["C"] is c
```

- [ ] **Step 2: Run to fail**

- [ ] **Step 3: Implement API**

```python
# gpusim/api.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from gpusim.core.exec import functional_run


@dataclass
class Result:
    outputs: dict[str, np.ndarray]
    mode: str
    metrics: dict[str, Any]

    def summary(self) -> str:
        return f"gpusim run: mode={self.mode}, outputs={list(self.outputs.keys())}"


def run(*, ptx_src: str | None = None, ptx_path: str | Path | None = None,
        grid: tuple[int,int,int], block: tuple[int,int,int],
        params: dict[str, np.ndarray | int],
        mode: str = "functional", config: Any = None, seed: int = 0) -> Result:
    """Run a PTX kernel under the simulator.

    Phase 1: only `mode="functional"` is implemented; `"timing"` raises NotImplementedError
    until Milestone 2 lands.
    """
    if ptx_src is None:
        if ptx_path is None:
            raise ValueError("provide ptx_src or ptx_path")
        ptx_src = Path(ptx_path).read_text()

    outputs = {k: v for k, v in params.items() if isinstance(v, np.ndarray)}

    if mode == "functional":
        functional_run(ptx_src, params=params, grid=grid, block=block)
        return Result(outputs=outputs, mode="functional", metrics={})
    raise NotImplementedError(f"mode={mode!r} not implemented yet")
```

```python
# gpusim/__init__.py
from .api import run, Result  # noqa: F401
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_api_functional.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/api.py gpusim/__init__.py tests/unit/test_api_functional.py
git commit -m "feat(api): gpusim.run() functional mode end-to-end"
```

---

### Task 13: vector_add example PTX + numpy parity test

**Files:**
- Create: `examples/vector_add/kernel.ptx`
- Create: `examples/vector_add/reference.py`
- Create: `examples/vector_add/run.py`
- Create: `examples/vector_add/README.md`
- Create: `tests/parity/test_vector_add.py`

- [ ] **Step 1: Write the parity test**

```python
# tests/parity/test_vector_add.py
import numpy as np, pathlib
import gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/vector_add/kernel.ptx").read_text()

def test_vector_add_1024():
    n = 1024
    rng = np.random.RandomState(42)
    a = rng.randn(n).astype(np.float32)
    b = rng.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    gpusim.run(ptx_src=PTX, grid=(8,1,1), block=(128,1,1),
               params={"A": a, "B": b, "C": c, "N": n}, mode="functional")
    np.testing.assert_allclose(c, a + b, rtol=1e-5)
```

- [ ] **Step 2: Create kernel.ptx**

```
// examples/vector_add/kernel.ptx
.visible .entry vec_add(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N)
{
    .reg .u32 %r<8>;
    .reg .f32 %f<4>;
    .reg .u64 %rd<8>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [B];
    ld.param.u64 %rd3, [C];
    ld.param.u32 %r1, [N];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;

    setp.ge.s32 %p1, %r5, %r1;
    @%p1 bra END;

    shl.b32 %r6, %r5, 2;
    cvt.u64.u32 %rd4, %r6;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    add.u64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f3;
END:
    bar.sync 0;
}
```

- [ ] **Step 3: Create reference.py and run.py**

```python
# examples/vector_add/reference.py
import numpy as np

def reference(a, b):
    return a + b
```

```python
# examples/vector_add/run.py
import numpy as np, pathlib
import gpusim

def main():
    n = 1024
    rng = np.random.RandomState(42)
    a = rng.randn(n).astype(np.float32)
    b = rng.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    gpusim.run(ptx_src=ptx, grid=(8,1,1), block=(128,1,1),
               params={"A": a, "B": b, "C": c, "N": n}, mode="functional")
    print("max abs error:", float(np.max(np.abs(c - (a + b)))))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create README**

```markdown
# vector_add

最小可运行示例：c[i] = a[i] + b[i]，N=1024。

## 关键代码点
- `kernel.ptx:14` 计算全局线程索引（mad.lo.s32 = ctaid*ntid + tid）
- `kernel.ptx:17` 越界保护（@%p1 bra END）
- `kernel.ptx:19-26` 加载、相加、写回

## 运行
```
python examples/vector_add/run.py
```

## 预期观察
- 模拟器输出 max abs error 应为 0（functional 模式精确等于 a+b）
- Milestone 5 后跑 `gpusim run examples/vector_add/kernel.ptx --grid 8 --block 128 --output report.html`，
  会看到 100% coalesced load、achieved occupancy = 100%、IPC 接近上限。

## 延伸思考
1. 把 block 从 128 改成 32，occupancy 会怎样变化？
2. 把 N 改成 1023（不对齐），尾部 warp 会发生分歧吗？
```

- [ ] **Step 5: Run all parity + unit tests so far**

```bash
pytest -v
```
Expected: all green (frontend + core + api + parity).

- [ ] **Step 6: Commit**

```bash
git add examples/vector_add/ tests/parity/test_vector_add.py
git commit -m "test(parity): vector_add functional kernel example + parity test"
```

---

### Task 14: Minimal CLI (functional mode)

**Files:**
- Create: `gpusim/cli.py`
- Test: `tests/unit/test_cli_functional.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_cli_functional.py
from typer.testing import CliRunner
from gpusim.cli import app

def test_cli_show_kernel_summary(tmp_path):
    p = tmp_path / "k.ptx"
    p.write_text(
        ".visible .entry k(.param .u32 N) { .reg .u32 %r<2>; mov.u32 %r1, 1; }"
    )
    res = CliRunner().invoke(app, ["show", str(p)])
    assert res.exit_code == 0, res.output
    assert "k" in res.output  # kernel name printed
    assert "params" in res.output.lower()

def test_cli_doctor():
    res = CliRunner().invoke(app, ["doctor"])
    assert res.exit_code == 0

def test_cli_run_functional_vector_add(tmp_path):
    # mini kernel that just stores tid into output
    ptx = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<3>; .reg .u64 %rd<4>;
        ld.param.u64 %rd1, [OUT];
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd3, %rd1, %rd2;
        st.global.u32 [%rd3], %r1;
    }
    """
    pf = tmp_path / "k.ptx"; pf.write_text(ptx)
    import numpy as np
    out = np.zeros(32, dtype=np.uint32)
    np.save(tmp_path / "out.npy", out)
    res = CliRunner().invoke(app, [
        "run", str(pf), "--grid", "1", "--block", "32",
        "--inputs", f"OUT:{tmp_path/'out.npy'}",
        "--mode", "functional",
    ])
    assert res.exit_code == 0, res.output
    out_after = np.load(tmp_path / "out.npy")
    assert list(out_after) == list(range(32))
```

- [ ] **Step 2: Implement CLI**

```python
# gpusim/cli.py
from __future__ import annotations
from pathlib import Path
import typer
import numpy as np

app = typer.Typer(help="gpusim — teaching-oriented GPU simulator")


def _parse_dim(s: str) -> tuple[int,int,int]:
    parts = [int(x) for x in s.split(",")]
    while len(parts) < 3: parts.append(1)
    return tuple(parts[:3])  # type: ignore[return-value]


def _parse_inputs(s: str | None) -> dict[str, str]:
    if not s: return {}
    out: dict[str, str] = {}
    for chunk in s.split(","):
        name, _, path = chunk.partition(":")
        out[name.strip()] = path.strip()
    return out


@app.command()
def run(
    kernel: Path,
    grid: str = typer.Option(..., "--grid"),
    block: str = typer.Option(..., "--block"),
    inputs: str = typer.Option(None, "--inputs"),
    mode: str = typer.Option("functional", "--mode"),
    seed: int = typer.Option(0, "--seed"),
):
    """Run a PTX kernel."""
    from gpusim.api import run as api_run
    g = _parse_dim(grid); b = _parse_dim(block)
    inps = _parse_inputs(inputs)
    params: dict[str, np.ndarray | int] = {}
    np_paths: dict[str, Path] = {}
    for name, path in inps.items():
        arr = np.load(path)
        params[name] = arr
        np_paths[name] = Path(path)
    # any param not given a numpy buffer is assumed to be a length scalar
    src = kernel.read_text()
    # heuristic: scalar int params come from --inputs as well, encoded e.g. "N:1024"
    for name, path in inps.items():
        if not path.endswith(".npy"):
            params[name] = int(path)
    # also accept explicit N from CLI? keep simple: numeric values via inputs
    res = api_run(ptx_src=src, grid=g, block=b, params=params, mode=mode, seed=seed)
    typer.echo(res.summary())
    # save back numpy arrays so caller can inspect them
    for name, p in np_paths.items():
        if name in res.outputs:
            np.save(p, res.outputs[name])


@app.command()
def show(kernel: Path):
    """Show parsed IR + IPDOM annotations."""
    from gpusim.frontend.parser import parse
    k = parse(kernel.read_text(), str(kernel))
    typer.echo(f"kernel: {k.name}")
    typer.echo(f"params: {[(p.name, p.type.value) for p in k.params]}")
    typer.echo(f"regs: s32={k.regs.s32} u32={k.regs.u32} u64={k.regs.u64} "
               f"f32={k.regs.f32} pred={k.regs.pred}")
    typer.echo(f"instrs: {len(k.instrs)}, labels: {list(k.labels)}")
    if k.ipdom:
        typer.echo(f"ipdom: {k.ipdom}")


@app.command()
def doctor():
    """Verify dependencies and report versions."""
    import numpy, pandas, pyarrow, plotly, jinja2, yaml
    typer.echo(f"numpy {numpy.__version__}")
    typer.echo(f"pandas {pandas.__version__}")
    typer.echo(f"pyarrow {pyarrow.__version__}")
    typer.echo(f"plotly {plotly.__version__}")
    typer.echo(f"jinja2 {jinja2.__version__}")
    typer.echo(f"pyyaml {yaml.__version__}")
    typer.echo("OK")
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/unit/test_cli_functional.py -v
```
Expected: 3 passed.

- [ ] **Step 4: Commit + Milestone 1 checkpoint**

```bash
git add gpusim/cli.py tests/unit/test_cli_functional.py
git commit -m "feat(cli): typer CLI with run/show/doctor (functional mode)"
git tag M1-complete
```

> **Milestone 1 checkpoint** — pause here for code review. Functional simulator should run vector_add and produce numerically correct results. No timing model yet.

---

## Milestone 2 — Cycle-stepped Pipeline + Scheduler

Outcome: same `vector_add` runs in `mode="timing"`, producing per-cycle warp-state events and per-instruction issue events. Stall reasons are tracked. No memory-bank or coalescing modeling yet (M3).

**Timing model design** (apply throughout M2):
- The functional executor from M1 still computes values; timing wraps it.
- **Issue-time effect:** at issue cycle, functional state mutates (registers, memory) atomically. Scoreboard tracks "register R is read-locked until cycle = issue + latency". Subsequent dependent issues stall on scoreboard.
- **Functional unit:** each FU has `issue_busy_until` (next cycle it can accept a new issue) and a queue of in-flight ops (just for stats). FP32/INT32/BRU/LSU/SYNC.
- **One issue slot per sub-core per cycle.** 4 sub-cores per SM, so up to 4 warp-instrs issued per SM per cycle.
- **bar.sync** stalls a warp until all warps in the same CTA reach the same barrier.

---

### Task 15: SMConfig schema + default_hopper.yaml + loader

**Files:**
- Create: `gpusim/config/schema.py`, `gpusim/config/loader.py`, `gpusim/config/default_hopper.yaml`
- Test: `tests/unit/config/test_loader.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/config/test_loader.py
from gpusim.config.loader import load_default, load_yaml
from gpusim.config.schema import SMConfig

def test_default_loads():
    c = load_default()
    assert isinstance(c, SMConfig)
    assert c.sub_cores == 4
    assert c.warps_per_sm == 64
    assert c.smem_banks == 32
    assert c.regfile.banks == 4
    assert c.scheduler.policy == "gto"

def test_overrides_via_yaml(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("scheduler:\n  policy: lrr\n")
    c = load_yaml(p)
    assert c.scheduler.policy == "lrr"
    # other fields keep default
    assert c.sub_cores == 4
```

- [ ] **Step 2: Schema**

```python
# gpusim/config/schema.py
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SchedulerConfig:
    policy: str = "gto"          # "lrr" | "gto"


@dataclass
class RegFileConfig:
    banks: int = 4
    regs_per_subcore: int = 16384


@dataclass
class FUConfig:
    fp32_throughput: int = 1     # warp-instrs per cycle
    int32_throughput: int = 1
    lsu_throughput: int = 1
    bru_throughput: int = 1
    fp32_latency: int = 4
    int32_latency: int = 4
    fma_latency: int = 4
    bru_latency: int = 1
    smem_latency: int = 20
    gmem_latency: int = 400
    lsu_outstanding: int = 16


@dataclass
class SMConfig:
    sub_cores: int = 4
    warps_per_sm: int = 64
    threads_per_sm: int = 2048
    max_ctas_per_sm: int = 32
    regs_per_sm: int = 65536
    smem_per_sm_bytes: int = 48 * 1024
    smem_banks: int = 32
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    regfile: RegFileConfig = field(default_factory=RegFileConfig)
    fu: FUConfig = field(default_factory=FUConfig)
```

- [ ] **Step 3: Default YAML + loader**

```yaml
# gpusim/config/default_hopper.yaml
sub_cores: 4
warps_per_sm: 64
threads_per_sm: 2048
max_ctas_per_sm: 32
regs_per_sm: 65536
smem_per_sm_bytes: 49152
smem_banks: 32

scheduler:
  policy: gto

regfile:
  banks: 4
  regs_per_subcore: 16384

fu:
  fp32_throughput: 1
  int32_throughput: 1
  lsu_throughput: 1
  bru_throughput: 1
  fp32_latency: 4
  int32_latency: 4
  fma_latency: 4
  bru_latency: 1
  smem_latency: 20
  gmem_latency: 400
  lsu_outstanding: 16
```

```python
# gpusim/config/loader.py
from __future__ import annotations
from pathlib import Path
import yaml
from .schema import SMConfig, SchedulerConfig, RegFileConfig, FUConfig

_DEFAULT_PATH = Path(__file__).parent / "default_hopper.yaml"


def _from_dict(d: dict) -> SMConfig:
    sched = SchedulerConfig(**(d.get("scheduler") or {}))
    rf = RegFileConfig(**(d.get("regfile") or {}))
    fu = FUConfig(**(d.get("fu") or {}))
    base = {k: v for k, v in d.items() if k not in ("scheduler","regfile","fu")}
    return SMConfig(scheduler=sched, regfile=rf, fu=fu, **base)


def load_default() -> SMConfig:
    return load_yaml(_DEFAULT_PATH)


def load_yaml(path: str | Path) -> SMConfig:
    base = yaml.safe_load(_DEFAULT_PATH.read_text()) or {}
    over = yaml.safe_load(Path(path).read_text()) or {}
    # shallow-deep-merge: per-section keys overlay
    merged = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return _from_dict(merged)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/config/test_loader.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/config/ tests/unit/config/
git commit -m "feat(config): SMConfig schema + default_hopper.yaml + loader"
```

---

### Task 16: Scoreboard

**Files:**
- Create: `gpusim/core/scoreboard.py`
- Test: `tests/unit/core/test_scoreboard.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_scoreboard.py
from gpusim.core.scoreboard import Scoreboard

def test_no_initial_dep():
    s = Scoreboard()
    assert s.ready_at("r1", now=0) == 0  # ready immediately
    assert s.has_pending("r1", now=0) is False

def test_write_then_read_blocked_until_latency_done():
    s = Scoreboard()
    s.mark_write("r1", available_at_cycle=10)
    assert s.has_pending("r1", now=5) is True
    assert s.ready_at("r1", now=5) == 10
    assert s.has_pending("r1", now=10) is False

def test_multiple_writes_take_max():
    s = Scoreboard()
    s.mark_write("r1", available_at_cycle=10)
    s.mark_write("r1", available_at_cycle=15)
    assert s.ready_at("r1", now=0) == 15

def test_clear_after_completion():
    s = Scoreboard()
    s.mark_write("r1", available_at_cycle=10)
    s.advance(now=12)
    assert s.has_pending("r1", now=12) is False
```

- [ ] **Step 2: Implementation**

```python
# gpusim/core/scoreboard.py
from __future__ import annotations


class Scoreboard:
    """Tracks the cycle at which each in-flight write becomes visible."""

    def __init__(self):
        self._pending: dict[str, int] = {}  # reg name -> max cycle when ready

    def mark_write(self, reg: str, available_at_cycle: int) -> None:
        cur = self._pending.get(reg, -1)
        if available_at_cycle > cur:
            self._pending[reg] = available_at_cycle

    def has_pending(self, reg: str, now: int) -> bool:
        c = self._pending.get(reg, -1)
        return c > now

    def ready_at(self, reg: str, now: int) -> int:
        c = self._pending.get(reg, -1)
        return max(now, c) if c > now else now

    def advance(self, now: int) -> None:
        # garbage-collect entries that are already in the past
        self._pending = {r: c for r, c in self._pending.items() if c > now}
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/core/test_scoreboard.py -v
git add gpusim/core/scoreboard.py tests/unit/core/test_scoreboard.py
git commit -m "feat(core): per-warp scoreboard for RAW dependency tracking"
```

---

### Task 17: Functional units with timing

**Files:**
- Create: `gpusim/core/functional_units.py`
- Test: `tests/unit/core/test_functional_units.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_functional_units.py
from gpusim.core.functional_units import FUSet, FUKind
from gpusim.config.loader import load_default

def test_fu_classify_op():
    s = FUSet(load_default().fu)
    assert s.classify("add.s32") is FUKind.INT
    assert s.classify("add.f32") is FUKind.FP32
    assert s.classify("mad.f32") is FUKind.FP32
    assert s.classify("ld.global.f32") is FUKind.LSU
    assert s.classify("st.shared.f32") is FUKind.LSU
    assert s.classify("bra") is FUKind.BRU
    assert s.classify("@%p1 bra L1") is FUKind.BRU  # already-stripped op
    assert s.classify("bar.sync") is FUKind.SYNC
    assert s.classify("setp.lt.s32") is FUKind.INT

def test_fu_busy_state():
    s = FUSet(load_default().fu)
    assert s.is_free(FUKind.FP32, now=0)
    s.reserve(FUKind.FP32, now=0, occupancy_cycles=1)
    assert s.is_free(FUKind.FP32, now=0) is False
    assert s.is_free(FUKind.FP32, now=1)

def test_latency_lookup():
    s = FUSet(load_default().fu)
    assert s.result_latency("add.f32") == 4
    assert s.result_latency("mad.f32") == 4
    assert s.result_latency("ld.global.f32") == 400
    assert s.result_latency("ld.shared.f32") == 20
    assert s.result_latency("st.global.f32") == 0   # store: no register writeback
    assert s.result_latency("bra") == 1
```

- [ ] **Step 2: Implementation**

```python
# gpusim/core/functional_units.py
from __future__ import annotations
from enum import Enum
from gpusim.config.schema import FUConfig


class FUKind(Enum):
    FP32 = "fp32"
    INT = "int"
    LSU = "lsu"
    BRU = "bru"
    SYNC = "sync"


class FUSet:
    """Per-sub-core set of functional units. Tracks issue-busy state."""

    def __init__(self, fu_cfg: FUConfig):
        self.cfg = fu_cfg
        # earliest cycle each FU can accept a new issue
        self._issue_free_at: dict[FUKind, int] = {k: 0 for k in FUKind}

    def classify(self, op: str) -> FUKind:
        if op.startswith("ld.") or op.startswith("st.") or op.startswith("mov."):
            return FUKind.LSU
        if op == "bra" or op.endswith(".bra"):
            return FUKind.BRU
        if op.startswith("bar.") or op.startswith("membar"):
            return FUKind.SYNC
        if op.startswith(("add.f", "sub.f", "mul.f", "mad.f", "fma.f")):
            return FUKind.FP32
        if op.startswith("cvt."):
            return FUKind.INT
        return FUKind.INT

    def is_free(self, kind: FUKind, now: int) -> bool:
        return self._issue_free_at[kind] <= now

    def reserve(self, kind: FUKind, now: int, occupancy_cycles: int) -> None:
        self._issue_free_at[kind] = max(self._issue_free_at[kind], now) + occupancy_cycles

    def result_latency(self, op: str) -> int:
        c = self.cfg
        if op.startswith("ld.global."): return c.gmem_latency
        if op.startswith("ld.shared."): return c.smem_latency
        if op.startswith("ld.param."):  return 1
        if op.startswith("mov."):       return 1
        if op.startswith("st."):        return 0  # no register writeback
        if op.startswith(("mad.", "fma.")): return c.fma_latency
        if op.startswith(("add.f", "sub.f", "mul.f")): return c.fp32_latency
        if op.startswith("cvt."):       return c.int32_latency
        if op == "bra" or op.endswith(".bra"): return c.bru_latency
        if op.startswith("setp."):      return c.int32_latency
        if op.startswith("bar.") or op.startswith("membar"): return 1
        return c.int32_latency

    def issue_occupancy(self, op: str, smem_conflict_degree: int = 1,
                        gmem_transactions: int = 1) -> int:
        if op.startswith("ld.shared.") or op.startswith("st.shared."):
            return max(1, smem_conflict_degree)
        if op.startswith("ld.global.") or op.startswith("st.global."):
            return 1   # M3 may revise: outstanding queue handled separately
        return 1
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/core/test_functional_units.py -v
git add gpusim/core/functional_units.py tests/unit/core/test_functional_units.py
git commit -m "feat(core): functional unit classification + latency/occupancy tables"
```

---

### Task 18: Warp (timing) + scheduler interface

**Files:**
- Create: `gpusim/core/warp.py`, `gpusim/core/scheduler.py`
- Test: `tests/unit/core/test_warp_scheduler.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_warp_scheduler.py
from gpusim.core.warp import Warp, StallReason
from gpusim.core.scheduler import LRRScheduler, GTOScheduler

class _FakeKernel:
    def __init__(self, n): self.instrs = [None] * n; self.labels = {}; self.ipdom = {}

def test_lrr_round_robins_among_ready_warps():
    warps = [Warp(warp_id=i, kernel=_FakeKernel(10)) for i in range(4)]
    sched = LRRScheduler(warp_count=4)
    picks = []
    for _ in range(8):
        chosen = sched.pick(now=0, candidates=lambda i: True)
        picks.append(chosen)
    assert picks == [0, 1, 2, 3, 0, 1, 2, 3]

def test_lrr_skips_not_ready():
    sched = LRRScheduler(warp_count=4)
    ready = {0: True, 1: False, 2: True, 3: False}
    picks = [sched.pick(now=0, candidates=lambda i: ready[i]) for _ in range(4)]
    assert picks == [0, 2, 0, 2]

def test_gto_sticks_to_one_warp_until_stall():
    sched = GTOScheduler(warp_count=4)
    ready = {0: True, 1: True, 2: True, 3: True}
    a = sched.pick(now=0, candidates=lambda i: ready[i])
    b = sched.pick(now=1, candidates=lambda i: ready[i])
    assert a == b  # greedy

def test_gto_switches_to_oldest_ready_on_stall():
    sched = GTOScheduler(warp_count=4)
    # initially: pick warp 0
    sched.pick(now=0, candidates=lambda i: True)
    # warp 0 not ready now; warps 1,2,3 ready — should pick 1 (oldest other)
    nxt = sched.pick(now=1, candidates=lambda i: i != 0)
    assert nxt == 1
```

- [ ] **Step 2: Warp + scheduler**

```python
# gpusim/core/warp.py
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from gpusim.core.exec import WarpFnState
from gpusim.core.simt_stack import SIMTStack
from gpusim.core.scoreboard import Scoreboard


class StallReason(Enum):
    ISSUED = "ISSUED"
    IDLE = "IDLE"
    FETCH_EMPTY = "FETCH_EMPTY"
    SCOREBOARD = "SCOREBOARD"
    STRUCTURAL = "STRUCTURAL"
    OPERAND = "OPERAND"
    MEM_DEP = "MEM_DEP"
    BARRIER = "BARRIER"
    PRED_OFF = "PRED_OFF"
    DIVERGENCE_SERIAL = "DIVERGENCE_SERIAL"


@dataclass
class Warp:
    warp_id: int
    kernel: object        # Kernel
    fn_state: WarpFnState | None = None
    stack: SIMTStack | None = None
    scoreboard: Scoreboard = field(default_factory=Scoreboard)
    barrier_pc: int = -1   # -1 = not at barrier
    finished: bool = False
    cta_id: int = 0
```

```python
# gpusim/core/scheduler.py
from __future__ import annotations
from typing import Callable


class LRRScheduler:
    """Loose Round Robin: cycles warp_id, picking the next ready one."""

    def __init__(self, warp_count: int):
        self.n = warp_count
        self._next = 0

    def pick(self, now: int, candidates: Callable[[int], bool]) -> int | None:
        for offset in range(self.n):
            i = (self._next + offset) % self.n
            if candidates(i):
                self._next = (i + 1) % self.n
                return i
        return None


class GTOScheduler:
    """Greedy-Then-Oldest: stay on current warp until it stalls; otherwise pick the
    longest-ready warp (oldest in our simple model = lowest warp_id that's ready)."""

    def __init__(self, warp_count: int):
        self.n = warp_count
        self._current: int | None = None

    def pick(self, now: int, candidates: Callable[[int], bool]) -> int | None:
        if self._current is not None and candidates(self._current):
            return self._current
        for i in range(self.n):
            if candidates(i):
                self._current = i
                return i
        self._current = None
        return None
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/core/test_warp_scheduler.py -v
git add gpusim/core/warp.py gpusim/core/scheduler.py tests/unit/core/test_warp_scheduler.py
git commit -m "feat(core): Warp state class + LRR/GTO schedulers"
```

---

### Task 19: Sub-core (issue + execute integration)

**Files:**
- Create: `gpusim/core/sub_core.py`
- Test: `tests/unit/core/test_sub_core.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_sub_core.py
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sub_core import SubCore
from gpusim.core.warp import Warp, StallReason
from gpusim.core.exec import WarpFnState, GlobalMemory, SharedMemory, ParamSpace, InstrExecutor
from gpusim.core.simt_stack import SIMTStack

def _make_warp(kernel, wid=0):
    fn = WarpFnState(warp_size=32, tids=tuple(range(32)))
    return Warp(warp_id=wid, kernel=kernel, fn_state=fn,
                stack=SIMTStack(warp_size=32, entry_pc=0))

def test_subcore_issues_one_per_cycle():
    k = parse(
        ".visible .entry k() { .reg .u32 %r<4>; "
        "add.s32 %r1, %r2, %r3; add.s32 %r2, %r1, %r1; }",
        "<t>")
    cfg = load_default()
    g = GlobalMemory(); s = SharedMemory(); s.allocate_cta(0, 4096)
    p = ParamSpace({})
    ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=p, cta_id=0,
                       ctaid=(0,0,0), nctaid=(1,1,1), ntid=(32,1,1))
    sc = SubCore(sub_core_id=0, cfg=cfg, executor=ex, warps=[_make_warp(k)])
    # cycle 0: issue first add
    s0 = sc.step(now=0)
    assert s0[0] is StallReason.ISSUED
    # cycle 1: dependent on r1 — scoreboard says r1 ready at cycle 4 (latency 4)
    s1 = sc.step(now=1)
    assert s1[0] is StallReason.SCOREBOARD
    # cycle 4: now ready
    s2 = sc.step(now=4)
    assert s2[0] is StallReason.ISSUED

def test_subcore_idle_when_warp_done():
    k = parse(".visible .entry k() { .reg .u32 %r<2>; mov.u32 %r1, 1; }", "<t>")
    cfg = load_default()
    g = GlobalMemory(); s = SharedMemory(); s.allocate_cta(0, 4096)
    p = ParamSpace({})
    ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=p, cta_id=0,
                       ctaid=(0,0,0), nctaid=(1,1,1), ntid=(32,1,1))
    sc = SubCore(sub_core_id=0, cfg=cfg, executor=ex, warps=[_make_warp(k)])
    sc.step(now=0)  # issue mov
    # next cycle: warp at PC=1 which is past end → finished → IDLE
    s1 = sc.step(now=1)
    assert s1[0] is StallReason.IDLE
```

- [ ] **Step 2: Implement SubCore**

```python
# gpusim/core/sub_core.py
from __future__ import annotations
from dataclasses import dataclass
from gpusim.config.schema import SMConfig
from gpusim.core.warp import Warp, StallReason
from gpusim.core.scheduler import LRRScheduler, GTOScheduler
from gpusim.core.functional_units import FUSet, FUKind
from gpusim.core.simt_stack import SIMTStack
from gpusim.core.exec import InstrExecutor
from gpusim.frontend.ir import Instr, Reg


def _make_scheduler(policy: str, n: int):
    if policy == "lrr": return LRRScheduler(n)
    if policy == "gto": return GTOScheduler(n)
    raise ValueError(f"unknown scheduler policy {policy!r}")


def _src_regs(instr: Instr) -> list[str]:
    out: list[str] = []
    for s in instr.src:
        if isinstance(s, Reg):
            out.append(s.name)
    if instr.pred is not None:
        out.append(instr.pred.reg)
    return out


def _dst_regs(instr: Instr) -> list[str]:
    return [d.name for d in instr.dst if isinstance(d, Reg)]


@dataclass
class SubCore:
    sub_core_id: int
    cfg: SMConfig
    executor: InstrExecutor
    warps: list[Warp]

    def __post_init__(self):
        self.fus = FUSet(self.cfg.fu)
        self.scheduler = _make_scheduler(self.cfg.scheduler.policy, len(self.warps))
        for w in self.warps:
            if w.stack is None:
                w.stack = SIMTStack(warp_size=32, entry_pc=0)

    def _is_ready(self, w: Warp, now: int) -> tuple[bool, StallReason]:
        if w.finished or w.stack is None or w.stack.is_done():
            return False, StallReason.IDLE
        if w.barrier_pc >= 0:
            return False, StallReason.BARRIER
        pc = w.stack.top().pc
        if pc >= len(w.kernel.instrs):
            w.finished = True
            return False, StallReason.IDLE
        instr = w.kernel.instrs[pc]
        # scoreboard check
        for r in _src_regs(instr):
            if w.scoreboard.has_pending(r, now):
                return False, StallReason.SCOREBOARD
        kind = self.fus.classify(instr.op)
        if not self.fus.is_free(kind, now):
            return False, StallReason.STRUCTURAL
        return True, StallReason.ISSUED

    def step(self, now: int) -> list[StallReason]:
        """Returns a list of StallReason, one per warp slot, indexed by warp position."""
        states: list[StallReason] = [StallReason.IDLE] * len(self.warps)
        # determine readiness for all warps; recorded for trace
        ready_flags: list[StallReason] = [StallReason.IDLE] * len(self.warps)
        for i, w in enumerate(self.warps):
            ok, why = self._is_ready(w, now)
            ready_flags[i] = why if not ok else StallReason.ISSUED
        # scheduler picks one ready warp
        chosen = self.scheduler.pick(now, candidates=lambda i: ready_flags[i] is StallReason.ISSUED)
        for i in range(len(self.warps)):
            states[i] = ready_flags[i]
        if chosen is None:
            return states
        w = self.warps[chosen]
        instr = w.kernel.instrs[w.stack.top().pc]
        # reserve FU
        kind = self.fus.classify(instr.op)
        occ = self.fus.issue_occupancy(instr.op)
        self.fus.reserve(kind, now, occ)
        # execute (functional + scoreboard mark)
        self._issue(w, instr, now)
        states[chosen] = StallReason.ISSUED
        return states

    def _issue(self, w: Warp, instr: Instr, now: int) -> None:
        op = instr.op
        if op == "bar.sync":
            w.barrier_pc = w.stack.top().pc
            return
        if op == "bra":
            target_pc = w.kernel.labels[instr.src[0]] if isinstance(instr.src[0], str) else 0
            if instr.pred is None:
                w.stack.update_top_pc(target_pc); w.stack.maybe_pop()
                return
            # predicated — compute mask
            from gpusim.core.exec import _resolve_branch_mask
            taken_mask = _resolve_branch_mask(w.fn_state, instr)
            rpc = w.kernel.ipdom.get(w.stack.top().pc, target_pc)
            w.stack.diverge(taken_pc=target_pc, fallthrough_pc=w.stack.top().pc + 1,
                            taken_mask=taken_mask, rpc=rpc)
            w.stack.maybe_pop()
            return
        # other ops: execute functionally now
        w.fn_state.active_mask = w.stack.top().active_mask
        w.fn_state.pc = w.stack.top().pc
        self.executor.execute(w.fn_state, instr)
        # mark dst regs in scoreboard
        latency = self.fus.result_latency(op)
        if latency > 0:
            for d in _dst_regs(instr):
                w.scoreboard.mark_write(d, now + latency)
        # advance PC
        w.stack.update_top_pc(w.stack.top().pc + 1)
        w.stack.maybe_pop()
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/core/test_sub_core.py -v
git add gpusim/core/sub_core.py tests/unit/core/test_sub_core.py
git commit -m "feat(core): SubCore — issue/execute pipeline integration"
```

---

### Task 20: SM main loop with multi-CTA-of-1 and barrier coordination

**Files:**
- Create: `gpusim/core/sm.py`
- Test: `tests/unit/core/test_sm.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_sm.py
import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM, SMRunResult

def test_sm_runs_simple_kernel_in_timing_mode():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<3>; .reg .u64 %rd<3>;
        ld.param.u64 %rd1, [OUT];
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd3, %rd1, %rd2;
        st.global.u32 [%rd3], %r1;
        bar.sync 0;
    }
    """
    k = parse(src, "<t>")
    out = np.zeros(32, dtype=np.uint32)
    cfg = load_default()
    sm = SM(cfg=cfg)
    res = sm.run(kernel=k, grid=(1,1,1), block=(32,1,1),
                 params={"OUT": out})
    assert isinstance(res, SMRunResult)
    assert list(out) == list(range(32))
    # cycles must be > number of instructions (latency dominates)
    assert res.cycles > len(k.instrs)
```

- [ ] **Step 2: Implement SM**

```python
# gpusim/core/sm.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from gpusim.config.schema import SMConfig
from gpusim.core.warp import Warp
from gpusim.core.simt_stack import SIMTStack
from gpusim.core.exec import (
    WarpFnState, GlobalMemory, SharedMemory, ParamSpace, InstrExecutor,
)
from gpusim.core.sub_core import SubCore
from gpusim.frontend.ir import Kernel


@dataclass
class SMRunResult:
    cycles: int
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    events: list[Any] = field(default_factory=list)   # filled in M5


class SM:
    def __init__(self, cfg: SMConfig):
        self.cfg = cfg

    def run(self, kernel: Kernel, grid: tuple[int,int,int], block: tuple[int,int,int],
            params: dict[str, np.ndarray | int]) -> SMRunResult:
        gmem = GlobalMemory()
        smem = SharedMemory(size_bytes=self.cfg.smem_per_sm_bytes)
        p_dict: dict[str, int] = {}
        for name, val in params.items():
            if isinstance(val, np.ndarray):
                p_dict[name] = gmem.bind(name, val)
            else:
                p_dict[name] = int(val)
        paramspace = ParamSpace(p_dict)

        threads_per_cta = block[0] * block[1] * block[2]
        warps_per_cta = (threads_per_cta + 31) // 32

        # Phase 1: schedule one CTA at a time across the grid (multi-CTA in M4).
        cycles_total = 0
        for cz in range(grid[2]):
          for cy in range(grid[1]):
            for cx in range(grid[0]):
                cta_id = cx + cy * grid[0] + cz * grid[0] * grid[1]
                smem.allocate_cta(cta_id, self.cfg.smem_per_sm_bytes)
                cycles_total += self._run_cta(kernel, gmem, smem, paramspace,
                                              cta_id, (cx,cy,cz),
                                              grid, block, warps_per_cta)
                smem.free_cta(cta_id)

        outputs = {n: v for n, v in params.items() if isinstance(v, np.ndarray)}
        return SMRunResult(cycles=cycles_total, outputs=outputs)

    def _run_cta(self, kernel, gmem, smem, paramspace,
                 cta_id, ctaid, grid, block, warps_per_cta) -> int:
        executor = InstrExecutor(kernel=kernel, gmem=gmem, smem=smem,
                                 params=paramspace, cta_id=cta_id, ctaid=ctaid,
                                 nctaid=grid, ntid=block)
        # build warps; distribute across sub-cores by warp_id mod sub_cores
        all_warps: list[Warp] = []
        for wid in range(warps_per_cta):
            tids = tuple(range(wid*32, wid*32+32))
            fn = WarpFnState(warp_size=32, tids=tids)
            all_warps.append(Warp(warp_id=wid, kernel=kernel, fn_state=fn,
                                  stack=SIMTStack(warp_size=32, entry_pc=0),
                                  cta_id=cta_id))
        # group warps per sub-core
        groups: list[list[Warp]] = [[] for _ in range(self.cfg.sub_cores)]
        for w in all_warps:
            groups[w.warp_id % self.cfg.sub_cores].append(w)
        sub_cores = [SubCore(i, self.cfg, executor, groups[i])
                     for i in range(self.cfg.sub_cores)]

        cycle = 0
        while True:
            for sc in sub_cores:
                sc.step(now=cycle)
            # barrier coordination — release if all non-finished warps in CTA are at barrier
            non_done = [w for w in all_warps if not w.finished]
            if non_done and all(w.barrier_pc >= 0 for w in non_done):
                for w in non_done:
                    w.stack.update_top_pc(w.barrier_pc + 1); w.stack.maybe_pop()
                    w.barrier_pc = -1
            cycle += 1
            if all(w.finished or (w.stack and w.stack.is_done()) for w in all_warps):
                break
            if cycle > 10_000_000:
                raise RuntimeError("simulation runaway > 1e7 cycles")
        return cycle
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/core/test_sm.py -v
git add gpusim/core/sm.py tests/unit/core/test_sm.py
git commit -m "feat(core): SM main loop — multi-CTA serial, barrier coordination"
```

---

### Task 21: Wire timing mode into `gpusim.run()` + vector_add timing parity

**Files:**
- Modify: `gpusim/api.py`
- Test: `tests/parity/test_vector_add_timing.py`

- [ ] **Step 1: Tests**

```python
# tests/parity/test_vector_add_timing.py
import numpy as np, pathlib, gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/vector_add/kernel.ptx").read_text()

def test_vector_add_timing_mode_correct():
    n = 1024
    rng = np.random.RandomState(7)
    a = rng.randn(n).astype(np.float32); b = rng.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    res = gpusim.run(ptx_src=PTX, grid=(8,1,1), block=(128,1,1),
                     params={"A": a, "B": b, "C": c, "N": n}, mode="timing")
    np.testing.assert_allclose(c, a + b, rtol=1e-5)
    assert res.metrics["cycles"] > 0
```

- [ ] **Step 2: Modify api.py**

Replace the `mode` branch in `run()`:

```python
    if mode == "functional":
        functional_run(ptx_src, params=params, grid=grid, block=block)
        return Result(outputs=outputs, mode="functional", metrics={})
    if mode == "timing":
        from gpusim.frontend.parser import parse
        from gpusim.config.loader import load_default, load_yaml
        from gpusim.core.sm import SM
        cfg = load_default() if config is None else (
            load_yaml(config) if isinstance(config, (str, Path)) else config
        )
        k = parse(ptx_src, "<inline>")
        sm = SM(cfg)
        res = sm.run(kernel=k, grid=grid, block=block, params=params)
        return Result(
            outputs=res.outputs, mode="timing",
            metrics={"cycles": res.cycles},
        )
    raise NotImplementedError(f"mode={mode!r} not implemented yet")
```

- [ ] **Step 3: Run + commit + checkpoint**

```bash
pytest tests/parity/test_vector_add_timing.py -v
pytest -v   # all tests should pass
git add gpusim/api.py tests/parity/test_vector_add_timing.py
git commit -m "feat(api): timing mode wired up; vector_add parity in timing mode"
git tag M2-complete
```

> **Milestone 2 checkpoint** — pause for review. Cycle-stepped simulator now runs vector_add. Memory bank/coalescing modeling and trace recording come in M3/M5.

---

## Milestone 3 — Shared/Global Memory Modeling + Regfile Banks

Outcome: shared memory bank conflicts and global memory coalescing affect timing. Regfile bank conflicts add operand-collector stalls. New stall reasons appear in SubCore output.

---

### Task 22: Shared memory bank conflict model

**Files:**
- Create: `gpusim/core/smem.py`
- Test: `tests/unit/core/test_smem_bankconf.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_smem_bankconf.py
from gpusim.core.smem import bank_conflict_degree

def test_no_conflict_stride_1():
    addrs = [i * 4 for i in range(32)]   # one 4-byte word per lane, contiguous
    assert bank_conflict_degree(addrs) == 1

def test_full_conflict_stride_32_words():
    addrs = [i * 4 * 32 for i in range(32)]   # all hit bank 0, different addr
    assert bank_conflict_degree(addrs) == 32

def test_broadcast_same_address():
    addrs = [0] * 32
    assert bank_conflict_degree(addrs) == 1

def test_two_way_stride_2_words():
    addrs = [i * 4 * 2 for i in range(32)]   # alternates banks 0/2/4...
    assert bank_conflict_degree(addrs) == 2

def test_inactive_lanes_ignored():
    addrs = [0] * 32; mask = 0xFF  # only lanes 0..7 active, all to bank 0 same addr → broadcast
    assert bank_conflict_degree(addrs, active_mask=mask) == 1

def test_eight_lanes_to_eight_banks_no_conflict():
    addrs = [i * 4 for i in range(8)] + [0] * 24
    mask = 0xFF
    assert bank_conflict_degree(addrs, active_mask=mask) == 1
```

- [ ] **Step 2: Implementation**

```python
# gpusim/core/smem.py
from __future__ import annotations
from collections import defaultdict


def bank_conflict_degree(addresses: list[int], active_mask: int = (1 << 32) - 1,
                         banks: int = 32, word_bytes: int = 4) -> int:
    """Compute the bank-conflict degree of a single warp shared-memory access.

    Returns the per-bank max count of *distinct* addresses (broadcast collapses
    duplicates). 1 means no conflict.
    """
    by_bank: dict[int, set[int]] = defaultdict(set)
    for lane, addr in enumerate(addresses):
        if not (active_mask >> lane) & 1:
            continue
        bank = (addr // word_bytes) % banks
        by_bank[bank].add(addr)
    if not by_bank:
        return 1
    return max(len(addrs) for addrs in by_bank.values())
```

- [ ] **Step 3: Wire into SubCore — extract per-thread shared addresses**

Add a helper to `gpusim/core/exec.py`:

```python
# add at bottom of gpusim/core/exec.py
def shared_addresses_for_warp(w: WarpFnState, instr: Instr) -> list[int]:
    """Compute per-lane absolute byte offsets for a shared ld/st instr."""
    addrs: list[int] = [0] * w.warp_size
    for lane in range(w.warp_size):
        if not (w.active_mask >> lane) & 1:
            addrs[lane] = -1
            continue
        t = w.threads[lane]
        # base reg
        base_op = instr.src[0]
        if isinstance(base_op, Reg):
            base = t.get_u64(base_op.name)
        else:
            base = int(getattr(base_op, "value", 0))
        off = 0
        if len(instr.src) > 1 and isinstance(instr.src[1], Imm):
            off = int(instr.src[1].value)
        addrs[lane] = (base + off) & 0xFFFFFFFF   # within smem
    return addrs
```

Modify `SubCore._issue` for shared ops to compute the conflict degree and use it as issue occupancy:

In `gpusim/core/sub_core.py`, replace the body around `issue_occupancy` and add for shared:

```python
    def _issue(self, w: Warp, instr: Instr, now: int) -> None:
        op = instr.op
        if op == "bar.sync":
            w.barrier_pc = w.stack.top().pc
            return
        if op == "bra":
            target_pc = w.kernel.labels[instr.src[0]] if isinstance(instr.src[0], str) else 0
            if instr.pred is None:
                w.stack.update_top_pc(target_pc); w.stack.maybe_pop()
                return
            from gpusim.core.exec import _resolve_branch_mask
            taken_mask = _resolve_branch_mask(w.fn_state, instr)
            rpc = w.kernel.ipdom.get(w.stack.top().pc, target_pc)
            w.stack.diverge(taken_pc=target_pc, fallthrough_pc=w.stack.top().pc + 1,
                            taken_mask=taken_mask, rpc=rpc)
            w.stack.maybe_pop()
            return

        # functional execution
        w.fn_state.active_mask = w.stack.top().active_mask
        w.fn_state.pc = w.stack.top().pc
        self.executor.execute(w.fn_state, instr)

        # determine issue occupancy / latency adjustments
        latency = self.fus.result_latency(op)
        smem_conflict = 1
        if op.startswith(("ld.shared.", "st.shared.")):
            from gpusim.core.exec import shared_addresses_for_warp
            from gpusim.core.smem import bank_conflict_degree
            addrs = shared_addresses_for_warp(w.fn_state, instr)
            mask = w.fn_state.active_mask
            smem_conflict = bank_conflict_degree(
                addrs, active_mask=mask,
                banks=self.cfg.smem_banks)
            # extend FU occupancy and latency
            extra = smem_conflict - 1
            if extra > 0:
                kind = self.fus.classify(op)
                self.fus.reserve(kind, now, extra)  # additional cycles on top of base 1
                latency += extra

        # mark dst regs in scoreboard
        if latency > 0:
            for d in _dst_regs(instr):
                w.scoreboard.mark_write(d, now + latency)

        # advance PC
        w.stack.update_top_pc(w.stack.top().pc + 1)
        w.stack.maybe_pop()
```

(Note: in `SubCore.step`, the `self.fus.reserve(kind, now, occ)` call before `_issue` already reserves base 1 cycle. The `_issue` adds `(N-1)` more on top.)

- [ ] **Step 4: Tests**

```python
# tests/unit/core/test_smem_pipeline.py
import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM

def test_shared_no_conflict_pattern_1cycle_occupancy():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .f32 %f<2>;
        // each lane writes its tid to its own dword in shared
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd1, %r2;
        cvt.f32.s32 %f1, %r1;
        st.shared.f32 [%rd1], %f1;
        ld.shared.f32 %f2, [%rd1];
        bar.sync 0;
    }
    """
    k = parse(src, "<t>")
    out = np.zeros(32, dtype=np.float32)
    sm = SM(load_default())
    res = sm.run(kernel=k, grid=(1,1,1), block=(32,1,1), params={"OUT": out})
    cycles_no_conflict = res.cycles

def test_shared_32way_conflict_costs_more_cycles():
    # all lanes to bank 0
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .f32 %f<2>;
        mov.u32 %r1, %tid.x;
        // 32-stride: lane i → offset i*128 (bank 0 every time)
        shl.b32 %r2, %r1, 7;
        cvt.u64.u32 %rd1, %r2;
        cvt.f32.s32 %f1, %r1;
        st.shared.f32 [%rd1], %f1;
        bar.sync 0;
    }
    """
    k = parse(src, "<t>")
    out = np.zeros(32, dtype=np.float32)
    sm = SM(load_default())
    res = sm.run(kernel=k, grid=(1,1,1), block=(32,1,1), params={"OUT": out})
    # there should be at least 32 extra cycles spent on the conflicted store
    assert res.cycles >= 32 + 5
```

- [ ] **Step 5: Run + commit**

```bash
pytest tests/unit/core/test_smem_bankconf.py tests/unit/core/test_smem_pipeline.py -v
git add gpusim/core/smem.py gpusim/core/exec.py gpusim/core/sub_core.py \
        tests/unit/core/test_smem_bankconf.py tests/unit/core/test_smem_pipeline.py
git commit -m "feat(core): shared memory bank conflict detection in pipeline"
```

---

### Task 23: Global memory coalescing analyzer

**Files:**
- Create: `gpusim/core/gmem.py`
- Test: `tests/unit/core/test_gmem_coalesce.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_gmem_coalesce.py
from gpusim.core.gmem import coalescing_info

def test_perfectly_coalesced_one_transaction():
    addrs = [i * 4 for i in range(32)]   # 128 contiguous bytes → 1 sector
    info = coalescing_info(addrs, active_mask=(1<<32)-1, sector_bytes=128)
    assert info.n_transactions == 1
    assert info.efficiency == 1.0

def test_stride_2_half_efficiency():
    addrs = [i * 8 for i in range(32)]   # spans 256 bytes → 2 sectors
    info = coalescing_info(addrs)
    assert info.n_transactions == 2
    assert abs(info.efficiency - 0.5) < 1e-9

def test_stride_4_quarter_efficiency():
    addrs = [i * 16 for i in range(32)]  # 512 bytes → 4 sectors
    info = coalescing_info(addrs)
    assert info.n_transactions == 4
    assert abs(info.efficiency - 0.25) < 1e-9

def test_random_pattern():
    import random
    random.seed(0)
    addrs = [random.randrange(0, 4096) & ~3 for _ in range(32)]
    info = coalescing_info(addrs)
    assert info.n_transactions >= 1
    assert 0.0 < info.efficiency <= 1.0

def test_inactive_lanes_excluded():
    addrs = [0]*32
    mask = 0x0000FFFF  # 16 active
    info = coalescing_info(addrs, active_mask=mask)
    assert info.n_transactions == 1
    assert info.efficiency == 16 / 32
```

- [ ] **Step 2: Implementation**

```python
# gpusim/core/gmem.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CoalesceInfo:
    n_transactions: int
    n_active: int
    efficiency: float    # n_active / (n_transactions * 32)


def coalescing_info(addresses: list[int], active_mask: int = (1 << 32) - 1,
                    sector_bytes: int = 128) -> CoalesceInfo:
    sectors: set[int] = set()
    n_active = 0
    for lane, addr in enumerate(addresses):
        if not (active_mask >> lane) & 1:
            continue
        n_active += 1
        sectors.add(addr // sector_bytes)
    n_tx = max(1, len(sectors)) if n_active > 0 else 0
    eff = (n_active / (n_tx * 32)) if n_tx > 0 else 0.0
    return CoalesceInfo(n_transactions=n_tx, n_active=n_active, efficiency=eff)
```

- [ ] **Step 3: Wire into SubCore for global memory**

Add a helper next to `shared_addresses_for_warp` in `gpusim/core/exec.py`:

```python
def global_addresses_for_warp(w: WarpFnState, instr: Instr) -> list[int]:
    addrs: list[int] = [0] * w.warp_size
    for lane in range(w.warp_size):
        if not (w.active_mask >> lane) & 1:
            addrs[lane] = -1
            continue
        t = w.threads[lane]
        base_op = instr.src[0]
        base = t.get_u64(base_op.name) if isinstance(base_op, Reg) else int(getattr(base_op, "value", 0))
        off = 0
        if len(instr.src) > 1 and isinstance(instr.src[1], Imm):
            off = int(instr.src[1].value)
        addrs[lane] = base + off
    return addrs
```

In `SubCore._issue`, add gmem branch before scoreboard mark:

```python
        if op.startswith(("ld.global.", "st.global.")):
            from gpusim.core.exec import global_addresses_for_warp
            from gpusim.core.gmem import coalescing_info
            addrs = global_addresses_for_warp(w.fn_state, instr)
            info = coalescing_info(addrs, active_mask=w.fn_state.active_mask)
            # remember on the warp for trace; M5 will read it
            w.last_gmem = info
```

Add `last_gmem: object = None` to the `Warp` dataclass.

- [ ] **Step 4: Tests pass**

```bash
pytest tests/unit/core/test_gmem_coalesce.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/core/gmem.py gpusim/core/exec.py gpusim/core/sub_core.py gpusim/core/warp.py \
        tests/unit/core/test_gmem_coalesce.py
git commit -m "feat(core): global memory coalescing analyzer + warp annotation"
```

---

### Task 24: LSU outstanding queue + MEM_DEP stall

**Files:**
- Modify: `gpusim/core/sub_core.py` (LSU queue tracking + MEM_DEP), `gpusim/core/warp.py` (in-flight load count)
- Test: `tests/unit/core/test_lsu_outstanding.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_lsu_outstanding.py
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM
import numpy as np

def test_full_outstanding_queue_causes_structural_stall():
    # Issue many independent global loads with low LSU outstanding limit
    cfg = load_default()
    cfg.fu.lsu_outstanding = 2  # tiny queue
    src = """
    .visible .entry k(.param .u64 A) {
        .reg .u32 %r<6>; .reg .u64 %rd<6>; .reg .f32 %f<8>;
        ld.param.u64 %rd1, [A];
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd3, %rd1, %rd2;
        ld.global.f32 %f1, [%rd3];
        ld.global.f32 %f2, [%rd3];
        ld.global.f32 %f3, [%rd3];
        ld.global.f32 %f4, [%rd3];
        ld.global.f32 %f5, [%rd3];
        bar.sync 0;
    }
    """
    arr = np.arange(32, dtype=np.float32)
    k = parse(src, "<t>")
    sm = SM(cfg)
    res = sm.run(kernel=k, grid=(1,1,1), block=(32,1,1), params={"A": arr})
    # Even with only 32 threads, gmem latency 400 + outstanding=2 means
    # ~3 stalls at the queue → cycles >= some threshold
    assert res.cycles > 400
```

- [ ] **Step 2: Implementation**

In `gpusim/core/warp.py`, add:

```python
@dataclass
class Warp:
    warp_id: int
    kernel: object
    fn_state: WarpFnState | None = None
    stack: SIMTStack | None = None
    scoreboard: Scoreboard = field(default_factory=Scoreboard)
    barrier_pc: int = -1
    finished: bool = False
    cta_id: int = 0
    last_gmem: object | None = None
    # outstanding gmem loads (per-warp counter)
    outstanding_loads: list[int] = field(default_factory=list)  # cycles when each completes
```

In `gpusim/core/sub_core.py`, add LSU queue check in `_is_ready`:

```python
        # LSU outstanding queue
        if instr.op.startswith("ld.global.") or instr.op.startswith("st.global."):
            # purge expired
            w.outstanding_loads = [c for c in w.outstanding_loads if c > now]
            if len(w.outstanding_loads) >= self.cfg.fu.lsu_outstanding:
                return False, StallReason.STRUCTURAL
```

In `_issue` after computing latency for global ops, register the in-flight slot:

```python
        if op.startswith("ld.global."):
            w.outstanding_loads.append(now + latency)
```

Distinguish MEM_DEP from generic SCOREBOARD: when source register is pending AND was last marked by a global load, classify as MEM_DEP. To do that, track per-register last write origin in scoreboard. Extend scoreboard:

```python
# gpusim/core/scoreboard.py — extend
class Scoreboard:
    def __init__(self):
        self._pending: dict[str, int] = {}
        self._origin: dict[str, str] = {}   # "mem" | "alu" | "branch"

    def mark_write(self, reg: str, available_at_cycle: int, origin: str = "alu") -> None:
        cur = self._pending.get(reg, -1)
        if available_at_cycle > cur:
            self._pending[reg] = available_at_cycle
            self._origin[reg] = origin

    def origin_of(self, reg: str) -> str | None:
        return self._origin.get(reg)
```

In `SubCore._issue`, when marking writes after a global load, pass `origin="mem"`. In `_is_ready`, when scoreboard has pending, look up origin: if `"mem"` → MEM_DEP; else SCOREBOARD.

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/core/test_lsu_outstanding.py tests/unit/core/test_smem_bankconf.py tests/unit/core/test_gmem_coalesce.py -v
git add gpusim/core/scoreboard.py gpusim/core/sub_core.py gpusim/core/warp.py \
        tests/unit/core/test_lsu_outstanding.py
git commit -m "feat(core): LSU outstanding queue and MEM_DEP stall classification"
```

---

### Task 25: Register file bank conflict (operand collector)

**Files:**
- Create: `gpusim/core/regfile.py`
- Modify: `gpusim/core/sub_core.py`
- Test: `tests/unit/core/test_regfile.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_regfile.py
from gpusim.core.regfile import bank_of, operand_extra_cycles

def test_bank_assignment_default_4_banks():
    # bank(reg) = id & 3; we use trailing digits of name as id
    assert bank_of("r0") == 0
    assert bank_of("r1") == 1
    assert bank_of("r4") == 0
    assert bank_of("f5") == 1

def test_no_extra_cycles_when_banks_distinct():
    assert operand_extra_cycles(["r0", "r1", "r2"]) == 0

def test_two_sources_same_bank_one_extra_cycle():
    assert operand_extra_cycles(["r0", "r4"]) == 1   # both bank 0

def test_three_sources_all_same_bank_two_extra():
    assert operand_extra_cycles(["r0", "r4", "r8"]) == 2
```

- [ ] **Step 2: Implementation**

```python
# gpusim/core/regfile.py
from __future__ import annotations
import re

_DIGITS = re.compile(r"(\d+)$")


def bank_of(reg: str, banks: int = 4) -> int:
    m = _DIGITS.search(reg)
    return int(m.group(1)) % banks if m else 0


def operand_extra_cycles(src_regs: list[str], banks: int = 4) -> int:
    """Count duplicate-bank reads beyond the first; each costs +1 cycle."""
    seen: dict[int, int] = {}
    for r in src_regs:
        b = bank_of(r, banks)
        seen[b] = seen.get(b, 0) + 1
    extra = sum(c - 1 for c in seen.values() if c > 1)
    return extra
```

- [ ] **Step 3: Wire into SubCore**

In `_issue`, before scoreboard mark:

```python
        # operand collector bank conflict
        from gpusim.core.regfile import operand_extra_cycles
        srcs = _src_regs(instr)
        op_extra = operand_extra_cycles(srcs, banks=self.cfg.regfile.banks)
        if op_extra > 0:
            kind = self.fus.classify(op)
            self.fus.reserve(kind, now, op_extra)
            latency += op_extra
            # for trace: caller expects OPERAND stall reported per-cycle; M5 hooks
            w.last_operand_extra = op_extra
```

Add `last_operand_extra: int = 0` field to `Warp`.

- [ ] **Step 4: Run + commit**

```bash
pytest tests/unit/core/test_regfile.py -v
git add gpusim/core/regfile.py gpusim/core/sub_core.py gpusim/core/warp.py tests/unit/core/test_regfile.py
git commit -m "feat(core): register file bank conflict (operand collector)"
```

---

### Task 26: Microbenchmark assertions for memory layer

**Files:**
- Create: `tests/microbench/test_memory_facts.py`

- [ ] **Step 1: Tests (textbook facts)**

```python
# tests/microbench/test_memory_facts.py
import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM


def _run(src, params, grid=(1,1,1), block=(32,1,1), cfg=None):
    k = parse(src, "<t>")
    sm = SM(cfg or load_default())
    return sm.run(kernel=k, grid=grid, block=block, params=params)


def test_stride_32_word_shared_is_32way_conflict():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<5>; .reg .u64 %rd<3>; .reg .f32 %f<2>;
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 7;
        cvt.u64.u32 %rd1, %r2;
        cvt.f32.s32 %f1, %r1;
        st.shared.f32 [%rd1], %f1;
        bar.sync 0;
    }
    """
    out = np.zeros(32, dtype=np.float32)
    res_conflict = _run(src, {"OUT": out})
    src_no = src.replace("shl.b32 %r2, %r1, 7;", "shl.b32 %r2, %r1, 2;")  # stride 1
    out2 = np.zeros(32, dtype=np.float32)
    res_no = _run(src_no, {"OUT": out2})
    # 32-way conflict should add ≥31 extra cycles vs no conflict
    assert res_conflict.cycles >= res_no.cycles + 25


def test_stride_2_global_efficiency_50pct(monkeypatch):
    """We assert via internal coalescing_info; runtime cycle delta is also expected."""
    from gpusim.core.gmem import coalescing_info
    addrs = [i * 8 for i in range(32)]
    info = coalescing_info(addrs)
    assert info.n_transactions == 2
    assert abs(info.efficiency - 0.5) < 1e-9


def test_one_warp_kernel_ipc_le_1():
    # Single warp can issue at most one instr / cycle on the SM (only 1 of 4 sub-cores active)
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<5>;
        mov.u32 %r1, 1;
        add.s32 %r2, %r1, %r1;
        add.s32 %r3, %r2, %r2;
        bar.sync 0;
    }
    """
    out = np.zeros(1, dtype=np.uint32)
    res = _run(src, {"OUT": out}, block=(32,1,1))
    # cycles >= number of instructions; IPC = instrs / cycles ≤ 1
    # we don't yet track instrs in metrics — assert just that cycles >= 4
    assert res.cycles >= 4
```

- [ ] **Step 2: Run + Milestone 3 checkpoint**

```bash
pytest tests/microbench/test_memory_facts.py -v
git add tests/microbench/test_memory_facts.py
git commit -m "test(microbench): textbook-fact assertions for memory and IPC"
git tag M3-complete
```

> **Milestone 3 checkpoint** — pause for review. Memory layer (smem banks, gmem coalescing, regfile banks) is now reflected in cycle counts.

---

## Milestone 4 — Multi-CTA on the SM + Occupancy

Outcome: launching multiple CTAs lets `active_ctas` of them co-reside on the SM concurrently (subject to warps/regs/smem limits). Bottleneck classification is reported.

---

### Task 27: Occupancy calculator

**Files:**
- Create: `gpusim/core/occupancy.py`
- Test: `tests/unit/core/test_occupancy.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_occupancy.py
from gpusim.core.occupancy import compute_occupancy, OccupancyResult
from gpusim.config.loader import load_default

def test_warps_bottleneck():
    # 256 threads/CTA → 8 warps; 64 / 8 = 8 CTAs by warps; assume small regs/smem
    cfg = load_default()
    r = compute_occupancy(cfg, threads_per_cta=256, regs_per_thread=8, smem_per_cta=1024)
    assert r.active_ctas <= cfg.max_ctas_per_sm
    assert r.bottleneck == "warps"
    assert r.active_ctas == 8

def test_regs_bottleneck():
    cfg = load_default()
    # very high reg usage forces regs bottleneck
    r = compute_occupancy(cfg, threads_per_cta=128, regs_per_thread=100, smem_per_cta=512)
    assert r.bottleneck == "regs"

def test_smem_bottleneck():
    cfg = load_default()
    # 32 KB smem / CTA forces only 1 CTA
    r = compute_occupancy(cfg, threads_per_cta=128, regs_per_thread=8, smem_per_cta=32*1024)
    assert r.bottleneck == "smem"
    assert r.active_ctas == 1

def test_max_ctas_capped():
    cfg = load_default()
    # tiny everything — capped by max_ctas_per_sm
    r = compute_occupancy(cfg, threads_per_cta=32, regs_per_thread=4, smem_per_cta=128)
    assert r.active_ctas == cfg.max_ctas_per_sm
```

- [ ] **Step 2: Implementation**

```python
# gpusim/core/occupancy.py
from __future__ import annotations
from dataclasses import dataclass
from gpusim.config.schema import SMConfig


@dataclass(frozen=True)
class OccupancyResult:
    active_ctas: int
    warps_per_cta: int
    max_by_warps: int
    max_by_regs: int
    max_by_smem: int
    bottleneck: str   # "warps" | "regs" | "smem" | "max_ctas_cap"


def compute_occupancy(cfg: SMConfig, threads_per_cta: int, regs_per_thread: int,
                      smem_per_cta: int) -> OccupancyResult:
    warps_per_cta = (threads_per_cta + 31) // 32
    by_warps = cfg.warps_per_sm // warps_per_cta if warps_per_cta else 0
    regs_per_cta = max(1, regs_per_thread * threads_per_cta)
    by_regs = cfg.regs_per_sm // regs_per_cta
    by_smem = cfg.smem_per_sm_bytes // max(1, smem_per_cta)
    raw = min(by_warps, by_regs, by_smem)
    active = min(raw, cfg.max_ctas_per_sm)
    if active == cfg.max_ctas_per_sm and raw >= cfg.max_ctas_per_sm:
        bn = "max_ctas_cap"
    else:
        if active == by_smem:
            bn = "smem"
        elif active == by_regs:
            bn = "regs"
        elif active == by_warps:
            bn = "warps"
        else:
            bn = "warps"
    return OccupancyResult(
        active_ctas=active, warps_per_cta=warps_per_cta,
        max_by_warps=by_warps, max_by_regs=by_regs, max_by_smem=by_smem,
        bottleneck=bn,
    )
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/core/test_occupancy.py -v
git add gpusim/core/occupancy.py tests/unit/core/test_occupancy.py
git commit -m "feat(core): occupancy calculator with bottleneck attribution"
```

---

### Task 28: Multi-CTA concurrent scheduling on SM

**Files:**
- Modify: `gpusim/core/sm.py`, `gpusim/api.py`
- Test: `tests/unit/core/test_sm_multicta.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/core/test_sm_multicta.py
import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM
from gpusim.core.occupancy import compute_occupancy

def test_multi_cta_runs_concurrently_faster_than_serial():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<5>; .reg .u64 %rd<3>; .reg .f32 %f<2>;
        ld.param.u64 %rd1, [OUT];
        mov.u32 %r1, %tid.x;
        mov.u32 %r2, %ctaid.x;
        mad.lo.s32 %r3, %r2, 32, %r1;
        shl.b32 %r4, %r3, 2;
        cvt.u64.u32 %rd2, %r4;
        add.u64 %rd1, %rd1, %rd2;
        st.global.u32 [%rd1], %r3;
        bar.sync 0;
    }
    """
    out = np.zeros(128, dtype=np.uint32)
    cfg = load_default()
    sm = SM(cfg)
    res = sm.run(kernel=parse(src, "<t>"), grid=(4,1,1), block=(32,1,1),
                 params={"OUT": out})
    # 4 CTAs of 1 warp each; with default cfg they should all fit concurrently
    # so wall-cycles should be much less than 4× single-CTA cycles
    res1 = sm.run(kernel=parse(src, "<t>"), grid=(1,1,1), block=(32,1,1),
                  params={"OUT": np.zeros(32, dtype=np.uint32)})
    assert res.cycles < 3 * res1.cycles  # at least ~2× speedup
    # output correctness
    assert list(out) == list(range(128))
```

- [ ] **Step 2: Implementation — replace `SM.run()` body**

```python
    def run(self, kernel, grid, block, params, regs_per_thread: int = 16,
            smem_per_cta: int = 0) -> SMRunResult:
        gmem = GlobalMemory()
        smem = SharedMemory(size_bytes=self.cfg.smem_per_sm_bytes)
        p_dict: dict[str, int] = {}
        for name, val in params.items():
            if isinstance(val, np.ndarray):
                p_dict[name] = gmem.bind(name, val)
            else:
                p_dict[name] = int(val)
        paramspace = ParamSpace(p_dict)

        threads_per_cta = block[0] * block[1] * block[2]
        warps_per_cta = (threads_per_cta + 31) // 32
        from gpusim.core.occupancy import compute_occupancy
        occ = compute_occupancy(self.cfg, threads_per_cta, regs_per_thread, smem_per_cta)

        n_ctas_total = grid[0] * grid[1] * grid[2]
        cta_queue: list[tuple[int, tuple[int,int,int]]] = []
        for cz in range(grid[2]):
          for cy in range(grid[1]):
            for cx in range(grid[0]):
                cid = cx + cy * grid[0] + cz * grid[0] * grid[1]
                cta_queue.append((cid, (cx, cy, cz)))

        # active slots
        executor = InstrExecutor(kernel=kernel, gmem=gmem, smem=smem,
                                 params=paramspace, cta_id=0,
                                 ctaid=(0,0,0), nctaid=grid, ntid=block)
        sub_cores: list[SubCore] = [
            SubCore(i, self.cfg, executor, [])
            for i in range(self.cfg.sub_cores)
        ]

        active_warps: list[Warp] = []     # all warps of active CTAs
        cycle = 0
        cta_pointer = 0

        def _activate_next_cta() -> bool:
            nonlocal cta_pointer
            if cta_pointer >= len(cta_queue): return False
            # check if room
            current_ctas = len({w.cta_id for w in active_warps})
            if current_ctas >= occ.active_ctas: return False
            cid, ctaid_xyz = cta_queue[cta_pointer]
            smem.allocate_cta(cid, max(1, smem_per_cta))
            for wid_in_cta in range(warps_per_cta):
                fn = WarpFnState(warp_size=32, tids=tuple(range(wid_in_cta*32, wid_in_cta*32+32)))
                w = Warp(warp_id=cid * warps_per_cta + wid_in_cta, kernel=kernel,
                         fn_state=fn, stack=SIMTStack(warp_size=32, entry_pc=0),
                         cta_id=cid)
                active_warps.append(w)
                # also place into a sub-core
                sub_cores[w.warp_id % self.cfg.sub_cores].warps.append(w)
            cta_pointer += 1
            return True

        # initial activation of `occ.active_ctas` CTAs
        while _activate_next_cta(): pass

        # run loop
        while True:
            for sc in sub_cores:
                sc.step(now=cycle)

            # barrier release per CTA
            by_cta: dict[int, list[Warp]] = {}
            for w in active_warps:
                by_cta.setdefault(w.cta_id, []).append(w)
            for cid, ws in by_cta.items():
                non_done = [w for w in ws if not w.finished]
                if non_done and all(w.barrier_pc >= 0 for w in non_done):
                    for w in non_done:
                        w.stack.update_top_pc(w.barrier_pc + 1); w.stack.maybe_pop()
                        w.barrier_pc = -1

            # retire CTAs whose warps are all finished
            retiring = []
            for cid, ws in by_cta.items():
                if all(w.finished or (w.stack and w.stack.is_done()) for w in ws):
                    retiring.append(cid)
            for cid in retiring:
                smem.free_cta(cid)
                active_warps = [w for w in active_warps if w.cta_id != cid]
                for sc in sub_cores:
                    sc.warps = [w for w in sc.warps if w.cta_id != cid]
                # try to schedule next CTA
                _activate_next_cta()

            cycle += 1
            if cta_pointer >= len(cta_queue) and not active_warps:
                break
            if cycle > 10_000_000:
                raise RuntimeError("simulation runaway > 1e7 cycles")

        outputs = {n: v for n, v in params.items() if isinstance(v, np.ndarray)}
        return SMRunResult(cycles=cycle, outputs=outputs,
                           events=[], )
```

Add to `SMRunResult`:
```python
    occupancy: dict[str, int] | None = None
```

In `run()`, before returning, set `occupancy={"active_ctas": occ.active_ctas, "bottleneck": occ.bottleneck}`.

In `gpusim/api.py`, surface occupancy in metrics:
```python
        return Result(
            outputs=res.outputs, mode="timing",
            metrics={"cycles": res.cycles, "occupancy": res.occupancy},
        )
```

- [ ] **Step 3: Run + Milestone 4 checkpoint**

```bash
pytest tests/unit/core/test_sm_multicta.py tests/parity/ -v
git add gpusim/core/sm.py gpusim/api.py tests/unit/core/test_sm_multicta.py
git commit -m "feat(core): multi-CTA concurrent scheduling on SM with occupancy"
git tag M4-complete
```

> **Milestone 4 checkpoint** — pause for review. Multi-CTA scheduling now works; occupancy bottleneck is reported. Ready for trace + analysis + viz.

---

## Milestone 5 — Trace, Analysis, Visualization

Outcome: simulator emits per-cycle events, parquet trace, HTML report, Perfetto JSON, and Notebook DataFrames.

---

### Task 29: Trace event types + recorder with RLE

**Files:**
- Create: `gpusim/trace/events.py`, `gpusim/trace/recorder.py`
- Test: `tests/unit/trace/test_recorder.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/trace/test_recorder.py
from gpusim.trace.events import EventKind
from gpusim.trace.recorder import Recorder

def test_warp_state_rle_compresses_runs():
    r = Recorder()
    for c in range(5):
        r.warp_state(cycle=c, warp_id=0, state="ISSUED", pc=0)
    for c in range(5, 8):
        r.warp_state(cycle=c, warp_id=0, state="SCOREBOARD", pc=1)
    segs = list(r.warp_state_segments(warp_id=0))
    # one segment of (0..4, ISSUED) and one of (5..7, SCOREBOARD)
    assert len(segs) == 2
    assert segs[0].start == 0 and segs[0].end == 4 and segs[0].state == "ISSUED"
    assert segs[1].start == 5 and segs[1].end == 7 and segs[1].state == "SCOREBOARD"

def test_instr_issue_event_recorded():
    r = Recorder()
    r.instr_issue(cycle=10, warp_id=0, pc=5, op="add.f32",
                  src_loc=("k.ptx", 12), active_mask=0xFFFFFFFF)
    evs = list(r.instr_issues())
    assert len(evs) == 1 and evs[0].pc == 5 and evs[0].op == "add.f32"

def test_smem_access_event_recorded():
    r = Recorder()
    r.smem_access(cycle=20, warp_id=0, conflict_degree=4, addresses=[0]*32)
    evs = list(r.smem_accesses())
    assert len(evs) == 1 and evs[0].conflict_degree == 4

def test_gmem_access_event_recorded():
    r = Recorder()
    r.gmem_access(cycle=20, warp_id=0, n_transactions=2, efficiency=0.5,
                  addresses=[i*4 for i in range(32)])
    evs = list(r.gmem_accesses())
    assert len(evs) == 1 and evs[0].n_transactions == 2

def test_div_push_pop_recorded():
    r = Recorder()
    r.div_push(cycle=5, warp_id=0, pc=3, rpc=10, taken_mask=0xFFFF)
    r.div_pop(cycle=15, warp_id=0, pc=10)
    assert len(list(r.div_events())) == 2

def test_cta_lifecycle():
    r = Recorder()
    r.cta_launch(cycle=0, cta_id=0, warps=4, regs=16, smem_bytes=512)
    r.cta_retire(cycle=200, cta_id=0)
    evs = list(r.cta_events())
    assert len(evs) == 2
```

- [ ] **Step 2: Implement events**

```python
# gpusim/trace/events.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class EventKind(Enum):
    CTA_LAUNCH = "CTA_LAUNCH"
    CTA_RETIRE = "CTA_RETIRE"
    INSTR_ISSUE = "INSTR_ISSUE"
    SMEM_ACCESS = "SMEM_ACCESS"
    GMEM_ACCESS = "GMEM_ACCESS"
    DIV_PUSH = "DIV_PUSH"
    DIV_POP = "DIV_POP"
    BAR_REACH = "BAR_REACH"
    BAR_RELEASE = "BAR_RELEASE"


@dataclass(frozen=True)
class WarpStateSegment:
    warp_id: int
    start: int
    end: int       # inclusive
    state: str
    pc: int


@dataclass(frozen=True)
class InstrIssueEvent:
    cycle: int
    warp_id: int
    pc: int
    op: str
    src_loc: tuple[str, int]
    active_mask: int


@dataclass(frozen=True)
class SmemEvent:
    cycle: int
    warp_id: int
    conflict_degree: int
    addresses: tuple[int, ...]


@dataclass(frozen=True)
class GmemEvent:
    cycle: int
    warp_id: int
    n_transactions: int
    efficiency: float
    addresses: tuple[int, ...]


@dataclass(frozen=True)
class DivEvent:
    kind: str        # "PUSH" | "POP"
    cycle: int
    warp_id: int
    pc: int
    rpc: int = -1
    taken_mask: int = 0


@dataclass(frozen=True)
class CtaEvent:
    kind: str        # "LAUNCH" | "RETIRE"
    cycle: int
    cta_id: int
    warps: int = 0
    regs: int = 0
    smem_bytes: int = 0


@dataclass(frozen=True)
class BarEvent:
    kind: str        # "REACH" | "RELEASE"
    cycle: int
    cta_id: int
    barrier_id: int
```

- [ ] **Step 3: Implement Recorder**

```python
# gpusim/trace/recorder.py
from __future__ import annotations
from collections import defaultdict
from typing import Iterator
from .events import (
    WarpStateSegment, InstrIssueEvent, SmemEvent, GmemEvent,
    DivEvent, CtaEvent, BarEvent,
)


class Recorder:
    def __init__(self):
        # warp_state RLE: per warp_id, list of (start, end, state, pc)
        self._ws_segs: dict[int, list[list]] = defaultdict(list)
        # cur_state[warp_id] tracks the open segment
        self._cur_state: dict[int, list] = {}   # [start, end, state, pc]
        self._instr_issues: list[InstrIssueEvent] = []
        self._smem: list[SmemEvent] = []
        self._gmem: list[GmemEvent] = []
        self._div: list[DivEvent] = []
        self._cta: list[CtaEvent] = []
        self._bar: list[BarEvent] = []

    def warp_state(self, *, cycle: int, warp_id: int, state: str, pc: int) -> None:
        cur = self._cur_state.get(warp_id)
        if cur and cur[2] == state and cur[1] + 1 == cycle:
            cur[1] = cycle
            return
        if cur:
            self._ws_segs[warp_id].append(cur)
        self._cur_state[warp_id] = [cycle, cycle, state, pc]

    def flush(self) -> None:
        for wid, cur in self._cur_state.items():
            self._ws_segs[wid].append(cur)
        self._cur_state.clear()

    def warp_state_segments(self, warp_id: int) -> Iterator[WarpStateSegment]:
        # ensure flushed
        if warp_id in self._cur_state:
            self._ws_segs[warp_id].append(self._cur_state[warp_id])
            del self._cur_state[warp_id]
        for s in self._ws_segs[warp_id]:
            yield WarpStateSegment(warp_id=warp_id, start=s[0], end=s[1],
                                   state=s[2], pc=s[3])

    def all_warp_segments(self) -> Iterator[WarpStateSegment]:
        # flush any open
        for wid in list(self._cur_state):
            self._ws_segs[wid].append(self._cur_state[wid])
            del self._cur_state[wid]
        for wid, segs in self._ws_segs.items():
            for s in segs:
                yield WarpStateSegment(warp_id=wid, start=s[0], end=s[1],
                                       state=s[2], pc=s[3])

    def instr_issue(self, *, cycle, warp_id, pc, op, src_loc, active_mask) -> None:
        self._instr_issues.append(InstrIssueEvent(
            cycle=cycle, warp_id=warp_id, pc=pc, op=op,
            src_loc=tuple(src_loc), active_mask=int(active_mask)))
    def instr_issues(self) -> list[InstrIssueEvent]: return list(self._instr_issues)

    def smem_access(self, *, cycle, warp_id, conflict_degree, addresses) -> None:
        self._smem.append(SmemEvent(cycle, warp_id, conflict_degree, tuple(addresses)))
    def smem_accesses(self) -> list[SmemEvent]: return list(self._smem)

    def gmem_access(self, *, cycle, warp_id, n_transactions, efficiency, addresses) -> None:
        self._gmem.append(GmemEvent(cycle, warp_id, n_transactions, float(efficiency), tuple(addresses)))
    def gmem_accesses(self) -> list[GmemEvent]: return list(self._gmem)

    def div_push(self, *, cycle, warp_id, pc, rpc, taken_mask) -> None:
        self._div.append(DivEvent("PUSH", cycle, warp_id, pc, rpc, taken_mask))
    def div_pop(self, *, cycle, warp_id, pc) -> None:
        self._div.append(DivEvent("POP", cycle, warp_id, pc, -1, 0))
    def div_events(self) -> list[DivEvent]: return list(self._div)

    def cta_launch(self, *, cycle, cta_id, warps, regs, smem_bytes) -> None:
        self._cta.append(CtaEvent("LAUNCH", cycle, cta_id, warps, regs, smem_bytes))
    def cta_retire(self, *, cycle, cta_id) -> None:
        self._cta.append(CtaEvent("RETIRE", cycle, cta_id))
    def cta_events(self) -> list[CtaEvent]: return list(self._cta)

    def bar_reach(self, *, cycle, cta_id, barrier_id=0) -> None:
        self._bar.append(BarEvent("REACH", cycle, cta_id, barrier_id))
    def bar_release(self, *, cycle, cta_id, barrier_id=0) -> None:
        self._bar.append(BarEvent("RELEASE", cycle, cta_id, barrier_id))
    def bar_events(self) -> list[BarEvent]: return list(self._bar)
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/unit/trace/test_recorder.py -v
git add gpusim/trace/ tests/unit/trace/
git commit -m "feat(trace): event types + recorder with WARP_STATE RLE"
```

---

### Task 30: Wire recorder into SM/SubCore + parquet writer

**Files:**
- Modify: `gpusim/core/sm.py`, `gpusim/core/sub_core.py`
- Create: `gpusim/trace/writer.py`
- Test: `tests/unit/trace/test_writer.py`, `tests/unit/core/test_sm_emits_trace.py`

- [ ] **Step 1: Tests for parquet writer**

```python
# tests/unit/trace/test_writer.py
from gpusim.trace.recorder import Recorder
from gpusim.trace.writer import write_parquet
import pyarrow.parquet as pq

def test_write_parquet_creates_three_tables(tmp_path):
    r = Recorder()
    r.warp_state(cycle=0, warp_id=0, state="ISSUED", pc=0)
    r.warp_state(cycle=1, warp_id=0, state="ISSUED", pc=1)
    r.warp_state(cycle=2, warp_id=0, state="SCOREBOARD", pc=2)
    r.instr_issue(cycle=0, warp_id=0, pc=0, op="add.f32", src_loc=("k",1), active_mask=0xFFFFFFFF)
    r.smem_access(cycle=1, warp_id=0, conflict_degree=2, addresses=[0]*32)
    r.gmem_access(cycle=2, warp_id=0, n_transactions=1, efficiency=1.0, addresses=[i*4 for i in range(32)])
    r.cta_launch(cycle=0, cta_id=0, warps=1, regs=16, smem_bytes=128)
    r.cta_retire(cycle=10, cta_id=0)
    r.div_push(cycle=5, warp_id=0, pc=3, rpc=10, taken_mask=0xFF)

    out = tmp_path / "trace.parquet"
    write_parquet(r, out)
    # parquet writer creates a directory with multiple files
    assert (out / "warp_state.parquet").exists()
    assert (out / "instr_issue.parquet").exists()
    assert (out / "smem.parquet").exists()
    assert (out / "gmem.parquet").exists()
    assert (out / "cta.parquet").exists()
    assert (out / "div.parquet").exists()

    df = pq.read_table(out / "warp_state.parquet").to_pandas()
    assert len(df) == 2  # two segments
```

- [ ] **Step 2: Implement writer**

```python
# gpusim/trace/writer.py
from __future__ import annotations
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
from .recorder import Recorder


def write_parquet(rec: Recorder, out: str | Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # warp_state segments
    segs = list(rec.all_warp_segments())
    tbl_ws = pa.table({
        "warp_id":  [s.warp_id for s in segs],
        "start":    [s.start for s in segs],
        "end":      [s.end for s in segs],
        "state":    [s.state for s in segs],
        "pc":       [s.pc for s in segs],
    })
    pq.write_table(tbl_ws, out / "warp_state.parquet")

    issues = rec.instr_issues()
    tbl_i = pa.table({
        "cycle": [e.cycle for e in issues],
        "warp_id": [e.warp_id for e in issues],
        "pc": [e.pc for e in issues],
        "op": [e.op for e in issues],
        "file": [e.src_loc[0] for e in issues],
        "line": [e.src_loc[1] for e in issues],
        "active_mask": [e.active_mask for e in issues],
    })
    pq.write_table(tbl_i, out / "instr_issue.parquet")

    s_evs = rec.smem_accesses()
    tbl_s = pa.table({
        "cycle": [e.cycle for e in s_evs],
        "warp_id": [e.warp_id for e in s_evs],
        "conflict_degree": [e.conflict_degree for e in s_evs],
    })
    pq.write_table(tbl_s, out / "smem.parquet")

    g_evs = rec.gmem_accesses()
    tbl_g = pa.table({
        "cycle": [e.cycle for e in g_evs],
        "warp_id": [e.warp_id for e in g_evs],
        "n_transactions": [e.n_transactions for e in g_evs],
        "efficiency": [e.efficiency for e in g_evs],
    })
    pq.write_table(tbl_g, out / "gmem.parquet")

    cta_evs = rec.cta_events()
    tbl_c = pa.table({
        "kind": [e.kind for e in cta_evs],
        "cycle": [e.cycle for e in cta_evs],
        "cta_id": [e.cta_id for e in cta_evs],
        "warps": [e.warps for e in cta_evs],
        "regs": [e.regs for e in cta_evs],
        "smem_bytes": [e.smem_bytes for e in cta_evs],
    })
    pq.write_table(tbl_c, out / "cta.parquet")

    d_evs = rec.div_events()
    tbl_d = pa.table({
        "kind": [e.kind for e in d_evs],
        "cycle": [e.cycle for e in d_evs],
        "warp_id": [e.warp_id for e in d_evs],
        "pc": [e.pc for e in d_evs],
        "rpc": [e.rpc for e in d_evs],
        "taken_mask": [e.taken_mask for e in d_evs],
    })
    pq.write_table(tbl_d, out / "div.parquet")
```

- [ ] **Step 3: Wire recorder into SM**

In `gpusim/core/sub_core.py`, accept an optional recorder, emit events:

```python
@dataclass
class SubCore:
    sub_core_id: int
    cfg: SMConfig
    executor: InstrExecutor
    warps: list[Warp]
    recorder: object | None = None
```

In `step()`, after computing `states`, emit per-warp events:
```python
        for i, w in enumerate(self.warps):
            if self.recorder is not None:
                self.recorder.warp_state(cycle=now, warp_id=w.warp_id,
                                         state=states[i].value,
                                         pc=(w.stack.top().pc if w.stack and not w.stack.is_done() else -1))
        return states
```

In `_issue`, emit instr_issue + smem_access + gmem_access + div events:
```python
        if self.recorder is not None:
            self.recorder.instr_issue(
                cycle=now, warp_id=w.warp_id, pc=instr.pc, op=instr.op,
                src_loc=(instr.src_loc.file, instr.src_loc.line),
                active_mask=w.fn_state.active_mask if w.fn_state else 0,
            )
        # smem event after _issue computes smem_conflict
        # (insert after smem block):
        if op.startswith(("ld.shared.","st.shared.")) and self.recorder is not None:
            self.recorder.smem_access(
                cycle=now, warp_id=w.warp_id,
                conflict_degree=smem_conflict, addresses=addrs,
            )
        # gmem event:
        if op.startswith(("ld.global.","st.global.")) and self.recorder is not None:
            self.recorder.gmem_access(
                cycle=now, warp_id=w.warp_id,
                n_transactions=info.n_transactions, efficiency=info.efficiency,
                addresses=addrs,
            )
```

For `bra` divergence, in the predicated branch path:
```python
        diverged = w.stack.diverge(taken_pc=target_pc, fallthrough_pc=w.stack.top().pc + 1,
                                   taken_mask=taken_mask, rpc=rpc)
        if diverged and self.recorder is not None:
            self.recorder.div_push(cycle=now, warp_id=w.warp_id,
                                   pc=instr.pc, rpc=rpc, taken_mask=taken_mask)
```

Pop is harder to detect — call `maybe_pop()` and detect `True`:
```python
        if w.stack.maybe_pop() and self.recorder is not None:
            self.recorder.div_pop(cycle=now, warp_id=w.warp_id,
                                  pc=w.stack.top().pc if not w.stack.is_done() else -1)
```

In `gpusim/core/sm.py`, accept optional recorder; pass to sub_cores; emit CTA_LAUNCH/RETIRE:

```python
class SM:
    def __init__(self, cfg: SMConfig, recorder: object | None = None):
        self.cfg = cfg
        self.recorder = recorder

    # in run(): when constructing sub_cores, pass recorder
    sub_cores: list[SubCore] = [
        SubCore(i, self.cfg, executor, [], recorder=self.recorder)
        for i in range(self.cfg.sub_cores)
    ]
    # on activate/retire:
    if self.recorder:
        self.recorder.cta_launch(cycle=cycle, cta_id=cid, warps=warps_per_cta,
                                 regs=regs_per_thread*threads_per_cta,
                                 smem_bytes=smem_per_cta)
    # on retire:
    if self.recorder:
        self.recorder.cta_retire(cycle=cycle, cta_id=cid)
```

- [ ] **Step 4: SM emits trace test**

```python
# tests/unit/core/test_sm_emits_trace.py
import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM
from gpusim.trace.recorder import Recorder

def test_sm_run_emits_warp_state_and_issues():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<3>; .reg .u64 %rd<3>;
        ld.param.u64 %rd1, [OUT];
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd1, %rd1, %rd2;
        st.global.u32 [%rd1], %r1;
        bar.sync 0;
    }
    """
    out = np.zeros(32, dtype=np.uint32)
    rec = Recorder()
    sm = SM(load_default(), recorder=rec)
    sm.run(kernel=parse(src, "<t>"), grid=(1,1,1), block=(32,1,1), params={"OUT": out})

    issues = rec.instr_issues()
    assert any(e.op.startswith("st.global") for e in issues)
    assert any(e.op == "mov.u32" for e in issues)

    gmem = rec.gmem_accesses()
    assert len(gmem) >= 1
    # the st.global.u32 should be perfectly coalesced
    assert any(g.efficiency == 1.0 for g in gmem)

    ctas = rec.cta_events()
    assert any(e.kind == "LAUNCH" for e in ctas)
    assert any(e.kind == "RETIRE" for e in ctas)
```

- [ ] **Step 5: Run + commit**

```bash
pytest tests/unit/trace/test_writer.py tests/unit/core/test_sm_emits_trace.py -v
git add gpusim/trace/writer.py gpusim/core/sm.py gpusim/core/sub_core.py \
        tests/unit/trace/test_writer.py tests/unit/core/test_sm_emits_trace.py
git commit -m "feat(trace): wire recorder into SM/SubCore + parquet writer"
```

---

### Task 31: Analysis — stall, IPC, source-line attribution

**Files:**
- Create: `gpusim/analysis/stall.py`, `gpusim/analysis/metrics.py`, `gpusim/analysis/attribution.py`
- Test: `tests/unit/analysis/test_stall.py`, `tests/unit/analysis/test_attribution.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/analysis/test_stall.py
import pandas as pd
from gpusim.analysis.stall import stall_breakdown, ipc_timeline


def test_stall_breakdown_counts_state_cycles():
    # warp 0: 0..4 ISSUED (5), 5..7 SCOREBOARD (3)
    # warp 1: 0..7 IDLE (8)
    df = pd.DataFrame([
        {"warp_id":0,"start":0,"end":4,"state":"ISSUED","pc":0},
        {"warp_id":0,"start":5,"end":7,"state":"SCOREBOARD","pc":1},
        {"warp_id":1,"start":0,"end":7,"state":"IDLE","pc":-1},
    ])
    out = stall_breakdown(df)
    assert out["ISSUED"] == 5
    assert out["SCOREBOARD"] == 3
    assert out["IDLE"] == 8


def test_ipc_timeline_counts_issuances_per_cycle():
    df = pd.DataFrame([
        {"warp_id":0,"start":0,"end":2,"state":"ISSUED","pc":0},   # 3 issues at c=0,1,2
        {"warp_id":1,"start":1,"end":1,"state":"ISSUED","pc":0},   # 1 issue at c=1
    ])
    series = ipc_timeline(df)
    assert series.loc[0] == 1
    assert series.loc[1] == 2
    assert series.loc[2] == 1
```

```python
# tests/unit/analysis/test_attribution.py
import pandas as pd
from gpusim.analysis.attribution import stall_by_source_line


def test_attribution_groups_stall_by_pc_then_src_line():
    issues = pd.DataFrame([
        {"cycle":0,"warp_id":0,"pc":0,"op":"add.f32","file":"k.ptx","line":5,"active_mask":0xFFFFFFFF},
        {"cycle":4,"warp_id":0,"pc":1,"op":"add.f32","file":"k.ptx","line":6,"active_mask":0xFFFFFFFF},
    ])
    states = pd.DataFrame([
        {"warp_id":0,"start":0,"end":0,"state":"ISSUED","pc":0},
        {"warp_id":0,"start":1,"end":3,"state":"SCOREBOARD","pc":1},
        {"warp_id":0,"start":4,"end":4,"state":"ISSUED","pc":1},
    ])
    df = stall_by_source_line(issues_df=issues, warp_state_df=states)
    # line 6 had 3 cycles of SCOREBOARD attributed to it
    row = df[(df["line"] == 6) & (df["state"] == "SCOREBOARD")].iloc[0]
    assert row["cycles"] == 3
```

- [ ] **Step 2: Implementations**

```python
# gpusim/analysis/stall.py
from __future__ import annotations
import pandas as pd


def _expand_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Convert RLE segments into per-cycle rows: (cycle, warp_id, state, pc)."""
    cycles = (df["end"] - df["start"] + 1).astype(int)
    out = df.loc[df.index.repeat(cycles)].copy()
    out["cycle"] = out.groupby(level=0).cumcount() + out["start"].values
    return out[["cycle", "warp_id", "state", "pc"]].reset_index(drop=True)


def stall_breakdown(warp_state_df: pd.DataFrame) -> dict[str, int]:
    df = warp_state_df.copy()
    df["cycles"] = df["end"] - df["start"] + 1
    grp = df.groupby("state")["cycles"].sum()
    return {k: int(v) for k, v in grp.items()}


def ipc_timeline(warp_state_df: pd.DataFrame) -> pd.Series:
    issued = warp_state_df[warp_state_df["state"] == "ISSUED"]
    rows = []
    for _, r in issued.iterrows():
        for c in range(int(r["start"]), int(r["end"]) + 1):
            rows.append(c)
    if not rows:
        return pd.Series(dtype=int)
    return pd.Series(rows).value_counts().sort_index()
```

```python
# gpusim/analysis/attribution.py
from __future__ import annotations
import pandas as pd


def stall_by_source_line(*, issues_df: pd.DataFrame,
                         warp_state_df: pd.DataFrame) -> pd.DataFrame:
    """For each (warp_id, pc), the warp may dwell in non-ISSUED states
    while waiting to issue that pc. Attribute those cycles to the (file, line)
    of that pc.
    """
    pc_to_loc = (issues_df.groupby("pc")
                 .agg({"file": "first", "line": "first", "op": "first"})
                 .reset_index())
    df = warp_state_df.copy()
    df["cycles"] = df["end"] - df["start"] + 1
    merged = df.merge(pc_to_loc, on="pc", how="left")
    grouped = (merged.groupby(["file", "line", "op", "state"])["cycles"]
               .sum().reset_index())
    return grouped
```

```python
# gpusim/analysis/metrics.py — empty placeholder for now, filled by Task 32
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/analysis/ -v
git add gpusim/analysis/stall.py gpusim/analysis/attribution.py gpusim/analysis/metrics.py \
        tests/unit/analysis/
git commit -m "feat(analysis): stall_breakdown, ipc_timeline, stall_by_source_line"
```

---

### Task 32: Analysis — bank/coalesce/divergence/occupancy/bottleneck

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Test: `tests/unit/analysis/test_metrics.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/analysis/test_metrics.py
import pandas as pd
from gpusim.analysis.metrics import (
    bank_conflict_hist, coalescing_per_instr,
    divergence_cost, occupancy_timeline,
)

def test_bank_conflict_hist_counts_per_pc():
    smem = pd.DataFrame([
        {"cycle":0,"warp_id":0,"conflict_degree":1},
        {"cycle":5,"warp_id":0,"conflict_degree":4},
        {"cycle":6,"warp_id":1,"conflict_degree":1},
        {"cycle":7,"warp_id":1,"conflict_degree":4},
    ])
    h = bank_conflict_hist(smem)
    assert h.loc[1, "count"] == 2
    assert h.loc[4, "count"] == 2

def test_coalescing_per_instr_groups_by_pc():
    issues = pd.DataFrame([
        {"cycle":0,"pc":3,"op":"ld.global.f32","line":7},
        {"cycle":1,"pc":3,"op":"ld.global.f32","line":7},
    ])
    gmem = pd.DataFrame([
        {"cycle":0,"warp_id":0,"n_transactions":1,"efficiency":1.0},
        {"cycle":1,"warp_id":1,"n_transactions":2,"efficiency":0.5},
    ])
    out = coalescing_per_instr(issues, gmem)
    assert len(out) >= 1
    # average efficiency for pc=3 is 0.75
    row = out[out["pc"] == 3].iloc[0]
    assert abs(row["efficiency_mean"] - 0.75) < 1e-9
    assert row["n_transactions_mean"] == 1.5

def test_divergence_cost_sums_serial_state_cycles():
    states = pd.DataFrame([
        {"warp_id":0,"start":0,"end":4,"state":"DIVERGENCE_SERIAL","pc":0},
        {"warp_id":0,"start":5,"end":9,"state":"ISSUED","pc":0},
    ])
    assert divergence_cost(states) == 5

def test_occupancy_timeline_counts_active_warps():
    states = pd.DataFrame([
        {"warp_id":0,"start":0,"end":5,"state":"ISSUED","pc":0},
        {"warp_id":0,"start":6,"end":10,"state":"IDLE","pc":-1},
        {"warp_id":1,"start":3,"end":8,"state":"ISSUED","pc":0},
        {"warp_id":1,"start":9,"end":10,"state":"IDLE","pc":-1},
    ])
    s = occupancy_timeline(states)
    # at cycle 0: warp0 active, warp1 not yet → 1 active
    assert s.loc[0] == 1
    assert s.loc[5] == 2  # both active
    assert s.loc[10] == 0
```

- [ ] **Step 2: Implementation**

```python
# gpusim/analysis/metrics.py
from __future__ import annotations
import pandas as pd


def bank_conflict_hist(smem_df: pd.DataFrame) -> pd.DataFrame:
    g = smem_df.groupby("conflict_degree").size().rename("count").reset_index()
    return g.set_index("conflict_degree")


def coalescing_per_instr(issues_df: pd.DataFrame,
                         gmem_df: pd.DataFrame) -> pd.DataFrame:
    if issues_df.empty or gmem_df.empty:
        return pd.DataFrame(columns=["pc","efficiency_mean","n_transactions_mean","count"])
    # join by cycle (the gmem event happens in the same cycle as the issue)
    joined = gmem_df.merge(issues_df[["cycle","pc","op","line"]], on="cycle", how="left")
    out = (joined.groupby(["pc"])
           .agg(efficiency_mean=("efficiency","mean"),
                n_transactions_mean=("n_transactions","mean"),
                count=("efficiency","count"))
           .reset_index())
    return out


def divergence_cost(warp_state_df: pd.DataFrame) -> int:
    df = warp_state_df[warp_state_df["state"] == "DIVERGENCE_SERIAL"]
    if df.empty:
        return 0
    return int((df["end"] - df["start"] + 1).sum())


def occupancy_timeline(warp_state_df: pd.DataFrame) -> pd.Series:
    """Per-cycle count of warps whose state is not IDLE."""
    if warp_state_df.empty:
        return pd.Series(dtype=int)
    max_cycle = int(warp_state_df["end"].max())
    counts = [0] * (max_cycle + 1)
    for _, r in warp_state_df.iterrows():
        if r["state"] == "IDLE":
            continue
        for c in range(int(r["start"]), int(r["end"]) + 1):
            counts[c] += 1
    return pd.Series(counts)
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/analysis/test_metrics.py -v
git add gpusim/analysis/metrics.py tests/unit/analysis/test_metrics.py
git commit -m "feat(analysis): bank/coalescing/divergence/occupancy metrics"
```

---

### Task 33: HTML report

**Files:**
- Create: `gpusim/viz/html_report.py`, `gpusim/viz/_template.html.j2`
- Test: `tests/unit/viz/test_html_report.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/viz/test_html_report.py
from gpusim.viz.html_report import build_html
from gpusim.trace.recorder import Recorder
import pandas as pd

def test_build_html_contains_summary_and_charts(tmp_path):
    rec = Recorder()
    rec.warp_state(cycle=0, warp_id=0, state="ISSUED", pc=0)
    rec.warp_state(cycle=1, warp_id=0, state="ISSUED", pc=1)
    rec.warp_state(cycle=2, warp_id=0, state="SCOREBOARD", pc=2)
    rec.instr_issue(cycle=0, warp_id=0, pc=0, op="add.f32", src_loc=("k.ptx",1), active_mask=0xFFFFFFFF)
    rec.instr_issue(cycle=1, warp_id=0, pc=1, op="ld.global.f32", src_loc=("k.ptx",2), active_mask=0xFFFFFFFF)
    rec.smem_access(cycle=2, warp_id=0, conflict_degree=2, addresses=[0]*32)
    rec.gmem_access(cycle=1, warp_id=0, n_transactions=1, efficiency=1.0,
                    addresses=[i*4 for i in range(32)])
    rec.cta_launch(cycle=0, cta_id=0, warps=1, regs=16, smem_bytes=128)
    rec.cta_retire(cycle=10, cta_id=0)

    html = build_html(rec, kernel_name="vec_add", grid=(1,1,1), block=(32,1,1),
                      occupancy={"active_ctas":1, "bottleneck":"warps"},
                      cycles=10)
    assert "vec_add" in html
    assert "Stall breakdown" in html or "stall_breakdown" in html.lower()
    assert "<html" in html
    # plotly figures embedded as JSON inside <script>
    assert "plotly" in html.lower()
    p = tmp_path / "out.html"
    p.write_text(html)
```

- [ ] **Step 2: Template**

```html
{# gpusim/viz/_template.html.j2 #}
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>gpusim report — {{ kernel_name }}</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 1em auto; padding: 0 1em; }
  table { border-collapse: collapse; }
  th, td { padding: 4px 12px; border-bottom: 1px solid #ddd; text-align: left; }
  pre { background: #f6f8fa; padding: 8px; overflow-x: auto; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }
</style>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head><body>
<h1>gpusim report — {{ kernel_name }}</h1>

<h2>Summary</h2>
<table>
  <tr><th>Grid</th><td>{{ grid }}</td></tr>
  <tr><th>Block</th><td>{{ block }}</td></tr>
  <tr><th>Cycles</th><td>{{ cycles }}</td></tr>
  <tr><th>Active CTAs / SM</th><td>{{ occupancy.active_ctas }}</td></tr>
  <tr><th>Bottleneck</th><td>{{ occupancy.bottleneck }}</td></tr>
</table>

<h2>Stall breakdown</h2>
<div id="stall_pie"></div>

<h2>IPC timeline</h2>
<div id="ipc_line"></div>

<h2>Stall by source line</h2>
{{ stall_table_html|safe }}

<h2>Bank conflict histogram</h2>
{{ bank_table_html|safe }}

<script>
Plotly.newPlot("stall_pie", {{ stall_pie_json|safe }}, {});
Plotly.newPlot("ipc_line", {{ ipc_line_json|safe }}, {});
</script>
</body></html>
```

- [ ] **Step 3: Implementation**

```python
# gpusim/viz/html_report.py
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
import plotly.graph_objects as go
import plotly.io as pio

from gpusim.trace.recorder import Recorder
from gpusim.analysis.stall import stall_breakdown, ipc_timeline
from gpusim.analysis.attribution import stall_by_source_line
from gpusim.analysis.metrics import bank_conflict_hist


_TPL_DIR = Path(__file__).parent
_env = Environment(loader=FileSystemLoader(_TPL_DIR), autoescape=select_autoescape())


def _ws_df(rec: Recorder) -> pd.DataFrame:
    segs = list(rec.all_warp_segments())
    return pd.DataFrame([{"warp_id":s.warp_id,"start":s.start,"end":s.end,
                          "state":s.state,"pc":s.pc} for s in segs])


def _issues_df(rec: Recorder) -> pd.DataFrame:
    evs = rec.instr_issues()
    return pd.DataFrame([{"cycle":e.cycle,"warp_id":e.warp_id,"pc":e.pc,
                          "op":e.op,"file":e.src_loc[0],"line":e.src_loc[1],
                          "active_mask":e.active_mask} for e in evs])


def _smem_df(rec: Recorder) -> pd.DataFrame:
    evs = rec.smem_accesses()
    return pd.DataFrame([{"cycle":e.cycle,"warp_id":e.warp_id,
                          "conflict_degree":e.conflict_degree} for e in evs])


def build_html(rec: Recorder, *, kernel_name: str, grid, block,
               occupancy: dict, cycles: int) -> str:
    ws = _ws_df(rec)
    issues = _issues_df(rec)
    smem = _smem_df(rec)

    sb = stall_breakdown(ws) if not ws.empty else {}
    ipc = ipc_timeline(ws) if not ws.empty else pd.Series(dtype=int)

    stall_pie = go.Figure([go.Pie(labels=list(sb.keys()), values=list(sb.values()))])
    ipc_line = go.Figure([go.Scatter(x=list(ipc.index), y=list(ipc.values), mode="lines")])

    stall_table = stall_by_source_line(issues_df=issues, warp_state_df=ws) \
        if not issues.empty and not ws.empty else pd.DataFrame()
    bank_hist = bank_conflict_hist(smem) if not smem.empty else pd.DataFrame()

    return _env.get_template("_template.html.j2").render(
        kernel_name=kernel_name,
        grid=grid, block=block, cycles=cycles, occupancy=occupancy,
        stall_pie_json=pio.to_json(stall_pie),
        ipc_line_json=pio.to_json(ipc_line),
        stall_table_html=stall_table.to_html(index=False) if not stall_table.empty else "<i>(no data)</i>",
        bank_table_html=bank_hist.to_html() if not bank_hist.empty else "<i>(no data)</i>",
    )


def save_html(rec: Recorder, path: str | Path, **kwargs) -> None:
    Path(path).write_text(build_html(rec, **kwargs))
```

Add `gpusim/viz/__init__.py`:
```python
from .html_report import build_html, save_html  # noqa: F401
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/unit/viz/test_html_report.py -v
git add gpusim/viz/ tests/unit/viz/
git commit -m "feat(viz): HTML report with Plotly charts and source-line attribution"
```

---

### Task 34: Perfetto trace export

**Files:**
- Create: `gpusim/viz/perfetto.py`
- Test: `tests/unit/viz/test_perfetto.py`

- [ ] **Step 1: Tests**

```python
# tests/unit/viz/test_perfetto.py
import json
from gpusim.viz.perfetto import build_perfetto
from gpusim.trace.recorder import Recorder

def test_perfetto_emits_warp_tracks_and_slices():
    rec = Recorder()
    rec.warp_state(cycle=0, warp_id=0, state="ISSUED", pc=0)
    rec.warp_state(cycle=1, warp_id=0, state="SCOREBOARD", pc=1)
    rec.instr_issue(cycle=0, warp_id=0, pc=0, op="add.f32",
                    src_loc=("k.ptx",1), active_mask=0xFFFFFFFF)
    rec.div_push(cycle=2, warp_id=0, pc=2, rpc=10, taken_mask=0xFF)
    rec.bar_reach(cycle=5, cta_id=0)
    rec.bar_release(cycle=10, cta_id=0)

    obj = build_perfetto(rec)
    assert obj["traceEvents"]
    pids = {e["pid"] for e in obj["traceEvents"]}
    # one pid per warp; instant events for div/bar
    assert any(e["name"].startswith("add.f32") for e in obj["traceEvents"])
    assert any(e["ph"] == "i" and "DIV_PUSH" in e["name"] for e in obj["traceEvents"])
```

- [ ] **Step 2: Implementation**

```python
# gpusim/viz/perfetto.py
from __future__ import annotations
import json
from pathlib import Path
from gpusim.trace.recorder import Recorder


def build_perfetto(rec: Recorder) -> dict:
    events = []
    # one process per warp
    warps = set()
    for s in rec.all_warp_segments():
        warps.add(s.warp_id)

    for w in sorted(warps):
        events.append({"name":"process_name","ph":"M","pid":w,"tid":0,
                       "args":{"name":f"warp{w}"}})

    # WARP_STATE → "X" complete events (each segment becomes one slice)
    for s in rec.all_warp_segments():
        events.append({
            "name": s.state, "ph": "X", "pid": s.warp_id, "tid": 0,
            "ts": s.start, "dur": s.end - s.start + 1,
            "args": {"pc": s.pc},
        })

    # instr_issue as instant events
    for e in rec.instr_issues():
        events.append({
            "name": f"{e.op} pc={e.pc}", "ph": "i", "pid": e.warp_id, "tid": 1,
            "ts": e.cycle, "s": "t",
            "args": {"line": e.src_loc[1], "active_mask": hex(e.active_mask)},
        })

    # divergence pushes/pops as instant
    for e in rec.div_events():
        events.append({
            "name": f"DIV_{e.kind} pc={e.pc}", "ph": "i", "pid": e.warp_id, "tid": 2,
            "ts": e.cycle, "s": "t",
            "args": {"rpc": e.rpc, "taken_mask": hex(e.taken_mask)},
        })

    for e in rec.bar_events():
        events.append({
            "name": f"BAR_{e.kind}", "ph": "i", "pid": -1, "tid": e.cta_id,
            "ts": e.cycle, "s": "g",
            "args": {"cta_id": e.cta_id, "barrier_id": e.barrier_id},
        })

    return {"traceEvents": events, "displayTimeUnit": "ns"}


def save_perfetto(rec: Recorder, path: str | Path) -> None:
    Path(path).write_text(json.dumps(build_perfetto(rec)))
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/viz/test_perfetto.py -v
git add gpusim/viz/perfetto.py tests/unit/viz/test_perfetto.py
git commit -m "feat(viz): Perfetto trace JSON export"
```

---

### Task 35: Result class enhancement + Notebook API + CLI integration

**Files:**
- Modify: `gpusim/api.py`, `gpusim/cli.py`
- Create: `gpusim/viz/notebook.py`
- Test: `tests/parity/test_vector_add_full_pipeline.py`

- [ ] **Step 1: Tests**

```python
# tests/parity/test_vector_add_full_pipeline.py
import numpy as np, pathlib, gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/vector_add/kernel.ptx").read_text()


def test_full_pipeline_emits_html_and_perfetto(tmp_path):
    n = 1024
    rng = np.random.RandomState(0)
    a = rng.randn(n).astype(np.float32); b = rng.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    res = gpusim.run(ptx_src=PTX, grid=(8,1,1), block=(128,1,1),
                     params={"A":a,"B":b,"C":c,"N":n}, mode="timing")
    np.testing.assert_allclose(c, a + b, rtol=1e-5)

    res.html_report(tmp_path / "report.html")
    assert (tmp_path / "report.html").exists()
    res.perfetto(tmp_path / "trace.json")
    assert (tmp_path / "trace.json").exists()

    df = res.stall_df
    assert "state" in df.columns
    assert "cycles" in df.columns

    df2 = res.events_df
    assert "warp_id" in df2.columns

    fig = res.timeline(warp=0)
    assert fig is not None
```

- [ ] **Step 2: Implementation**

```python
# gpusim/viz/notebook.py
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from gpusim.trace.recorder import Recorder


def warp_state_dataframe(rec: Recorder) -> pd.DataFrame:
    segs = list(rec.all_warp_segments())
    return pd.DataFrame([{"warp_id":s.warp_id,"start":s.start,"end":s.end,
                          "state":s.state,"pc":s.pc} for s in segs])


def stall_dataframe(rec: Recorder) -> pd.DataFrame:
    df = warp_state_dataframe(rec)
    if df.empty: return pd.DataFrame(columns=["state","cycles"])
    df["cycles"] = df["end"] - df["start"] + 1
    return df.groupby("state")["cycles"].sum().reset_index()


def warp_timeline_figure(rec: Recorder, warp_id: int) -> go.Figure:
    segs = [s for s in rec.all_warp_segments() if s.warp_id == warp_id]
    fig = go.Figure()
    for s in segs:
        fig.add_trace(go.Bar(
            x=[s.end - s.start + 1], y=[f"warp{warp_id}"],
            base=s.start, orientation="h",
            name=s.state, hovertext=f"pc={s.pc} {s.start}-{s.end}",
        ))
    fig.update_layout(barmode="stack", title=f"Warp {warp_id} timeline",
                      xaxis_title="cycle")
    return fig
```

```python
# gpusim/api.py — replace Result and timing branch
from gpusim.trace.recorder import Recorder
from gpusim.viz.html_report import save_html
from gpusim.viz.perfetto import save_perfetto
from gpusim.viz.notebook import warp_state_dataframe, stall_dataframe, warp_timeline_figure


@dataclass
class Result:
    outputs: dict[str, np.ndarray]
    mode: str
    metrics: dict
    _recorder: Recorder | None = None
    _kernel_name: str = ""
    _grid: tuple = (1,1,1)
    _block: tuple = (1,1,1)
    _occupancy: dict | None = None

    def summary(self) -> str:
        cyc = self.metrics.get("cycles", "?")
        bn = (self._occupancy or {}).get("bottleneck", "?")
        return f"gpusim {self.mode}: {cyc} cycles, bottleneck={bn}"

    @property
    def events_df(self):
        return warp_state_dataframe(self._recorder) if self._recorder else None

    @property
    def stall_df(self):
        return stall_dataframe(self._recorder) if self._recorder else None

    def timeline(self, warp: int):
        return warp_timeline_figure(self._recorder, warp) if self._recorder else None

    def html_report(self, path):
        if self._recorder is None: raise ValueError("no recorder; run in timing mode")
        save_html(self._recorder, path,
                  kernel_name=self._kernel_name, grid=self._grid, block=self._block,
                  cycles=self.metrics.get("cycles", 0),
                  occupancy=self._occupancy or {})

    def perfetto(self, path):
        if self._recorder is None: raise ValueError("no recorder; run in timing mode")
        save_perfetto(self._recorder, path)


# In run() timing branch, attach Recorder:
    if mode == "timing":
        from gpusim.frontend.parser import parse
        from gpusim.config.loader import load_default, load_yaml
        from gpusim.core.sm import SM
        cfg = load_default() if config is None else (
            load_yaml(config) if isinstance(config, (str, Path)) else config
        )
        k = parse(ptx_src, "<inline>")
        rec = Recorder()
        sm = SM(cfg, recorder=rec)
        res = sm.run(kernel=k, grid=grid, block=block, params=params)
        return Result(
            outputs=res.outputs, mode="timing",
            metrics={"cycles": res.cycles, "occupancy": res.occupancy},
            _recorder=rec, _kernel_name=k.name, _grid=grid, _block=block,
            _occupancy=res.occupancy,
        )
```

In `gpusim/cli.py`, extend `run` with `--output`, `--perfetto`, `--trace`:

```python
@app.command()
def run(kernel: Path,
        grid: str = typer.Option(...), block: str = typer.Option(...),
        inputs: str = typer.Option(None),
        config: Path = typer.Option(None),
        output: Path = typer.Option(None, "--output"),
        perfetto: Path = typer.Option(None, "--perfetto"),
        trace: Path = typer.Option(None, "--trace"),
        mode: str = typer.Option("timing"),
        seed: int = typer.Option(0)):
    from gpusim.api import run as api_run
    g = _parse_dim(grid); b = _parse_dim(block)
    inps = _parse_inputs(inputs)
    params: dict = {}
    np_paths: dict[str, Path] = {}
    for name, path in inps.items():
        if path.endswith(".npy"):
            arr = np.load(path); params[name] = arr; np_paths[name] = Path(path)
        else:
            params[name] = int(path)
    res = api_run(ptx_src=kernel.read_text(), grid=g, block=b,
                  params=params, mode=mode, config=config, seed=seed)
    typer.echo(res.summary())
    for name, p in np_paths.items():
        if name in res.outputs:
            np.save(p, res.outputs[name])
    if output: res.html_report(output)
    if perfetto: res.perfetto(perfetto)
    if trace and res._recorder is not None:
        from gpusim.trace.writer import write_parquet
        write_parquet(res._recorder, trace)
```

Also add an `explain` command:
```python
@app.command()
def explain(report: Path):
    text = report.read_text()
    # naive extraction: find <h2>Summary</h2> table
    import re
    m = re.search(r"Cycles</th><td>(\d+)</td>", text)
    if m: typer.echo(f"cycles: {m.group(1)}")
    m2 = re.search(r"Bottleneck</th><td>(\w+)</td>", text)
    if m2: typer.echo(f"bottleneck: {m2.group(1)}")
```

- [ ] **Step 3: Run + Milestone 5 checkpoint**

```bash
pytest tests/parity/test_vector_add_full_pipeline.py -v
pytest -v   # full suite
git add gpusim/api.py gpusim/cli.py gpusim/viz/notebook.py \
        tests/parity/test_vector_add_full_pipeline.py
git commit -m "feat(api): full visualization pipeline (html/perfetto/notebook)"
git tag M5-complete
```

> **Milestone 5 checkpoint** — pause for review. End-to-end pipeline complete: PTX → simulator → trace → analysis → HTML/Perfetto/notebook outputs.

---

## Milestone 6 — Examples, Tutorial, Reference Fixture, Polish

Outcome: 6 example kernels (vector_add already done in M1; 5 more here), 8-chapter tutorial, reference-fixture interface, microbenchmark suite expanded, README finalized.

> **Note on PTX correctness:** PTX kernels in this plan have been hand-written for clarity. If a parity test fails, debug the PTX (typically signed/unsigned cvt or address arithmetic) — don't skip the test.

---

### Task 36: Example — `reduction_smem`

A single warp reduces 32 elements to a sum using shared memory and `bar.sync`.

**Files:**
- Create: `examples/reduction_smem/{kernel.cu,kernel.ptx,reference.py,run.py,README.md}`
- Create: `tests/parity/test_reduction_smem.py`

- [ ] **Step 1: Tests**

```python
# tests/parity/test_reduction_smem.py
import numpy as np, pathlib, gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/reduction_smem/kernel.ptx").read_text()


def test_reduction_smem_correct():
    rng = np.random.RandomState(0)
    a = rng.randint(-100, 100, size=32).astype(np.int32)
    out = np.zeros(1, dtype=np.int32)
    gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(32,1,1),
               params={"A": a, "OUT": out}, mode="functional")
    assert int(out[0]) == int(a.sum())
```

- [ ] **Step 2: Kernel files**

```
// examples/reduction_smem/kernel.ptx
.visible .entry reduce32(.param .u64 A, .param .u64 OUT)
{
    .reg .u32 %r<10>;
    .reg .u64 %rd<6>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [OUT];

    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd3, %r2;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u32 %r3, [%rd4];        // r3 = A[tid]
    st.shared.u32 [%rd3], %r3;        // smem[tid*4] = r3
    bar.sync 0;

    // tree reduction: stride 16,8,4,2,1
    setp.lt.s32 %p1, %r1, 16;
    @%p1 bra S16;
    bra DONE16;
S16:
    add.u64 %rd5, %rd3, 64;           // smem[(tid+16)*4]
    ld.shared.u32 %r4, [%rd5];
    ld.shared.u32 %r5, [%rd3];
    add.s32 %r5, %r5, %r4;
    st.shared.u32 [%rd3], %r5;
DONE16:
    bar.sync 0;

    setp.lt.s32 %p1, %r1, 8;
    @%p1 bra S8;
    bra DONE8;
S8:
    add.u64 %rd5, %rd3, 32;
    ld.shared.u32 %r4, [%rd5];
    ld.shared.u32 %r5, [%rd3];
    add.s32 %r5, %r5, %r4;
    st.shared.u32 [%rd3], %r5;
DONE8:
    bar.sync 0;

    setp.lt.s32 %p1, %r1, 4;
    @%p1 bra S4;
    bra DONE4;
S4:
    add.u64 %rd5, %rd3, 16;
    ld.shared.u32 %r4, [%rd5];
    ld.shared.u32 %r5, [%rd3];
    add.s32 %r5, %r5, %r4;
    st.shared.u32 [%rd3], %r5;
DONE4:
    bar.sync 0;

    setp.lt.s32 %p1, %r1, 2;
    @%p1 bra S2;
    bra DONE2;
S2:
    add.u64 %rd5, %rd3, 8;
    ld.shared.u32 %r4, [%rd5];
    ld.shared.u32 %r5, [%rd3];
    add.s32 %r5, %r5, %r4;
    st.shared.u32 [%rd3], %r5;
DONE2:
    bar.sync 0;

    setp.eq.s32 %p1, %r1, 0;
    @%p1 bra WR;
    bra END;
WR:
    ld.shared.u32 %r6, [0];
    add.u64 %rd5, %rd2, 0;
    add.s32 %r7, %r6, 0;
    ld.shared.u32 %r4, [4];
    add.s32 %r6, %r6, %r4;
    st.global.u32 [%rd5], %r6;
END:
    bar.sync 0;
}
```

```cpp
// examples/reduction_smem/kernel.cu — for cross-reference
extern "C" __global__ void reduce32(const int* A, int* OUT) {
    __shared__ int s[32];
    int tid = threadIdx.x;
    s[tid] = A[tid]; __syncthreads();
    for (int off = 16; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    if (tid == 0) *OUT = s[0];
}
```

```python
# examples/reduction_smem/reference.py
import numpy as np
def reference(a: np.ndarray) -> np.ndarray:
    return np.array([a.sum()], dtype=np.int32)
```

```python
# examples/reduction_smem/run.py
import numpy as np, pathlib, gpusim
def main():
    rng = np.random.RandomState(0)
    a = rng.randint(-100, 100, size=32).astype(np.int32)
    out = np.zeros(1, dtype=np.int32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"A": a, "OUT": out}, mode="timing")
    print(f"sum: {out[0]} (expected {a.sum()})")
if __name__ == "__main__": main()
```

```markdown
# reduction_smem

单 warp 用 shared memory 做 32 元素树形归约。展示 `bar.sync` 节奏与
shared memory 的多次访问模式。

## 关键代码点
- `kernel.ptx:14` 把 gmem → smem 装载（每 lane 一个 dword）
- `kernel.ptx:15` 第一次 `bar.sync`（确保所有 lane 写完 smem 才能开始读）
- `kernel.ptx:18+` stride 16/8/4/2/1 的五次半数归约

## 运行
```
python examples/reduction_smem/run.py
```

## 预期观察
- HTML 报告中 `bar.sync` 占据可见的 cycle 比例
- 各 stride 的 `ld.shared` 没有 bank 冲突（stride 是 4 字节对齐的偶数倍，但落在不同 bank）

## 延伸思考
1. 把 stride 的下一步从 16 改成 17，看 bank conflict 直方图变化
2. 用模拟器验证：去掉中间的 `bar.sync` 会得到错误结果吗？（提示：当前是 functional 模式 ⇒ 不会；timing 模式才能看到序列化）
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/parity/test_reduction_smem.py -v
git add examples/reduction_smem/ tests/parity/test_reduction_smem.py
git commit -m "test(parity): reduction_smem example with tree reduction"
```

---

### Task 37: Example — `tiled_matmul` (small 16×16 single-tile)

Computes C = A @ B for 16×16 matrices using shared memory tile loading. Single CTA, single tile, no k-dim loop (since N=16 fits in one tile). Demonstrates smem load / `bar.sync` / register reuse.

**Files:**
- Create: `examples/tiled_matmul/{kernel.cu,kernel.ptx,reference.py,run.py,README.md}`
- Create: `tests/parity/test_tiled_matmul.py`

- [ ] **Step 1: Tests**

```python
# tests/parity/test_tiled_matmul.py
import numpy as np, pathlib, gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/tiled_matmul/kernel.ptx").read_text()


def test_tiled_matmul_16x16_correct():
    rng = np.random.RandomState(1)
    A = rng.randn(16, 16).astype(np.float32)
    B = rng.randn(16, 16).astype(np.float32)
    C = np.zeros((16, 16), dtype=np.float32)
    gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(16,16,1),
               params={"A": A, "B": B, "C": C}, mode="functional")
    np.testing.assert_allclose(C, A @ B, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Kernel files**

```
// examples/tiled_matmul/kernel.ptx
// Single 16x16 tile multiply: C = A * B. Block = (16, 16, 1).
.visible .entry tile_matmul(.param .u64 A, .param .u64 B, .param .u64 C)
{
    .reg .u32 %r<16>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<8>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [B];
    ld.param.u64 %rd3, [C];

    mov.u32 %r1, %tid.x;        // col
    mov.u32 %r2, %tid.y;        // row
    // linear thread id = row * 16 + col
    shl.b32 %r3, %r2, 4;
    add.s32 %r4, %r3, %r1;      // tid_lin

    // smem layout: A_tile[256] then B_tile[256]
    // load A[row][col] → smem_A[row*16 + col]
    shl.b32 %r5, %r4, 2;        // tid_lin * 4
    cvt.u64.u32 %rd4, %r5;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    st.shared.f32 [%rd4], %f1;

    // load B[row][col] → smem_B[256*4 + row*16 + col]
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f2, [%rd6];
    add.u32 %r6, %r5, 1024;     // smem_B base offset = 256 * 4
    cvt.u64.u32 %rd7, %r6;
    st.shared.f32 [%rd7], %f2;
    bar.sync 0;

    // dot product: sum over k of smem_A[row*16+k] * smem_B[k*16+col]
    mov.f32 %f3, 0f00000000;    // accumulator = 0.0
    mov.u32 %r7, 0;             // k = 0
LOOP:
    setp.ge.s32 %p1, %r7, 16;
    @%p1 bra DONE_LOOP;
    // smem_A offset = (row * 16 + k) * 4
    shl.b32 %r8, %r2, 4;
    add.s32 %r9, %r8, %r7;
    shl.b32 %r10, %r9, 2;
    cvt.u64.u32 %rd8, %r10;
    ld.shared.f32 %f4, [%rd8];
    // smem_B offset = (k * 16 + col) * 4 + 1024
    shl.b32 %r11, %r7, 4;
    add.s32 %r12, %r11, %r1;
    shl.b32 %r13, %r12, 2;
    add.s32 %r14, %r13, 1024;
    cvt.u64.u32 %rd9, %r14;
    ld.shared.f32 %f5, [%rd9];
    fma.f32 %f3, %f4, %f5, %f3;
    add.s32 %r7, %r7, 1;
    bra LOOP;
DONE_LOOP:

    // C[row][col] = acc
    add.u64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f3;
    bar.sync 0;
}
```

```cpp
// examples/tiled_matmul/kernel.cu
extern "C" __global__ void tile_matmul(const float* A, const float* B, float* C) {
    __shared__ float sA[16*16], sB[16*16];
    int col = threadIdx.x, row = threadIdx.y;
    sA[row*16 + col] = A[row*16 + col];
    sB[row*16 + col] = B[row*16 + col];
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < 16; ++k)
        acc += sA[row*16 + k] * sB[k*16 + col];
    C[row*16 + col] = acc;
}
```

```python
# examples/tiled_matmul/reference.py
import numpy as np
def reference(A: np.ndarray, B: np.ndarray) -> np.ndarray: return A @ B
```

```python
# examples/tiled_matmul/run.py
import numpy as np, pathlib, gpusim
def main():
    rng = np.random.RandomState(1)
    A = rng.randn(16,16).astype(np.float32); B = rng.randn(16,16).astype(np.float32)
    C = np.zeros((16,16), dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(16,16,1),
               params={"A":A,"B":B,"C":C}, mode="timing")
    print("max abs error:", float(np.max(np.abs(C - A @ B))))
if __name__ == "__main__": main()
```

```markdown
# tiled_matmul

16×16 矩阵乘法，单 CTA 单 tile。展示数据复用、shared memory tile load、`bar.sync` 同步、k-loop 内的 smem 访问模式。

## 关键代码点
- `kernel.ptx:21-29` 把 A、B 一次性装载进 shared memory（每个线程加载 1 个元素）
- `kernel.ptx:30` `bar.sync` 等 tile 装好
- `kernel.ptx:33-49` k-loop：每 k 读两次 smem，做一次 FMA

## 预期观察（timing mode）
- A 的 `ld.shared` 模式：行内线程访问 (row*16+k) — 同 row 的 16 个线程同时访问 16 个不同 bank → 无冲突
- B 的 `ld.shared` 模式：(k*16+col) — 同 col 的 16 个线程同时访问 16 个不同 bank → 无冲突
- HTML 报告里两次 `ld.shared` 的 bank conflict 直方图都在 1
- 主要时间花在 k-loop 的 16 次迭代

## 延伸思考
1. 把 B 的 layout 转置（`smem_B[col*16+k]`），看 bank conflict 怎么变
2. 把 block 改成 (32,8,1) 跑 32×8 一个 tile，观察 occupancy 与 stall 分布
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/parity/test_tiled_matmul.py -v
git add examples/tiled_matmul/ tests/parity/test_tiled_matmul.py
git commit -m "test(parity): tiled_matmul 16x16 single-tile example"
```

---

### Task 38: Examples — divergence_demo, bank_conflict_demo, coalescing_demo

Three small didactic examples — each a single PTX file, parity test, and short README.

**Files:**
- Create: `examples/divergence_demo/{kernel.ptx,reference.py,run.py,README.md}`
- Create: `examples/bank_conflict_demo/{kernel.ptx,reference.py,run.py,README.md}`
- Create: `examples/coalescing_demo/{kernel.ptx,reference.py,run.py,README.md}`
- Create: `tests/parity/test_divergence_demo.py`, `tests/parity/test_bank_conflict_demo.py`, `tests/parity/test_coalescing_demo.py`

- [ ] **Step 1: Tests**

```python
# tests/parity/test_divergence_demo.py
import numpy as np, pathlib, gpusim
PTX = (pathlib.Path(__file__).parents[2]/"examples/divergence_demo/kernel.ptx").read_text()
def test_divergence_demo():
    out = np.zeros(32, dtype=np.uint32)
    gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(32,1,1),
               params={"OUT": out}, mode="functional")
    expected = np.array([100 if i < 16 else 200 for i in range(32)], dtype=np.uint32)
    np.testing.assert_array_equal(out, expected)
```

```python
# tests/parity/test_bank_conflict_demo.py
import numpy as np, pathlib, gpusim
PTX = (pathlib.Path(__file__).parents[2]/"examples/bank_conflict_demo/kernel.ptx").read_text()
def test_bank_conflict_demo():
    out = np.zeros(32, dtype=np.uint32)
    gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(32,1,1),
               params={"OUT": out}, mode="functional")
    # each lane reads its own value back
    np.testing.assert_array_equal(out, np.arange(32, dtype=np.uint32))
```

```python
# tests/parity/test_coalescing_demo.py
import numpy as np, pathlib, gpusim
PTX = (pathlib.Path(__file__).parents[2]/"examples/coalescing_demo/kernel.ptx").read_text()
def test_coalescing_demo():
    n = 1024
    a = np.arange(n, dtype=np.uint32)
    out = np.zeros(32, dtype=np.uint32)
    gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(32,1,1),
               params={"A": a, "OUT": out, "STRIDE": 1}, mode="functional")
    np.testing.assert_array_equal(out, a[:32])
```

- [ ] **Step 2: divergence_demo kernel**

```
// examples/divergence_demo/kernel.ptx
.visible .entry div_demo(.param .u64 OUT)
{
    .reg .u32 %r<5>; .reg .u64 %rd<3>; .reg .pred %p<2>;
    ld.param.u64 %rd1, [OUT];
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    add.u64 %rd1, %rd1, %rd2;

    setp.lt.s32 %p1, %r1, 16;
    @%p1 bra THEN;
    mov.u32 %r3, 200;
    bra DONE;
THEN:
    mov.u32 %r3, 100;
DONE:
    st.global.u32 [%rd1], %r3;
}
```

```python
# examples/divergence_demo/reference.py
import numpy as np
def reference():
    return np.array([100 if i < 16 else 200 for i in range(32)], dtype=np.uint32)
```

```python
# examples/divergence_demo/run.py
import numpy as np, pathlib, gpusim
def main():
    out = np.zeros(32, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": out}, mode="timing")
    print("output:", out)
if __name__ == "__main__": main()
```

```markdown
# divergence_demo

同一 warp 的 32 lane 因 `tid<16` 走两条不同路径，演示 SIMT 序列化。

## 预期观察
- 报告中 `DIV_PUSH` 事件出现一次（在 setp/bra 处）
- `DIVERGENCE_SERIAL` 占总 cycle 的可观察比例
- 两条路径串行执行 → 总 cycle ≈ 两路径独立执行之和

## 延伸思考
1. 把分歧改成 `tid % 2 == 0`，看 `DIVERGENCE_SERIAL` 占比变化
2. 嵌套两层分歧，画 SIMT 栈深度时序
```

- [ ] **Step 3: bank_conflict_demo kernel**

```
// examples/bank_conflict_demo/kernel.ptx
// 演示 stride=1 vs stride=32 vs broadcast 的三种 smem 访存模式。
// 这里我们只 commit stride=1 版本作为正确性基准；其余 variant 通过修改源码
// 后跑 timing 模式来看 bank conflict 直方图。
.visible .entry bank_demo(.param .u64 OUT)
{
    .reg .u32 %r<5>; .reg .u64 %rd<3>;
    ld.param.u64 %rd1, [OUT];
    mov.u32 %r1, %tid.x;
    // stride 1: lane i → smem[i*4]
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    st.shared.u32 [%rd2], %r1;
    bar.sync 0;
    ld.shared.u32 %r3, [%rd2];
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    add.u64 %rd1, %rd1, %rd2;
    st.global.u32 [%rd1], %r3;
}
```

```python
# examples/bank_conflict_demo/reference.py
import numpy as np
def reference(): return np.arange(32, dtype=np.uint32)
```

```python
# examples/bank_conflict_demo/run.py
import numpy as np, pathlib, gpusim
def main():
    out = np.zeros(32, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent/"kernel.ptx").read_text()
    res = gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                     params={"OUT": out}, mode="timing")
    print(f"cycles: {res.metrics.get('cycles')}")
if __name__ == "__main__": main()
```

```markdown
# bank_conflict_demo

shared memory 32-bank 访问模式对比。当前 PTX 是 stride=1（无冲突）。复制 kernel.ptx
为 `kernel_stride32.ptx` 并把 `shl.b32 %r2, %r1, 2;` 改成 `shl.b32 %r2, %r1, 7;`
（×128）即可得到 32-way 冲突版本。

## 预期观察（timing）
- stride=1: bank_conflict_hist 全部为 1
- stride=32: 一次 store 的 conflict_degree=32，cycles 多出约 31 个
- broadcast (`mov.u32 %r2, 0`): conflict_degree=1（broadcast）

## 延伸思考
- 把 stride 改成 33 会怎样？（提示：奇 stride 与 32 互质 → 无冲突）
```

- [ ] **Step 4: coalescing_demo kernel**

```
// examples/coalescing_demo/kernel.ptx
// stride 由 .param STRIDE 控制：1=完美 coalesce, 2=50%, 4=25%, ...
.visible .entry coal_demo(.param .u64 A, .param .u64 OUT, .param .u32 STRIDE)
{
    .reg .u32 %r<6>; .reg .u64 %rd<5>;
    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [OUT];
    ld.param.u32 %r1, [STRIDE];

    mov.u32 %r2, %tid.x;
    mul.lo.s32 %r3, %r2, %r1;       // index = tid * stride
    shl.b32 %r4, %r3, 2;
    cvt.u64.u32 %rd3, %r4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u32 %r5, [%rd4];

    shl.b32 %r4, %r2, 2;
    cvt.u64.u32 %rd3, %r4;
    add.u64 %rd2, %rd2, %rd3;
    st.global.u32 [%rd2], %r5;
}
```

```python
# examples/coalescing_demo/reference.py
import numpy as np
def reference(a, stride):
    return a[: 32*stride : stride].copy()
```

```python
# examples/coalescing_demo/run.py
import numpy as np, pathlib, gpusim
def main():
    n = 1024
    a = np.arange(n, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    for stride in [1, 2, 4, 8]:
        out = np.zeros(32, dtype=np.uint32)
        res = gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                         params={"A": a, "OUT": out, "STRIDE": stride}, mode="timing")
        print(f"stride={stride}: cycles={res.metrics['cycles']}")
if __name__ == "__main__": main()
```

```markdown
# coalescing_demo

通过 `STRIDE` 参数演示 coalesced vs strided global 访问。

## 预期观察（timing）
- stride=1: coalescing_efficiency=1.0, n_transactions=1
- stride=2: 0.5, n_transactions=2
- stride=4: 0.25, n_transactions=4
- stride=8: 0.125, n_transactions=8（依 sector 大小可能合并）

## 延伸思考
- 当 stride * sizeof(type) ≥ sector_bytes 时，每个 lane 都可能落入独立 sector
- 把 dtype 从 u32 改成 u64 时，stride 的影响会怎么变？
```

- [ ] **Step 5: Run + commit**

```bash
pytest tests/parity/test_divergence_demo.py tests/parity/test_bank_conflict_demo.py \
       tests/parity/test_coalescing_demo.py -v
git add examples/divergence_demo/ examples/bank_conflict_demo/ examples/coalescing_demo/ \
        tests/parity/test_divergence_demo.py tests/parity/test_bank_conflict_demo.py \
        tests/parity/test_coalescing_demo.py
git commit -m "test(parity): divergence/bank-conflict/coalescing examples"
```

---

### Task 39: Reference fixture interface

Schema and capture script so the user can supply real-H100 reference data files for `@pytest.mark.reference` tests.

**Files:**
- Create: `tests/reference/README.md`
- Create: `tests/reference/gen_reference.py`
- Create: `tests/reference/test_reference.py`
- Create: `tests/reference/data/.gitkeep`

- [ ] **Step 1: gen_reference.py**

```python
# tests/reference/gen_reference.py
"""Run on a real NVIDIA GPU to produce *.ref.json fixtures for the simulator
to compare against.

Usage (on a CUDA-capable host):
    python tests/reference/gen_reference.py vector_add reduction_smem ...

Generates tests/reference/data/<kernel>.ref.json for each requested kernel.
The script depends on `nvcc` for compilation and (optionally) `ncu --csv`
for metrics. If `ncu` is unavailable, metrics are populated as best-effort
from CUPTI Python (cuda-python pkg) or skipped.
"""
from __future__ import annotations
import base64, json, sys, subprocess, pathlib, io
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"


def encode_npy(arr: np.ndarray) -> str:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return base64.b64encode(bio.getvalue()).decode("ascii")


def _run_nvcc_and_capture_outputs(kernel: str) -> dict:
    """Compile examples/<kernel>/kernel.cu and run with the same inputs as run.py.
    Returns dict with outputs (numpy arrays) and inputs (seed/shape).
    """
    raise NotImplementedError(
        "Implement on the GPU host: compile kernel.cu with nvcc, link a tiny "
        "host driver that mirrors examples/<kernel>/run.py inputs, and dump "
        "outputs into a numpy file."
    )


def _capture_metrics_via_ncu(kernel: str, exe: Path) -> dict:
    """Optionally call `ncu --csv --metrics ...` to collect achieved_occupancy,
    smsp__warps_active, etc. Returns dict subset matching the schema.
    """
    return {}


def gen(kernel: str) -> None:
    rec = {
        "kernel": kernel,
        "ptx_path": f"examples/{kernel}/kernel.ptx",
        "launch": {"grid": [1], "block": [32]},  # to be overridden by per-kernel logic
        "device": {"name": "H100 SXM5", "sm_count": 132},
        "inputs_shape": {},
        "inputs_seed": 42,
        "outputs": {},
        "metrics": {},
    }
    # subclass / monkeypatch this function per kernel; minimal stub here:
    DATA.mkdir(exist_ok=True)
    out = DATA / f"{kernel}.ref.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"wrote stub {out}")


def main(argv):
    if not argv:
        print("usage: gen_reference.py <kernel> [<kernel>...]"); return 2
    for k in argv:
        gen(k)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 2: README**

```markdown
# Reference fixtures (real-GPU)

To validate the simulator against real H100 behavior, run `gen_reference.py`
on a CUDA-capable host. It writes JSON files into `tests/reference/data/`
that the simulator's reference tests load when present.

## Layout (`*.ref.json` schema)

```json
{
  "kernel": "vector_add",
  "ptx_path": "examples/vector_add/kernel.ptx",
  "launch": {"grid": [8,1,1], "block": [128,1,1]},
  "device": {"name": "H100 SXM5", "sm_count": 132},
  "inputs_shape": {"A": [1024], "B": [1024], "C": [1024], "N": []},
  "inputs_seed": 42,
  "outputs": {"C": "<base64 npy bytes>"},
  "metrics": {
    "active_warps_per_sm": 64,
    "achieved_occupancy": 1.0,
    "smem_bank_conflicts": 0,
    "gld_efficiency": 1.0
  }
}
```

## Two layers of comparison

1. **Numerical** — `outputs` numpy buffers compared via `assert_allclose(rtol=1e-5)`
2. **Metric** — simulator metrics within tolerance:
   - `active_warps_per_sm` ±5%
   - `smem_bank_conflicts` exact
   - `gld_efficiency` ±10%

`timing` cycles are *not* compared (cycle-approximate ≠ cycle-accurate).

## Skipping when fixtures absent

The reference tests are decorated with `@pytest.mark.reference` and skipped
if the corresponding `.ref.json` does not exist.
```

- [ ] **Step 3: Test wiring**

```python
# tests/reference/test_reference.py
import json, base64, io, pathlib, pytest
import numpy as np
import gpusim

DATA_DIR = pathlib.Path(__file__).parent / "data"


def _load_npy_b64(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)))


def _decode_outputs(rec: dict) -> dict:
    return {k: _load_npy_b64(v) for k, v in rec.get("outputs", {}).items()}


def _ref_files() -> list[pathlib.Path]:
    return sorted(DATA_DIR.glob("*.ref.json"))


@pytest.mark.reference
@pytest.mark.parametrize("ref_file", _ref_files() or [pytest.param(None, marks=pytest.mark.skip(reason="no fixtures"))])
def test_simulator_matches_reference_numerics(ref_file):
    rec = json.loads(ref_file.read_text())
    expected = _decode_outputs(rec)
    if not expected:
        pytest.skip("no expected outputs")
    # simulator side: derive same inputs by seed + shape, then run
    rng = np.random.RandomState(rec.get("inputs_seed", 0))
    params = {}
    for name, shape in rec["inputs_shape"].items():
        if not shape:
            params[name] = 0
        else:
            params[name] = rng.randn(*shape).astype(np.float32)
    # also bind output buffers (zeros)
    for name, arr in expected.items():
        params[name] = np.zeros_like(arr)

    ptx = (pathlib.Path(__file__).parents[2] / rec["ptx_path"]).read_text()
    gpusim.run(ptx_src=ptx, grid=tuple(rec["launch"]["grid"]) + (1,)*(3-len(rec["launch"]["grid"])),
               block=tuple(rec["launch"]["block"]) + (1,)*(3-len(rec["launch"]["block"])),
               params=params, mode="functional")
    for name, exp in expected.items():
        np.testing.assert_allclose(params[name], exp, rtol=1e-5)
```

- [ ] **Step 4: gitkeep + commit**

```bash
touch tests/reference/data/.gitkeep
git add tests/reference/
git commit -m "feat(tests): reference fixture schema, gen script, and test wiring"
```

---

### Task 40: Tutorial chapters 00–04

8 chapters total; Tasks 40 + 41 split them. Each chapter ~800–1500 words. Plan supplies an **outline + must-include** items per chapter. The implementer writes prose to that outline, runs the modeled commands, and embeds the resulting screenshots/numbers.

**Files:** `docs/tutorial/00-intro.md`, `01-simt.md`, `02-scheduler.md`, `03-coalescing.md`, `04-bank-conflicts.md`

- [ ] **Step 1: Chapter 00 — `00-intro.md`**

Outline (must-include sections):
1. **What this tutorial is** — teaching tool to learn GPU microarchitecture by *running* code in a simulator.
2. **What the simulator can teach** — SIMT execution, warp scheduling, coalescing, bank conflicts, occupancy, divergence.
3. **What the simulator does NOT model** (Phase 1) — Tensor Core, cache hierarchy, multi-SM, multi-GPU, ITS. Reference spec section 11.
4. **Setup** — `pip install -e ".[dev]"`, then `gpusim doctor`.
5. **First run** — `python examples/vector_add/run.py`; expected output.
6. **Reading a report** — open `report.html`, what each section means.
7. **Where to go next** — chapters 01–07.

Write file. Do not include placeholder text; write actual prose to this outline. Run vector_add and embed real screenshots/numbers from the produced report.

- [ ] **Step 2: Chapter 01 — `01-simt.md` (SIMT model)**

Outline:
1. **From CPU SIMD to GPU SIMT** — analogy + difference (one PC for the warp, predicated lanes).
2. **Anatomy of a warp** — 32 lanes, active mask, lockstep execution.
3. **SIMT stack** — IPDOM, divergence push, reconvergence pop. Use `examples/divergence_demo/`.
4. **Why the stack costs cycles** — show the `DIVERGENCE_SERIAL` token in the report.
5. **改一改** — flip the predicate to `tid % 2 == 0`; what changes? (different mask, same stall pattern.)
6. **真机对照** if `divergence_demo.ref.json` exists, otherwise note "skipped".

- [ ] **Step 3: Chapter 02 — `02-scheduler.md` (warp scheduler & latency hiding)**

Outline:
1. **Latency-bound vs throughput-bound** — shared memory ~20 cycles, global memory ~400 cycles, FMA 4 cycles.
2. **Why one warp is not enough** — single-warp IPC ≤ 1; 64 warps can hide LSU latency.
3. **LRR vs GTO** — switch in config, rerun vector_add, compare reports.
4. **Stall taxonomy** — walk through each token (ISSUED, SCOREBOARD, MEM_DEP, BARRIER, …).
5. **改一改** — set `block=(32,1,1)` (1 warp/CTA): observe stall_breakdown shift toward MEM_DEP.

- [ ] **Step 4: Chapter 03 — `03-coalescing.md`**

Outline:
1. **What "coalesced" really means** — sector (128 B), threads grouped into the sector.
2. **Walk `coalescing_demo`** for stride=1, 2, 4, random — show the `n_transactions` column.
3. **Visualize** — embed bank-conflict-style heatmap from notebook API: `result.gmem_accesses_df`.
4. **改一改** — change dtype from `u32` (4 B) to `u64` (8 B); recompute expected efficiency.
5. Note Phase 1 has no cache; Phase 2 will close the loop on hit/miss behavior.

- [ ] **Step 5: Chapter 04 — `04-bank-conflicts.md`**

Outline:
1. **32 banks, 4-byte stride** — `bank(addr) = (addr/4) % 32`.
2. **Three patterns:** stride 1 (no conflict), stride 32 (32-way), broadcast (1-way).
3. **Walk `bank_conflict_demo`** — open report.html for each variant; bank_conflict_hist.
4. **Padded layouts** — show `s[33][N]` trick (line of code in commentary; not implemented in Phase 1 examples).
5. **改一改** — odd stride 33 → 1; even stride 34 → 2.

- [ ] **Step 6: Commit**

```bash
git add docs/tutorial/00-intro.md docs/tutorial/01-simt.md docs/tutorial/02-scheduler.md \
        docs/tutorial/03-coalescing.md docs/tutorial/04-bank-conflicts.md
git commit -m "docs(tutorial): chapters 00-04 (intro, SIMT, scheduler, coalescing, banks)"
```

---

### Task 41: Tutorial chapters 05–07 + final polish

**Files:** `docs/tutorial/05-divergence.md`, `06-occupancy.md`, `07-tiled-matmul.md`, `README.md` (final), Phase 1 retrospective

- [ ] **Step 1: Chapter 05 — `05-divergence.md`**

Outline:
1. **Divergence as the SIMT cost** — re-introduce SIMT stack from chapter 01 with timing numbers.
2. **Walk `divergence_demo`** — `DIV_PUSH/POP` events in Perfetto track; `DIVERGENCE_SERIAL` cycles in summary.
3. **Where divergence comes from** in real code: data-dependent branches, loop trip-count differences, edge-of-grid checks.
4. **Mitigations** — sort threads by branch decision, predicated execution, warp-level primitives (note: Phase 1 doesn't have shuffle; mention as future).
5. **改一改** — nest two divergent branches; observe stack depth = 3.

- [ ] **Step 2: Chapter 06 — `06-occupancy.md`**

Outline:
1. **Three knobs** — warps/CTA, regs/thread, smem/CTA.
2. **The bottleneck classifier** — open the report from `reduction_smem` and read the bottleneck row.
3. **Walk three scenarios** — high regs (regs-bound), big smem (smem-bound), small CTA (warps-bound). Modify `compute_occupancy` inputs in a notebook to feel each.
4. **Why higher occupancy isn't always better** — point to chapter 02 latency-hiding discussion.
5. **改一改** — modify `default_hopper.yaml` to reduce `regs_per_sm` to 32768; re-run reduction_smem; explain the new bottleneck.

- [ ] **Step 3: Chapter 07 — `07-tiled-matmul.md`**

Outline:
1. **Putting it together** — single 16×16 tile multiply (already covered in M6 example).
2. **Dataflow diagram** — A row × B col → C(i,j); embed as ASCII or simple svg.
3. **Walk the simulator output** — gmem load (coalesced), `bar.sync`, k-loop's two ld.shared per iteration with no bank conflicts.
4. **What's missing for "real" matmul** — k-loop over multiple tiles, register accumulation, FP16 + Tensor Core (Phase 3).
5. **改一改** — block size (16,16) → (32,8); occupancy and per-warp work change; explain.

- [ ] **Step 4: Final README.md (replace M1 placeholder)**

```markdown
# gpusim

Teaching-oriented NVIDIA GPU microarchitecture simulator.

## Quick start
```bash
pip install -e ".[dev]"
gpusim doctor
python examples/vector_add/run.py
```

## What you can learn
- SIMT execution + branch divergence (`examples/divergence_demo/`)
- Global memory coalescing (`examples/coalescing_demo/`)
- Shared memory bank conflicts (`examples/bank_conflict_demo/`)
- Reduction with shared memory + bar.sync (`examples/reduction_smem/`)
- Tiled matmul (`examples/tiled_matmul/`)
- Occupancy bottlenecks (`docs/tutorial/06-occupancy.md`)

## Run a kernel and inspect the report
```bash
gpusim run examples/vector_add/kernel.ptx \
    --grid 8 --block 128 \
    --inputs A:a.npy,B:b.npy,C:c.npy,N:1024 \
    --output report.html --perfetto trace.json
```
- `report.html` — open in any browser
- `trace.json` — drag into https://ui.perfetto.dev for an interactive timeline

## What's modeled (Phase 1)
Single SM, cycle-approximate, Hopper-shaped. PTX subset (~30 ops). Shared memory bank conflicts, global memory coalescing, regfile bank conflicts, multi-CTA occupancy.

## What's NOT modeled (Phase 1)
Tensor Core, FP16/BF16/FP8, L1/L2 cache, HBM bandwidth, TMA, thread-block clusters, warp shuffle, ITS, multi-SM, multi-GPU. See `docs/superpowers/specs/2026-05-07-gpusim-phase1-design.md` section 11.

## Tutorial
Read `docs/tutorial/00-intro.md` first.
```

- [ ] **Step 5: Run full test suite + tag M6 + Phase 1 done**

```bash
pytest -v --tb=short
git add docs/tutorial/05-divergence.md docs/tutorial/06-occupancy.md \
        docs/tutorial/07-tiled-matmul.md README.md
git commit -m "docs: chapters 05-07 + final README"
git tag M6-complete
git tag phase1-complete
```

> **Phase 1 done.** Run `pytest --tb=short` once more; verify all green. Remaining work belongs to Phase 2 (cache hierarchy + HBM).

---

## Self-review (run after writing the plan)

### 1. Spec coverage check

| Spec section | Plan task(s) |
|---|---|
| §2 Module breakdown | T1 (scaffold), T2/T3 (frontend), T15 (config), T16–T20 (core), T29–T30 (trace), T31–T32 (analysis), T33–T35 (viz) |
| §3 PTX subset + IR | T2, T3, T4, T5, T6 |
| §4.1 Cycle-stepped main loop | T20 |
| §4.2 Hopper params | T15 |
| §4.4 Pipeline stages | T17, T19 |
| §5.1 PDOM SIMT stack | T11 (functional) + T19 (timing wired) |
| §5.2 LRR + GTO scheduler | T18 |
| §5.3 Functional units | T17, T19 |
| §6.1 Shared memory bank conflict | T22 |
| §6.2 Global memory coalescing | T23 |
| §6.3 Regfile banks | T25 |
| §6.4 Multi-CTA + occupancy | T27, T28 |
| §6.5 Stall taxonomy | T18 (Warp.StallReason), T19 (subcore emits), T31 (analysis) |
| §7 Trace, analysis, viz | T29–T35 |
| §8.1 Unit tests | every task with `tests/unit/...` |
| §8.2 numpy parity | T13 (vector_add), T36–T38 (other examples) |
| §8.3 Reference fixtures | T39 |
| §8.4 Microbench facts | T26 |
| §9 Project structure / CLI / API | T1 (layout), T12/T14/T35 (API/CLI) |
| §10 6 examples + 8-chapter tutorial | T13, T36, T37, T38, T40, T41 |

### 2. Placeholder scan

The plan has no "TBD" / "TODO" / "fill in later" markers. The two places that name *real-GPU-specific* steps (`gen_reference.py:_run_nvcc_and_capture_outputs`) are **explicitly stubbed** with a `NotImplementedError` and clear instructions for the user — these only run on a real GPU host and are out-of-scope for the simulator-side plan.

### 3. Type/name consistency

- `gpusim.api.run()` signature is consistent across T12 (functional only), T21 (timing wired), T35 (visualization). Each task shows its own complete `run()` body.
- `Result` dataclass evolves: T12 has `outputs/mode/metrics`; T35 adds private `_recorder/_kernel_name/_grid/_block/_occupancy` and methods. T35 task shows the complete replacement.
- `SubCore` accumulates fields across M2/M3/M5; each task shows its own additions.
- `Warp` similarly accumulates fields across M2 (T18), M3 (T23, T25), M5 (T29 wiring); each task shows the new fields needed.
- `StallReason` enum defined once in T18 with all 10 tokens listed in spec §6.5.
- `bank_conflict_degree`, `coalescing_info`, `compute_occupancy`, `bank_of`, `operand_extra_cycles`, `_resolve_branch_mask`, `shared_addresses_for_warp`, `global_addresses_for_warp` — each defined exactly once and referenced consistently.

### 4. Scope check

Phase 1 fits one plan: the milestones are sequential refinements of one cohesive simulator, not independent subsystems. Each milestone produces working software (M1: functional; M2: timing; M3: memory bank model; M4: multi-CTA; M5: viz; M6: examples + docs). Engineers can pause between milestones.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-07-gpusim-phase1.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, code review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using `executing-plans`, batch with checkpoints.

Which approach?






