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
    "<": "LT", ">": "GT", "+": "PLUS",
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
