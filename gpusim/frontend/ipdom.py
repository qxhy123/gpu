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
        if ins.op == "bra":
            ip = ipdom_of(pc)
            if ip is not None:
                out[pc] = ip
    return out
