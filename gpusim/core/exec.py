from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import struct
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

    def load_s32(self, addr: int) -> int:
        buf, off = self._seg_for(addr)
        return int(buf[off:off+4].view(np.int32)[0])

    def store_s32(self, addr: int, v: int) -> None:
        buf, off = self._seg_for(addr)
        buf[off:off+4].view(np.int32)[0] = np.int32(v)

    def load_bytes(self, addr: int, n: int) -> bytes:
        buf, off = self._seg_for(addr)
        return bytes(buf[off:off+n])

    def store_bytes(self, addr: int, data: bytes) -> None:
        buf, off = self._seg_for(addr)
        buf[off:off+len(data)] = np.frombuffer(data, dtype=np.uint8)

    def load_f16(self, addr: int) -> float:
        buf, off = self._seg_for(addr)
        return float(buf[off:off+2].view(np.float16)[0])

    def store_f16(self, addr: int, v: float) -> None:
        buf, off = self._seg_for(addr)
        buf[off:off+2].view(np.float16)[0] = np.float16(v)

    def load_bf16(self, addr: int) -> float:
        import ml_dtypes
        buf, off = self._seg_for(addr)
        return float(buf[off:off+2].view(ml_dtypes.bfloat16)[0])

    def store_bf16(self, addr: int, v: float) -> None:
        import ml_dtypes
        buf, off = self._seg_for(addr)
        buf[off:off+2].view(ml_dtypes.bfloat16)[0] = ml_dtypes.bfloat16(v)

    def load_e4m3(self, addr: int) -> float:
        import ml_dtypes
        buf, off = self._seg_for(addr)
        return float(buf[off:off+1].view(ml_dtypes.float8_e4m3fn)[0])

    def store_e4m3(self, addr: int, v: float) -> None:
        import ml_dtypes
        buf, off = self._seg_for(addr)
        buf[off:off+1].view(ml_dtypes.float8_e4m3fn)[0] = ml_dtypes.float8_e4m3fn(v)

    def load_e5m2(self, addr: int) -> float:
        import ml_dtypes
        buf, off = self._seg_for(addr)
        return float(buf[off:off+1].view(ml_dtypes.float8_e5m2)[0])

    def store_e5m2(self, addr: int, v: float) -> None:
        import ml_dtypes
        buf, off = self._seg_for(addr)
        buf[off:off+1].view(ml_dtypes.float8_e5m2)[0] = ml_dtypes.float8_e5m2(v)

    def load_s8(self, addr: int) -> int:
        buf, off = self._seg_for(addr)
        return int(buf[off:off+1].view(np.int8)[0])

    def store_s8(self, addr: int, v: int) -> None:
        buf, off = self._seg_for(addr)
        buf[off:off+1].view(np.int8)[0] = np.int8(v)


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

    def load_f16(self, cta_id: int, offset: int) -> float:
        return float(self._cta[cta_id][offset:offset+2].view(np.float16)[0])

    def store_f16(self, cta_id: int, offset: int, value: float) -> None:
        self._cta[cta_id][offset:offset+2].view(np.float16)[0] = np.float16(value)


class ParamSpace:
    def __init__(self, params: dict[str, int]):
        self._params = dict(params)

    def read_u64(self, name: str) -> int:
        return int(self._params[name])

    def read_u32(self, name: str) -> int:
        return int(self._params[name]) & 0xFFFFFFFF


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
        # Use the register's own declared type for lookup; fall back to instruction ty.
        # _write always writes to both u32/s32 stores for 32-bit int types, so cross-type
        # reads (e.g. mov.u32 storing then add.s32 reading) work correctly.
        reg_ty = op.type if hasattr(op, "type") and op.type is not None else ty
        if reg_ty in (PtxType.u32, PtxType.b32):
            return t.get_u32(name)
        if reg_ty is PtxType.s32:
            return t.get_s32(name)
        if reg_ty in (PtxType.s64, PtxType.u64, PtxType.b64):
            return t.get_u64(name)
        if reg_ty in (PtxType.f32, PtxType.f16, PtxType.bf16, PtxType.e4m3,
                      PtxType.e5m2, PtxType.tf32):
            return t.get_f32(name)
        if reg_ty is PtxType.s8:
            return t.get_s32(name)
        if reg_ty is PtxType.pred:
            return t.get_pred(name)
        return t.get_u32(name)

    @staticmethod
    def _write(t: ThreadState, op: Reg, value, ty: PtxType):
        name = op.name
        if ty is PtxType.s32:
            # Write to both s32 and u32 stores so cross-type reads work
            t.set_s32(name, int(value))
            t.set_u32(name, int(value) & 0xFFFFFFFF)
        elif ty in (PtxType.u32, PtxType.b32):
            # Write to both u32 and s32 stores so cross-type reads work
            t.set_u32(name, int(value))
            t.set_s32(name, int(value))
        elif ty in (PtxType.s64, PtxType.u64, PtxType.b64):
            t.set_u64(name, int(value))
        elif ty in (PtxType.f32, PtxType.f16, PtxType.bf16, PtxType.e4m3,
                    PtxType.e5m2, PtxType.tf32):
            t.set_f32(name, float(value))   # store all floats in f32 register slot
        elif ty is PtxType.s8:
            # s8: store in s32/u32 for integer reads, and f32 for mma collect
            t.set_s32(name, int(value)); t.set_u32(name, int(value) & 0xFFFFFFFF)
            t.set_f32(name, float(value))
        elif ty is PtxType.pred:
            t.set_pred(name, bool(value))
        else:
            t.set_u32(name, int(value))

    def _resolve_special(self, t: ThreadState, sreg: str, lane: int, tid_x: int = -1) -> int:
        linear = tid_x if tid_x >= 0 else lane
        if sreg == "tid.x": return linear % self.ntid[0]
        if sreg == "tid.y": return (linear // self.ntid[0]) % self.ntid[1]
        if sreg == "tid.z": return linear // (self.ntid[0] * self.ntid[1])
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
        # mma.sync — warp-level operation (all lanes participate)
        if op.startswith("mma.sync."):
            from gpusim.core.tensor_core.mma_spec import parse_mma_op
            from gpusim.core.tensor_core.mma import execute_mma
            spec = parse_mma_op(op)
            if spec is not None:
                dst = instr.dst[0]; a = instr.src[0]; b = instr.src[1]
                c = instr.src[2] if len(instr.src) > 2 else dst
                execute_mma(spec, w, dst, a, b, c)
            return
        for lane in range(w.warp_size):
            if not self._lane_active(w, lane, instr):
                continue
            tid_x = w.tids[lane] if lane < len(w.tids) else lane
            self._exec_lane(w.threads[lane], instr, lane, tid_x=tid_x)

    def _exec_lane(self, t: ThreadState, instr: Instr, lane: int, tid_x: int = -1) -> None:
        op = instr.op
        ty = instr.type

        # ret — no-op; warp terminates when PC passes end of instructions
        if op == "ret":
            return

        # mov
        if op.startswith("mov."):
            src = instr.src[0]
            if isinstance(src, Reg) and src.name in (
                "tid.x","tid.y","tid.z","ntid.x","ntid.y","ntid.z",
                "ctaid.x","ctaid.y","ctaid.z","nctaid.x","nctaid.y","nctaid.z"):
                v = self._resolve_special(t, src.name, lane, tid_x=tid_x)
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
        if op in ("add.s32","add.u32","sub.s32","mul.lo.s32","mul.lo.u32",
                  "shl.b32","shr.s32","shr.u32","add.u64","sub.u64",
                  "and.b32","or.b32","xor.b32"):
            a = self._read(t, instr.src[0], ty)
            b = self._read(t, instr.src[1], ty)
            if op.startswith("add."):        r = a + b
            elif op.startswith("sub."):      r = a - b
            elif op in ("mul.lo.s32","mul.lo.u32"):  r = (a * b) & 0xFFFFFFFF
            elif op == "shl.b32":            r = (a << (b & 31)) & 0xFFFFFFFF
            elif op in ("shr.s32","shr.u32"): r = (a >> (b & 31))
            elif op == "and.b32":            r = (a & b) & 0xFFFFFFFF
            elif op == "or.b32":             r = (a | b) & 0xFFFFFFFF
            elif op == "xor.b32":            r = (a ^ b) & 0xFFFFFFFF
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
                if   ty is PtxType.f32:  v = self.gmem.load_f32(addr)
                elif ty is PtxType.f16:  v = self.gmem.load_f16(addr)
                elif ty is PtxType.bf16: v = self.gmem.load_bf16(addr)
                elif ty is PtxType.e4m3: v = self.gmem.load_e4m3(addr)
                elif ty is PtxType.e5m2: v = self.gmem.load_e5m2(addr)
                elif ty is PtxType.s8:   v = self.gmem.load_s8(addr)
                elif ty is PtxType.s32:  v = self.gmem.load_s32(addr)
                else:                    v = self.gmem.load_u32(addr)
            else:
                if ty is PtxType.f32:  v = self.smem.load_f32(self.cta_id, addr)
                elif ty is PtxType.f16: v = self.smem.load_f16(self.cta_id, addr)
                else:                  v = self.smem.load_u32(self.cta_id, addr)
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
                if   ty is PtxType.f32:  self.gmem.store_f32(addr, float(v))
                elif ty is PtxType.f16:  self.gmem.store_f16(addr, float(v))
                elif ty is PtxType.bf16: self.gmem.store_bf16(addr, float(v))
                elif ty is PtxType.e4m3: self.gmem.store_e4m3(addr, float(v))
                elif ty is PtxType.e5m2: self.gmem.store_e5m2(addr, float(v))
                elif ty is PtxType.s8:   self.gmem.store_s8(addr, int(v))
                elif ty is PtxType.s32:  self.gmem.store_s32(addr, int(v))
                else:                    self.gmem.store_u32(addr, int(v))
            else:
                if ty is PtxType.f32:  self.smem.store_f32(self.cta_id, addr, float(v))
                elif ty is PtxType.f16: self.smem.store_f16(self.cta_id, addr, float(v))
                else:                  self.smem.store_u32(self.cta_id, addr, int(v))
            return

        # gpusim.tma_desc — allocate descriptor (only lane 0 acts)
        if op == "gpusim.tma_desc":
            # Side-effect handled at SubCore._issue; per-lane is no-op
            return

        # cp.async.bulk.tensor.2d — handled at SubCore._issue (no per-lane work)
        if op.startswith("cp.async.bulk.tensor."):
            return

        # cp.async.bulk.commit_group / wait_group — no-op per lane (timing side handled elsewhere)
        if op in ("cp.async.bulk.commit_group", "cp.async.bulk.wait_group"):
            return

        # mbarrier.* — handled at SubCore._issue
        if op.startswith("mbarrier."):
            # mbarrier.try_wait writes a pred result; that's done at SubCore level.
            return

        raise NotImplementedError(f"opcode {op!r}")


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
        stack.update_top_pc(pc + 1); stack.maybe_pop()
        return False

    # all other ops: per-lane execution then PC++
    ex.execute(w, instr)
    stack.update_top_pc(pc + 1)
    stack.maybe_pop()
    return False


def shared_addresses_for_warp(w: WarpFnState, instr: Instr) -> list[int]:
    """Compute per-lane absolute byte offsets for a shared ld/st instr."""
    addrs: list[int] = [0] * w.warp_size
    for lane in range(w.warp_size):
        if not (w.active_mask >> lane) & 1:
            addrs[lane] = -1
            continue
        t = w.threads[lane]
        base_op = instr.src[0]
        if isinstance(base_op, Reg):
            base = t.get_u64(base_op.name)
        else:
            base = int(getattr(base_op, "value", 0))
        off = 0
        if len(instr.src) > 1 and isinstance(instr.src[1], Imm):
            off = int(instr.src[1].value)
        addrs[lane] = (base + off) & 0xFFFFFFFF
    return addrs


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


def _read_smem_matrix_fn(smem: "SharedMemory", cta_id: int, base: int,
                          rows: int, cols: int, dtype) -> np.ndarray:
    """Read a row-major rows×cols matrix from shared memory (functional_run helper)."""
    from gpusim.core.tensor_core.precision import storage_bytes, numpy_dtype_for
    elem = storage_bytes(dtype)
    nbytes = rows * cols * elem
    raw = bytes(smem._cta[cta_id][base:base + nbytes])
    return np.frombuffer(raw, dtype=numpy_dtype_for(dtype)).reshape(rows, cols).copy()


def _exec_tma_desc_fn(w: WarpFnState, instr: "Instr", tma_pool: "TensorDescriptorPool") -> None:
    """Functional-mode handler for gpusim.tma_desc: allocate descriptor and write handle."""
    t0 = w.threads[0]
    gmem_base = t0.get_u64(instr.src[0].name)
    dim_x = int(instr.src[1].value)
    dim_y = int(instr.src[2].value)
    stride_y = int(instr.src[3].value)
    elem_bytes = int(instr.src[4].value)
    handle = tma_pool.allocate(
        gmem_base=gmem_base, dim_x=dim_x, dim_y=dim_y,
        stride_y=stride_y, elem_bytes=elem_bytes,
    )
    handle_reg = instr.dst[0]
    for t in w.threads:
        t.set_u64(handle_reg.name, handle)


def _exec_cp_async_bulk_fn(w: WarpFnState, instr: "Instr",
                            gmem: GlobalMemory, smem: SharedMemory,
                            cta_id: int,
                            tma_pool: "TensorDescriptorPool",
                            mbar_pool: "MbarrierPool") -> None:
    """Functional-mode handler for cp.async.bulk.tensor.2d: synchronous copy + mbarrier arrive."""
    from gpusim.core.tma import do_bulk_copy_2d
    from gpusim.core.mbarrier import MbarrierPool
    t0 = w.threads[0]
    smem_dst = t0.get_u64(instr.src[0].name)
    handle = t0.get_u64(instr.src[1].name)
    mbar_addr = t0.get_u64(instr.src[2].name)
    desc = tma_pool.lookup(handle)
    do_bulk_copy_2d(gmem=gmem, smem=smem, cta_id=cta_id,
                    smem_dst=smem_dst, desc=desc)
    # Immediately arrive so try_wait succeeds (functional = instant TMA)
    mbar_pool.arrive(smem_addr=mbar_addr)


def _exec_mbarrier_fn(w: WarpFnState, instr: "Instr",
                       mbar_pool: "MbarrierPool") -> None:
    """Functional-mode handler for mbarrier.init / mbarrier.arrive / mbarrier.try_wait."""
    op = instr.op
    t0 = w.threads[0]
    if op.startswith("mbarrier.init."):
        mbar_addr = t0.get_u64(instr.src[0].name)
        count = int(instr.src[1].value)
        mbar_pool.init(smem_addr=mbar_addr, expected=count)
    elif op.startswith("mbarrier.arrive."):
        mbar_addr = t0.get_u64(instr.src[0].name)
        mbar_pool.arrive(smem_addr=mbar_addr)
    elif op.startswith("mbarrier.try_wait."):
        pred_reg = instr.dst[0]
        mbar_addr = t0.get_u64(instr.src[0].name)
        expected_phase = int(instr.src[1].value)
        result = mbar_pool.try_wait(smem_addr=mbar_addr, expected_phase=expected_phase)
        for t in w.threads:
            t.set_pred(pred_reg.name, bool(result))


def _is_bulk_store_op(op: str) -> bool:
    """Return True for cp.async.bulk.tensor store form (shared→global)."""
    return op.startswith("cp.async.bulk.tensor.") and "global.shared" in op


def _is_tma_mbarrier_op(op: str) -> bool:
    """Return True for warp-group-uniform TMA / mbarrier ops that need single execution."""
    return (op == "gpusim.tma_desc" or
            (op.startswith("cp.async.bulk.tensor.") and "global.shared" not in op) or
            op.startswith("mbarrier."))


def functional_run(ptx_src: str, *, params: dict[str, np.ndarray | int],
                   grid: tuple[int,int,int], block: tuple[int,int,int]) -> None:
    """Run kernel functionally over the grid. Mutates numpy arrays in `params` in place."""
    from gpusim.core.tma import TensorDescriptorPool
    from gpusim.core.mbarrier import MbarrierPool
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
            # Per-CTA TMA descriptor pool and mbarrier pool for functional mode
            fn_tma_pool = TensorDescriptorPool()
            fn_mbar_pool = MbarrierPool()
            warps = []
            for wid in range(warps_per_cta):
                tid_base = wid * 32
                tids = tuple(tid_base + i for i in range(32))
                w = WarpFnState(warp_size=32, tids=tids)
                warps.append((w, SIMTStack(warp_size=32, entry_pc=0)))

            done = [False] * len(warps)
            barrier_pcs = [-1] * len(warps)
            wgmma_pcs = [-1] * len(warps)  # warp-group wgmma sync
            tma_pcs = [-1] * len(warps)    # TMA/mbarrier barrier-like sync per warp-group
            while not all(done):
                progressed = False
                for i, (w, st) in enumerate(warps):
                    if done[i] or barrier_pcs[i] >= 0 or wgmma_pcs[i] >= 0 or tma_pcs[i] >= 0:
                        continue
                    pc = st.top().pc if not st.is_done() else -1
                    if pc < 0 or pc >= len(k.instrs):
                        finished = _step_warp(k, w, ex, st, {})
                        if finished: done[i] = True
                        progressed = True
                        continue
                    op = k.instrs[pc].op
                    if op == "bar.sync":
                        barrier_pcs[i] = pc
                        continue
                    # wgmma fence/commit/wait: skip (no-op in functional mode)
                    if op in ("wgmma.fence.sync.aligned",
                              "wgmma.commit_group.sync.aligned",
                              "wgmma.wait_group.sync.aligned"):
                        st.update_top_pc(pc + 1); st.maybe_pop()
                        progressed = True
                        continue
                    # cp.async.bulk commit/wait: no-op in functional mode
                    if op in ("cp.async.bulk.commit_group",
                              "cp.async.bulk.wait_group"):
                        st.update_top_pc(pc + 1); st.maybe_pop()
                        progressed = True
                        continue
                    # wgmma.mma_async: hold until all warps in warp-group arrive
                    if op.startswith("wgmma.mma_async."):
                        wgmma_pcs[i] = pc
                        continue
                    # cp.async.bulk.tensor store (shared→global): single-warp, execute immediately
                    if _is_bulk_store_op(op):
                        from gpusim.core.tma_store import do_bulk_store_2d
                        instr = k.instrs[pc]
                        handle = w.threads[0].get_u64(instr.src[0].name)
                        smem_src = w.threads[0].get_u64(instr.src[1].name)
                        desc = fn_tma_pool.lookup(handle)
                        do_bulk_store_2d(gmem=g, smem=s, cta_id=cta_id,
                                         smem_src=int(smem_src), desc=desc)
                        st.update_top_pc(pc + 1); st.maybe_pop()
                        progressed = True
                        continue
                    # TMA + mbarrier: warp-group-uniform — collect all warps, execute once from warp 0
                    if _is_tma_mbarrier_op(op):
                        tma_pcs[i] = pc
                        continue
                    finished = _step_warp(k, w, ex, st, {})
                    if finished: done[i] = True
                    progressed = True
                # Release bar.sync when all non-done warps are waiting
                if all((done[i] or barrier_pcs[i] >= 0) for i in range(len(warps))) \
                   and not all(done):
                    for i in range(len(warps)):
                        if barrier_pcs[i] >= 0:
                            warps[i][1].update_top_pc(barrier_pcs[i] + 1)
                            warps[i][1].maybe_pop()
                            barrier_pcs[i] = -1
                    progressed = True
                # Execute wgmma when all warps in a warp-group arrive at same PC
                # Warp-group: 4 consecutive warps (warps 0..3, 4..7, etc.)
                wg_size = 4
                n_wgs = max(1, warps_per_cta // wg_size)
                for wg in range(n_wgs):
                    wg_warp_ids = list(range(wg * wg_size,
                                             min((wg + 1) * wg_size, warps_per_cta)))
                    if len(wg_warp_ids) != 4:
                        continue
                    wg_pcs = [wgmma_pcs[i] for i in wg_warp_ids]
                    if all(p >= 0 for p in wg_pcs) and len(set(wg_pcs)) == 1:
                        wgmma_pc = wg_pcs[0]
                        instr = k.instrs[wgmma_pc]
                        from gpusim.core.tensor_core.mma_spec import parse_mma_op
                        from gpusim.core.tensor_core.wgmma import execute_wgmma_for_group
                        spec = parse_mma_op(instr.op)
                        if spec is not None and spec.is_async:
                            a_desc = instr.src[0]
                            b_desc = instr.src[1]
                            wg_warps = [warps[j][0] for j in wg_warp_ids]
                            a_base = wg_warps[0].threads[0].get_u64(a_desc.name)
                            b_base = wg_warps[0].threads[0].get_u64(b_desc.name)
                            a_arr = _read_smem_matrix_fn(
                                s, cta_id, base=int(a_base),
                                rows=spec.m, cols=spec.k, dtype=spec.dtype_a)
                            b_arr = _read_smem_matrix_fn(
                                s, cta_id, base=int(b_base),
                                rows=spec.k, cols=spec.n, dtype=spec.dtype_b)
                            dst_grp = instr.dst[0]
                            c_grp = instr.src[2] if len(instr.src) > 2 else dst_grp
                            execute_wgmma_for_group(
                                spec=spec, warps=wg_warps,
                                a_smem_array=a_arr, b_smem_array=b_arr,
                                dst_per_warp=tuple([dst_grp] * 4),
                                c_per_warp=tuple([c_grp] * 4),
                            )
                        # Advance all warps past the wgmma instruction
                        for j in wg_warp_ids:
                            warps[j][1].update_top_pc(wgmma_pc + 1)
                            warps[j][1].maybe_pop()
                            wgmma_pcs[j] = -1
                        progressed = True
                # Execute TMA/mbarrier ops when all warps in warp-group arrive at same PC
                # (warp-group uniform ops: execute once from warp 0, propagate result to all)
                for wg in range(n_wgs):
                    wg_warp_ids = list(range(wg * wg_size,
                                             min((wg + 1) * wg_size, warps_per_cta)))
                    if len(wg_warp_ids) != 4:
                        continue
                    wg_tma_pcs = [tma_pcs[j] for j in wg_warp_ids]
                    # A warp-group can fire when all warps are either done or waiting at same PC
                    active_pcs = [wg_tma_pcs[k2] for k2, j in enumerate(wg_warp_ids)
                                  if not done[j]]
                    if (active_pcs and all(p >= 0 for p in active_pcs)
                            and len(set(active_pcs)) == 1
                            and all(done[j] or wg_tma_pcs[k2] >= 0
                                    for k2, j in enumerate(wg_warp_ids))):
                        tma_pc = active_pcs[0]
                        instr = k.instrs[tma_pc]
                        op = instr.op
                        # elect first non-done warp as warp 0
                        wg_warps = [warps[j][0] for j in wg_warp_ids]
                        w0 = next(warps[j][0] for j in wg_warp_ids if not done[j])
                        if op == "gpusim.tma_desc":
                            handle = fn_tma_pool.allocate(
                                gmem_base=w0.threads[0].get_u64(instr.src[0].name),
                                dim_x=int(instr.src[1].value),
                                dim_y=int(instr.src[2].value),
                                stride_y=int(instr.src[3].value),
                                elem_bytes=int(instr.src[4].value),
                            )
                            handle_reg = instr.dst[0]
                            for ww in wg_warps:
                                for t in ww.threads:
                                    t.set_u64(handle_reg.name, handle)
                        elif op.startswith("cp.async.bulk.tensor."):
                            # Load form only (store form is handled per-warp above)
                            # src[0] = smem_dst, src[1] = handle, src[2] = mbar
                            from gpusim.core.tma import do_bulk_copy_2d
                            smem_dst = w0.threads[0].get_u64(instr.src[0].name)
                            handle = w0.threads[0].get_u64(instr.src[1].name)
                            mbar_addr = w0.threads[0].get_u64(instr.src[2].name)
                            desc = fn_tma_pool.lookup(handle)
                            do_bulk_copy_2d(gmem=g, smem=s, cta_id=cta_id,
                                            smem_dst=smem_dst, desc=desc)
                            fn_mbar_pool.arrive(smem_addr=mbar_addr)
                        elif op.startswith("mbarrier.init."):
                            mbar_addr = w0.threads[0].get_u64(instr.src[0].name)
                            count = int(instr.src[1].value)
                            fn_mbar_pool.init(smem_addr=mbar_addr, expected=count)
                        elif op.startswith("mbarrier.arrive."):
                            mbar_addr = w0.threads[0].get_u64(instr.src[0].name)
                            fn_mbar_pool.arrive(smem_addr=mbar_addr)
                        elif op.startswith("mbarrier.try_wait."):
                            pred_reg = instr.dst[0]
                            mbar_addr = w0.threads[0].get_u64(instr.src[0].name)
                            expected_phase = int(instr.src[1].value)
                            result = fn_mbar_pool.try_wait(smem_addr=mbar_addr,
                                                           expected_phase=expected_phase)
                            for ww in wg_warps:
                                for t in ww.threads:
                                    t.set_pred(pred_reg.name, bool(result))
                        # Advance non-done warps past this instruction
                        for k2, j in enumerate(wg_warp_ids):
                            if not done[j]:
                                warps[j][1].update_top_pc(tma_pc + 1)
                                warps[j][1].maybe_pop()
                            tma_pcs[j] = -1
                        progressed = True
                if not progressed:
                    raise RuntimeError("functional_run: no warp progressed (deadlock)")
            s.free_cta(cta_id)
