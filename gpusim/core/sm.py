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


def _read_smem_matrix(smem, cta_id: int, base: int, rows: int, cols: int,
                      dtype) -> np.ndarray:
    """Read a row-major rows×cols matrix from shared memory."""
    from gpusim.core.tensor_core.precision import storage_bytes, numpy_dtype_for
    elem = storage_bytes(dtype)
    nbytes = rows * cols * elem
    raw = bytes(smem._cta[cta_id][base:base + nbytes])
    return np.frombuffer(raw, dtype=numpy_dtype_for(dtype)).reshape(rows, cols).copy()


@dataclass
class SMRunResult:
    cycles: int
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    events: list[Any] = field(default_factory=list)
    occupancy: dict[str, int] | None = None


class SM:
    def __init__(self, cfg, sm_id: int = 0, recorder: object | None = None,
                 l2=None, hbm=None):
        # M1 transitional shim: accept DeviceConfig and extract sm + inject cache/hbm
        from gpusim.config.schema import DeviceConfig
        if isinstance(cfg, DeviceConfig):
            sm_cfg = cfg.sm
            sm_cfg._cache_for_run = cfg.cache
            sm_cfg._hbm_for_run = cfg.hbm
            cfg = sm_cfg
        self.cfg = cfg
        self.sm_id = sm_id
        self.recorder = recorder
        self.l2 = l2
        self.hbm = hbm
        self._active_warps: list = []
        self._active_cta_ids: set = set()

    def can_admit_cta(self, occ) -> bool:
        return len(self._active_cta_ids) < occ.active_ctas

    def active_warp_count(self) -> int:
        return sum(1 for w in self._active_warps if not w.finished)

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

        cta_queue: list[tuple[int, tuple[int,int,int]]] = []
        for cz in range(grid[2]):
            for cy in range(grid[1]):
                for cx in range(grid[0]):
                    cid = cx + cy * grid[0] + cz * grid[0] * grid[1]
                    cta_queue.append((cid, (cx, cy, cz)))

        executor = InstrExecutor(kernel=kernel, gmem=gmem, smem=smem,
                                 params=paramspace, cta_id=0,
                                 ctaid=(0,0,0), nctaid=grid, ntid=block)

        from gpusim.core.cache.l1 import L1Cache
        from gpusim.core.cache.l2 import L2Cache
        from gpusim.core.hbm import HBM
        from gpusim.config.schema import CacheConfig, HBMConfig
        # Lazy-construct L2/HBM if not externally injected
        if self.l2 is None or self.hbm is None:
            # M1 transitional shim: cache/hbm injected via transient attrs by api.py
            cache_cfg = (getattr(self.cfg, "_cache_for_run", None)
                          or getattr(self.cfg, "cache", None)
                          or CacheConfig())
            hbm_cfg = (getattr(self.cfg, "_hbm_for_run", None)
                        or getattr(self.cfg, "hbm", None)
                        or HBMConfig())
            self.hbm = HBM(hbm_cfg, recorder=self.recorder)
            self.l2 = L2Cache(cache_cfg, self.hbm, recorder=self.recorder)
        # Use injected/lazy refs
        hbm = self.hbm
        l2 = self.l2
        l1 = L1Cache(l2.cfg, l2, recorder=self.recorder)

        # Per-warp-group state for wgmma
        from gpusim.core.tensor_core.wgmma import WgmmaQueue
        wgmma_queues: dict[int, WgmmaQueue] = {}

        from gpusim.core.mbarrier import MbarrierPool
        from gpusim.core.tma import TensorDescriptorPool
        mbarrier_pools: dict[int, MbarrierPool] = {}   # cta_id -> pool
        tma_descriptor_pool = TensorDescriptorPool()    # per-SM (shared across CTAs)

        sub_cores: list[SubCore] = [
            SubCore(i, self.cfg, executor, [], recorder=self.recorder, l1=l1,
                    wgmma_queues=wgmma_queues, smem=smem,
                    mbarrier_pools=mbarrier_pools,
                    tma_descriptor_pool=tma_descriptor_pool,
                    hbm=hbm)
            for i in range(self.cfg.sub_cores)
        ]

        active_warps: list[Warp] = []
        cycle = 0
        cta_pointer = 0

        def _activate_next_cta() -> bool:
            nonlocal cta_pointer
            if cta_pointer >= len(cta_queue): return False
            current_ctas = len({w.cta_id for w in active_warps})
            if current_ctas >= occ.active_ctas: return False
            cid, ctaid_xyz = cta_queue[cta_pointer]
            # Allocate at least smem_per_sm_bytes so shared-mem ops always have
            # a valid backing buffer, even when smem_per_cta is 0.
            alloc_bytes = smem_per_cta if smem_per_cta > 0 else self.cfg.smem_per_sm_bytes
            smem.allocate_cta(cid, alloc_bytes)
            mbarrier_pools[cid] = MbarrierPool()
            cta_executor = InstrExecutor(kernel=kernel, gmem=gmem, smem=smem,
                                         params=paramspace, cta_id=cid,
                                         ctaid=ctaid_xyz, nctaid=grid, ntid=block)
            for wid_in_cta in range(warps_per_cta):
                fn = WarpFnState(warp_size=32, tids=tuple(range(wid_in_cta*32, wid_in_cta*32+32)))
                w = Warp(warp_id=cid * warps_per_cta + wid_in_cta, kernel=kernel,
                         fn_state=fn, stack=SIMTStack(warp_size=32, entry_pc=0),
                         cta_id=cid, executor=cta_executor)
                active_warps.append(w)
                sub_cores[w.warp_id % self.cfg.sub_cores].warps.append(w)
            if self.recorder is not None:
                self.recorder.cta_launch(cycle=cycle, cta_id=cid,
                                         warps=warps_per_cta,
                                         regs=regs_per_thread * threads_per_cta,
                                         smem_bytes=smem_per_cta)
            cta_pointer += 1
            return True

        while _activate_next_cta(): pass

        while True:
            for sc in sub_cores:
                sc.step(now=cycle)
            l1.install_completed_lines(now=cycle)

            for cta_id, pool in mbarrier_pools.items():
                flipped = pool.tick(now=cycle)
                if self.recorder is not None:
                    for addr, new_phase in flipped:
                        self.recorder.mbarrier(
                            kind="FLIP", cycle=cycle, cta_id=cta_id,
                            smem_addr=addr,
                            expected=pool._barriers[addr].expected_count,
                            arrived=0, phase=new_phase,
                        )

            by_cta: dict[int, list[Warp]] = {}
            for w in active_warps:
                by_cta.setdefault(w.cta_id, []).append(w)
            for cid, ws in by_cta.items():
                non_done = [w for w in ws if not w.finished]
                if non_done and all(w.barrier_pc >= 0 for w in non_done):
                    for w in non_done:
                        w.stack.update_top_pc(w.barrier_pc + 1); w.stack.maybe_pop()
                        w.barrier_pc = -1

            # Phase 3: warp-group wgmma sync coordination
            from gpusim.core.tensor_core.wgmma import (
                InflightWgmma, execute_wgmma_for_group,
            )
            from gpusim.core.tensor_core.mma_spec import parse_mma_op
            by_wg: dict[int, list[Warp]] = {}
            for w in active_warps:
                by_wg.setdefault(w.warp_group_id, []).append(w)
            for wg_id, ws in by_wg.items():
                non_done = [w for w in ws if not w.finished]
                if not non_done or len(non_done) != 4:
                    continue
                if (all(w.wgmma_pending_pc >= 0 for w in non_done)
                        and len({w.wgmma_pending_pc for w in non_done}) == 1):
                    # All 4 warps arrived at the same wgmma. Issue.
                    pc = non_done[0].wgmma_pending_pc
                    instr = non_done[0].kernel.instrs[pc]
                    spec = parse_mma_op(instr.op)
                    if spec is not None and spec.is_async:
                        cta_id = non_done[0].cta_id
                        # Resolve A/B descriptors: src[0] is A reg (u64 smem offset),
                        # src[1] is B reg. Thread 0 of warp 0 holds the descriptor.
                        a_desc = instr.src[0]
                        b_desc = instr.src[1]
                        a_base = non_done[0].fn_state.threads[0].get_u64(a_desc.name)
                        b_base = non_done[0].fn_state.threads[0].get_u64(b_desc.name)
                        a_arr = _read_smem_matrix(
                            smem, cta_id, base=a_base,
                            rows=spec.m, cols=spec.k, dtype=spec.dtype_a)
                        b_arr = _read_smem_matrix(
                            smem, cta_id, base=b_base,
                            rows=spec.k, cols=spec.n, dtype=spec.dtype_b)
                        dst_grp = instr.dst[0]
                        c_grp = instr.src[2] if len(instr.src) > 2 else dst_grp
                        execute_wgmma_for_group(
                            spec=spec, warps=[w.fn_state for w in non_done],
                            a_smem_array=a_arr, b_smem_array=b_arr,
                            dst_per_warp=tuple([dst_grp] * 4),
                            c_per_warp=tuple([c_grp] * 4),
                        )
                        # Push InflightWgmma to queue
                        q = wgmma_queues.setdefault(wg_id, WgmmaQueue(
                            capacity=self.cfg.tensor_core.wgmma_queue_capacity))
                        f = InflightWgmma(
                            issued_at=cycle,
                            completion_at=cycle + self.cfg.tensor_core.tc_wgmma_latency,
                            dst_regs=tuple(
                                tuple(r.name for r in dst_grp.regs)
                                for _ in range(4)),
                        )
                        q.try_push(f)
                        # Advance all 4 warps' PCs and reset pending flag
                        for w in non_done:
                            w.stack.update_top_pc(pc + 1); w.stack.maybe_pop()
                            w.wgmma_pending_pc = -1
                        if self.recorder is not None:
                            self.recorder.instr_issue(
                                cycle=cycle, warp_id=non_done[0].warp_id,
                                pc=pc, op=instr.op,
                                src_loc=(instr.src_loc.file, instr.src_loc.line),
                                active_mask=non_done[0].fn_state.active_mask,
                            )
                            self.recorder.wgmma(
                                kind="ISSUE", cycle=cycle,
                                warp_group_id=wg_id, pc=pc,
                                precision=spec.dtype_a.value,
                                shape_m=spec.m, shape_n=spec.n, shape_k=spec.k,
                                accum_dtype=spec.dtype_d.value,
                                completion_at=f.completion_at,
                            )

            # Drain wgmma queues each cycle
            for wg_id, q in wgmma_queues.items():
                q.drain_completed_groups(now=cycle)

            retiring = []
            for cid, ws in by_cta.items():
                if all(w.finished or (w.stack and w.stack.is_done()) for w in ws):
                    retiring.append(cid)
            for cid in retiring:
                if self.recorder is not None:
                    self.recorder.cta_retire(cycle=cycle, cta_id=cid)
                smem.free_cta(cid)
                active_warps = [w for w in active_warps if w.cta_id != cid]
                for sc in sub_cores:
                    sc.warps = [w for w in sc.warps if w.cta_id != cid]
                _activate_next_cta()

            cycle += 1
            if cta_pointer >= len(cta_queue) and not active_warps:
                break
            if cycle > 10_000_000:
                raise RuntimeError("simulation runaway > 1e7 cycles")

        outputs = {n: v for n, v in params.items() if isinstance(v, np.ndarray)}
        return SMRunResult(
            cycles=cycle, outputs=outputs, events=[],
            occupancy={"active_ctas": occ.active_ctas, "bottleneck": occ.bottleneck},
        )
