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
        # Per-SM state for Device-driven mode
        self._sub_cores = None
        self._gmem = None
        self._smem = None
        self._paramspace = None
        self._kernel = None
        self._grid = None
        self._block = None
        self._occupancy = None
        self._executor = None
        self._l1 = None
        self._wgmma_queues: dict = {}
        self._mbarrier_pools: dict = {}
        self._tma_descriptor_pool = None
        self._device_cluster_barriers = {}

    def can_admit_cta(self, occ) -> bool:
        return len(self._active_cta_ids) < occ.active_ctas

    def active_warp_count(self) -> int:
        return sum(1 for w in self._active_warps if not w.finished)

    def set_cluster_barriers(self, cluster_barriers: dict):
        self._device_cluster_barriers = cluster_barriers
        if hasattr(self, "_sub_cores") and self._sub_cores:
            for sc in self._sub_cores:
                sc._device_cluster_barriers = cluster_barriers

    def initialize_for_run(self, kernel, gmem, smem, paramspace, grid, block,
                            occupancy, cluster_size: int = 1):
        """Called once by Device.run before main loop starts."""
        from gpusim.core.exec import InstrExecutor
        from gpusim.core.cache.l1 import L1Cache
        self._kernel = kernel
        self._gmem = gmem
        self._smem = smem
        self._paramspace = paramspace
        self._grid = grid
        self._block = block
        self._occupancy = occupancy
        self._executor = InstrExecutor(
            kernel=kernel, gmem=gmem, smem=smem,
            params=paramspace, cta_id=0,
            ctaid=(0, 0, 0), nctaid=grid, ntid=block,
        )
        # L1 takes config from L2 (which has cache_cfg)
        self._l1 = L1Cache(self.l2.cfg, self.l2, recorder=self.recorder,
                            sm_id=self.sm_id)
        from gpusim.core.tma import TensorDescriptorPool
        self._tma_descriptor_pool = TensorDescriptorPool()
        self._wgmma_queues = {}
        self._bulk_store_queues = {}
        self._mbarrier_pools = {}
        self._cluster_size = cluster_size
        from gpusim.core.sub_core import SubCore
        self._sub_cores = []
        for i in range(self.cfg.sub_cores):
            sc = SubCore(
                i, self.cfg, self._executor, [], recorder=self.recorder,
                l1=self._l1, wgmma_queues=self._wgmma_queues,
                smem=self._smem, mbarrier_pools=self._mbarrier_pools,
                tma_descriptor_pool=self._tma_descriptor_pool,
                hbm=self.hbm,
            )
            self._sub_cores.append(sc)
        # Propagate cluster barrier ref to sub_cores (for SubCore._is_ready)
        for sc in self._sub_cores:
            sc._device_cluster_barriers = self._device_cluster_barriers

    def activate_cta(self, cta_id, ctaid_xyz, regs_per_thread, smem_per_cta,
                      threads_per_cta, warps_per_cta, cycle,
                      *, cluster_id: int = -1, cluster_rank: int = -1):
        """Called by Device when scheduler picks this SM for a CTA."""
        from gpusim.core.exec import WarpFnState, InstrExecutor
        from gpusim.core.simt_stack import SIMTStack
        from gpusim.core.warp import Warp
        from gpusim.core.mbarrier import MbarrierPool
        alloc_bytes = (smem_per_cta if smem_per_cta > 0
                        else self.cfg.smem_per_sm_bytes)
        self._smem.allocate_cta(cta_id, alloc_bytes)
        self._mbarrier_pools[cta_id] = MbarrierPool()
        cta_executor = InstrExecutor(
            kernel=self._kernel, gmem=self._gmem, smem=self._smem,
            params=self._paramspace, cta_id=cta_id,
            ctaid=ctaid_xyz, nctaid=self._grid, ntid=self._block,
        )
        cta_executor.cluster_id = cluster_id
        cta_executor.cluster_rank = cluster_rank
        cta_executor.cluster_size = getattr(self, "_cluster_size", 1)
        for wid_in_cta in range(warps_per_cta):
            fn = WarpFnState(warp_size=32,
                              tids=tuple(range(wid_in_cta * 32,
                                                wid_in_cta * 32 + 32)))
            warp_id = cta_id * warps_per_cta + wid_in_cta
            w = Warp(warp_id=warp_id, kernel=self._kernel, fn_state=fn,
                      stack=SIMTStack(warp_size=32, entry_pc=0),
                      cta_id=cta_id, executor=cta_executor,
                      cluster_id=cluster_id, cluster_rank=cluster_rank)
            self._active_warps.append(w)
            self._sub_cores[warp_id % self.cfg.sub_cores].warps.append(w)
        self._active_cta_ids.add(cta_id)
        if self.recorder is not None:
            self.recorder.cta_launch(
                cycle=cycle, cta_id=cta_id, warps=warps_per_cta,
                regs=regs_per_thread * threads_per_cta,
                smem_bytes=smem_per_cta,
            )

    def step_cycle(self, cycle: int) -> list[int]:
        """Advance one cycle. Returns list of cta_ids that retired this cycle."""
        for sc in self._sub_cores:
            sc.step(now=cycle)
        self._l1.install_completed_lines(now=cycle)

        # mbarrier tick
        for cta_id, pool in self._mbarrier_pools.items():
            flipped = pool.tick(now=cycle)
            if self.recorder is not None:
                for addr, new_phase in flipped:
                    self.recorder.mbarrier(
                        kind="FLIP", cycle=cycle, cta_id=cta_id,
                        smem_addr=addr,
                        expected=pool._barriers[addr].expected_count,
                        arrived=0, phase=new_phase,
                    )

        # CTA barrier release coordination
        by_cta: dict = {}
        for w in self._active_warps:
            by_cta.setdefault(w.cta_id, []).append(w)
        for cid, ws in by_cta.items():
            non_done = [w for w in ws if not w.finished]
            if non_done and all(w.barrier_pc >= 0 for w in non_done):
                instr = non_done[0].kernel.instrs[non_done[0].barrier_pc]
                if instr.op == "barrier.cluster.arrive":
                    cluster_id = non_done[0].cluster_id
                    rank = non_done[0].cluster_rank
                    pool = self._device_cluster_barriers.get(cluster_id)
                    if pool is not None:
                        pool.arrive(rank)
                    if (self.recorder is not None
                            and hasattr(self.recorder, "cluster_barrier")):
                        self.recorder.cluster_barrier(
                            kind="ARRIVE", cycle=cycle,
                            cluster_id=cluster_id, cta_id=cid,
                            rank=rank, sm_id=self.sm_id,
                            arrived_count=bin(pool.arrived_mask).count("1") if pool else 0,
                        )
                    for w in non_done:
                        w.stack.update_top_pc(w.barrier_pc + 1); w.stack.maybe_pop()
                        w.barrier_pc = -1
                else:
                    # bar.sync existing path
                    for w in non_done:
                        w.stack.update_top_pc(w.barrier_pc + 1); w.stack.maybe_pop()
                        w.barrier_pc = -1

        # Phase 5: check cluster barrier waits
        for w in self._active_warps:
            if w.cluster_barrier_wait_pc >= 0:
                pool = self._device_cluster_barriers.get(w.cluster_id)
                if pool is None:
                    continue
                if pool.is_released(w.cluster_barrier_phase_at_wait):
                    w.stack.update_top_pc(w.cluster_barrier_wait_pc + 1)
                    w.stack.maybe_pop()
                    w.cluster_barrier_wait_pc = -1
                    if (self.recorder is not None
                            and hasattr(self.recorder, "cluster_barrier")):
                        self.recorder.cluster_barrier(
                            kind="WAIT_RELEASE", cycle=cycle,
                            cluster_id=w.cluster_id,
                            cta_id=w.cta_id, rank=w.cluster_rank,
                            sm_id=self.sm_id,
                        )

        # warp-group wgmma sync coordination
        self._wgmma_coordinate(cycle)
        self._bulk_store_coordinate(cycle)

        # wgmma queue drain
        for q in self._wgmma_queues.values():
            q.drain_completed_groups(now=cycle)
        # bulk store queue drain
        for q in getattr(self, "_bulk_store_queues", {}).values():
            q.drain_completed_groups(now=cycle)

        # CTA retirement
        retiring = []
        for cid, ws in by_cta.items():
            if all(w.finished or (w.stack and w.stack.is_done()) for w in ws):
                retiring.append(cid)
        for cid in retiring:
            if self.recorder is not None:
                self.recorder.cta_retire(cycle=cycle, cta_id=cid)
            self._smem.free_cta(cid)
            self._active_warps = [w for w in self._active_warps if w.cta_id != cid]
            for sc in self._sub_cores:
                sc.warps = [w for w in sc.warps if w.cta_id != cid]
            self._active_cta_ids.discard(cid)
        return retiring

    def has_active_warps(self) -> bool:
        return any(not w.finished for w in self._active_warps)

    def _wgmma_coordinate(self, cycle):
        """Warp-group wgmma sync coordination (extracted from old SM.run inner block)."""
        from gpusim.core.tensor_core.wgmma import (
            InflightWgmma, WgmmaQueue, execute_wgmma_for_group,
        )
        from gpusim.core.tensor_core.mma_spec import parse_mma_op
        by_wg: dict[int, list] = {}
        for w in self._active_warps:
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
                if spec is None or not spec.is_async:
                    continue
                cta_id = non_done[0].cta_id
                # Resolve A/B descriptors: src[0] is A reg (u64 smem offset),
                # src[1] is B reg. Thread 0 of warp 0 holds the descriptor.
                a_desc = instr.src[0]
                b_desc = instr.src[1]
                a_base = non_done[0].fn_state.threads[0].get_u64(a_desc.name)
                b_base = non_done[0].fn_state.threads[0].get_u64(b_desc.name)
                a_arr = _read_smem_matrix(
                    self._smem, cta_id, base=a_base,
                    rows=spec.m, cols=spec.k, dtype=spec.dtype_a)
                b_arr = _read_smem_matrix(
                    self._smem, cta_id, base=b_base,
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
                q = self._wgmma_queues.setdefault(wg_id, WgmmaQueue(
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

    def _bulk_store_coordinate(self, cycle):
        """Warp-group bulk-store coordination: issue when all 4 warps have pending store."""
        from gpusim.core.tma_store import (
            BulkStoreQueue, InflightBulkStore, do_bulk_store_2d,
        )
        if not hasattr(self, "_bulk_store_queues"):
            self._bulk_store_queues = {}
        # Share queue dict with sub_cores so they see the same queue objects.
        # Always overwrite so sub_cores that created their own {} get merged.
        for sc in self._sub_cores:
            if (not hasattr(sc, "bulk_store_queues")
                    or sc.bulk_store_queues is None
                    or sc.bulk_store_queues is not self._bulk_store_queues):
                # Merge any entries the sub_core created before we synced
                if hasattr(sc, "bulk_store_queues") and sc.bulk_store_queues:
                    self._bulk_store_queues.update(sc.bulk_store_queues)
                sc.bulk_store_queues = self._bulk_store_queues
        # Gather warps with a pending bulk-store PC
        pending: list = [w for w in self._active_warps
                         if not w.finished and w.bulk_store_pending_pc >= 0]
        # Group by (warp_group_id, pc) so we handle each unique store once
        by_wg_pc: dict = {}
        for w in pending:
            key = (w.warp_group_id, w.bulk_store_pending_pc)
            by_wg_pc.setdefault(key, []).append(w)
        for (wg_id, pc), ws in by_wg_pc.items():
            instr = ws[0].kernel.instrs[pc]
            desc_reg = instr.src[0]
            smem_src_reg = instr.src[1]
            handle = ws[0].fn_state.threads[0].get_u64(desc_reg.name)
            smem_src = ws[0].fn_state.threads[0].get_u64(smem_src_reg.name)
            desc = self._tma_descriptor_pool.lookup(handle)
            tx_bytes = do_bulk_store_2d(
                gmem=self._gmem, smem=self._smem,
                cta_id=ws[0].cta_id, smem_src=smem_src, desc=desc,
            )
            n_lines = (tx_bytes + 127) // 128
            latency_per_line = self.cfg.tensor_core.bulk_store_latency_per_line
            latency = max(8, n_lines * latency_per_line)
            completion_at = cycle + latency
            cap = self.cfg.tensor_core.bulk_store_queue_capacity
            q = self._bulk_store_queues.setdefault(
                wg_id, BulkStoreQueue(capacity=cap))
            f = InflightBulkStore(
                issued_at=cycle, completion_at=completion_at,
                bytes_total=tx_bytes,
            )
            q.try_push(f)
            for w in ws:
                w.stack.update_top_pc(pc + 1); w.stack.maybe_pop()
                w.bulk_store_pending_pc = -1
            if self.recorder is not None:
                self.recorder.instr_issue(
                    cycle=cycle, warp_id=ws[0].warp_id,
                    pc=pc, op=instr.op,
                    src_loc=(instr.src_loc.file, instr.src_loc.line),
                    active_mask=ws[0].fn_state.active_mask,
                )
                self.recorder.bulk_store(
                    kind="ISSUE", cycle=cycle,
                    warp_group_id=wg_id, sm_id=self.sm_id, pc=pc,
                    smem_src=smem_src, gmem_base=desc.gmem_base,
                    bytes_total=tx_bytes,
                    completion_at=completion_at,
                )

    def run(self, kernel, grid, block, params, regs_per_thread: int = 16,
             smem_per_cta: int = 0) -> SMRunResult:
        from gpusim.core.exec import GlobalMemory, SharedMemory, ParamSpace
        from gpusim.core.occupancy import compute_occupancy

        # Lazy-construct L2/HBM if not externally injected
        if self.l2 is None or self.hbm is None:
            cache_cfg = (getattr(self.cfg, "_cache_for_run", None)
                          or getattr(self.cfg, "cache", None))
            hbm_cfg = (getattr(self.cfg, "_hbm_for_run", None)
                        or getattr(self.cfg, "hbm", None))
            if cache_cfg is None:
                from gpusim.config.schema import CacheConfig
                cache_cfg = CacheConfig()
            if hbm_cfg is None:
                from gpusim.config.schema import HBMConfig
                hbm_cfg = HBMConfig()
            from gpusim.core.hbm import HBM
            from gpusim.core.cache.l2 import L2Cache
            self.hbm = HBM(hbm_cfg, recorder=self.recorder)
            self.l2 = L2Cache(cache_cfg, self.hbm, recorder=self.recorder)

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
        occ = compute_occupancy(self.cfg, threads_per_cta, regs_per_thread,
                                  smem_per_cta)
        self.initialize_for_run(kernel, gmem, smem, paramspace, grid, block, occ)

        cta_queue: list = []
        for cz in range(grid[2]):
            for cy in range(grid[1]):
                for cx in range(grid[0]):
                    cid = cx + cy * grid[0] + cz * grid[0] * grid[1]
                    cta_queue.append((cid, (cx, cy, cz)))
        cta_pointer = 0
        cycle = 0

        def _try_dispatch():
            nonlocal cta_pointer
            while cta_pointer < len(cta_queue) and self.can_admit_cta(occ):
                cid, ctaid_xyz = cta_queue[cta_pointer]
                self.activate_cta(cid, ctaid_xyz, regs_per_thread, smem_per_cta,
                                   threads_per_cta, warps_per_cta, cycle)
                cta_pointer += 1
        _try_dispatch()

        while True:
            self.step_cycle(cycle)
            self.l2.tick(now=cycle)
            _try_dispatch()
            cycle += 1
            if cta_pointer >= len(cta_queue) and not self.has_active_warps():
                break
            if cycle > 10_000_000:
                raise RuntimeError("simulation runaway > 1e7 cycles")

        outputs = {n: v for n, v in params.items() if isinstance(v, np.ndarray)}
        return SMRunResult(
            cycles=cycle, outputs=outputs, events=[],
            occupancy={"active_ctas": occ.active_ctas, "bottleneck": occ.bottleneck},
        )
