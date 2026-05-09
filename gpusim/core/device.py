from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from gpusim.config.schema import DeviceConfig


@dataclass
class DeviceRunResult:
    cycles: int
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    occupancy: dict[str, int] | None = None


class Device:
    def __init__(self, cfg: DeviceConfig, recorder=None):
        self.cfg = cfg
        self.n_sm = cfg.n_sm
        self.recorder = recorder

    def run(self, kernel, grid, block, params,
             regs_per_thread: int = 16, smem_per_cta: int = 0,
             stream_id: int = 0, kernel_name: str = "<unnamed>") -> DeviceRunResult:
        import numpy as np
        from gpusim.core.sm import SM
        from gpusim.core.exec import GlobalMemory, SharedMemory, ParamSpace
        from gpusim.core.cache.l2 import L2Cache
        from gpusim.core.hbm import HBM
        from gpusim.core.occupancy import compute_occupancy
        from gpusim.core.scheduler import make_cta_scheduler

        gmem = GlobalMemory()
        # SharedMemory pool sized for all CTAs across all SMs
        # (each CTA gets its own region keyed by cta_id, so capacity is just
        # max simultaneous CTAs * smem_per_cta)
        smem = SharedMemory(size_bytes=self.cfg.sm.smem_per_sm_bytes
                                          * max(self.n_sm, 1))
        p_dict: dict[str, int] = {}
        for name, val in params.items():
            if isinstance(val, np.ndarray):
                p_dict[name] = gmem.bind(name, val)
            else:
                p_dict[name] = int(val)
        paramspace = ParamSpace(p_dict)
        threads_per_cta = block[0] * block[1] * block[2]
        warps_per_cta = (threads_per_cta + 31) // 32
        occ = compute_occupancy(self.cfg.sm, threads_per_cta,
                                  regs_per_thread, smem_per_cta)

        hbm = HBM(self.cfg.hbm, recorder=self.recorder)
        l2 = L2Cache(self.cfg.cache, hbm, recorder=self.recorder)

        # Phase 8 M4: apply any pre-registered stream L2 windows
        stream_windows = getattr(self, "_stream_windows", {})
        if stream_id in stream_windows:
            start_set, n_sets = stream_windows[stream_id]
            l2.register_stream_window(stream_id, start_set, n_sets)

        sms = []
        for i in range(self.n_sm):
            sm = SM(self.cfg.sm, sm_id=i, recorder=self.recorder, l2=l2, hbm=hbm)
            sm.initialize_for_run(kernel, gmem, smem, paramspace, grid, block, occ,
                                    cluster_size=self.cfg.cluster_size)
            sms.append(sm)

        cta_queue = []
        for cz in range(grid[2]):
            for cy in range(grid[1]):
                for cx in range(grid[0]):
                    cid = cx + cy * grid[0] + cz * grid[0] * grid[1]
                    cta_queue.append((cid, (cx, cy, cz)))
        cluster_size = self.cfg.cluster_size
        grid_size = grid[0] * grid[1] * grid[2]
        if cluster_size > 1 and grid_size % cluster_size != 0:
            raise ValueError(
                f"cluster_size ({cluster_size}) must divide grid_size ({grid_size})"
            )

        scheduler = make_cta_scheduler(self.cfg.scheduler.cta_policy)
        cycle = 0
        cta_pointer = 0

        from gpusim.core.cluster import ClusterBarrierPool
        cluster_barriers: dict[int, ClusterBarrierPool] = {}
        for sm in sms:
            sm.set_cluster_barriers(cluster_barriers)

        def _try_dispatch():
            nonlocal cta_pointer
            while cta_pointer < len(cta_queue):
                target_sms = scheduler.peek(sms, occ, k=cluster_size)
                if target_sms is None:
                    return
                scheduler.commit(k=cluster_size)
                cluster_id = cta_pointer // cluster_size
                if cluster_size > 1:
                    cluster_barriers[cluster_id] = ClusterBarrierPool(
                        expected=cluster_size,
                    )
                for i, sm in enumerate(target_sms):
                    cid, ctaid_xyz = cta_queue[cta_pointer + i]
                    sm.activate_cta(
                        cid, ctaid_xyz, regs_per_thread, smem_per_cta,
                        threads_per_cta, warps_per_cta, cycle,
                        cluster_id=cluster_id if cluster_size > 1 else -1,
                        cluster_rank=i if cluster_size > 1 else -1,
                        stream_id=stream_id,
                    )
                    if self.recorder is not None:
                        self.recorder.cta_dispatch(
                            cycle=cycle, cta_id=cid, sm_id=sm.sm_id,
                            queue_position=cta_pointer + i,
                            active_warps_at_dispatch=sm.active_warp_count(),
                            stream_id=stream_id,
                        )
                if cluster_size > 1 and self.recorder is not None:
                    self.recorder.cluster_dispatch(
                        cycle=cycle, cluster_id=cluster_id,
                        cluster_size=cluster_size,
                        sm_ids=tuple(sm.sm_id for sm in target_sms),
                        cta_ids=tuple(cta_queue[cta_pointer + i][0]
                                        for i in range(cluster_size)),
                        queue_position=cluster_id,
                    )
                cta_pointer += cluster_size
        _try_dispatch()

        while True:
            for sm in sms:
                sm.step_cycle(cycle)
            l2.tick(now=cycle)
            _try_dispatch()
            cycle += 1
            if (cta_pointer >= len(cta_queue)
                  and not any(sm.has_active_warps() for sm in sms)):
                break
            if cycle > 10_000_000:
                raise RuntimeError("simulation runaway > 1e7 cycles")

        outputs = {n: v for n, v in params.items() if isinstance(v, np.ndarray)}
        return DeviceRunResult(
            cycles=cycle, outputs=outputs,
            occupancy={"active_ctas": occ.active_ctas, "bottleneck": occ.bottleneck},
        )

    def _available_sms(self) -> list:
        """SMs with capacity for at least one more CTA. Phase 9."""
        out = []
        for sm in getattr(self, "sms", []):
            cap_fn = getattr(sm, "remaining_cta_capacity", None)
            if cap_fn is None:
                # Fallback: assume always has capacity
                out.append(sm)
                continue
            if cap_fn() > 0:
                out.append(sm)
        return out

    def _dispatch_cta_to_sm(self, sm, stream, cta_idx, cycle: int) -> None:
        """Dispatch one CTA from stream to sm at the given cycle. Phase 9."""
        if hasattr(sm, "activate_cta"):
            sm.activate_cta(cta_idx, stream_id=stream.stream_id)
        elif hasattr(sm, "dispatch_cta"):
            sm.dispatch_cta(cta_idx, stream_id=stream.stream_id)

    def _stream_grid_retired(self, stream) -> bool:
        """True if all CTAs of stream's inflight grid have completed. Phase 9."""
        return stream.in_flight_ctas == 0 and stream.inflight is not None

    def run_streams(self, streams: list) -> "MultiStreamResult":
        """Multi-stream run loop. Each stream's launches are processed in FIFO order;
        across streams CTAs are interleaved by the MultiStreamScheduler (RR).

        Phase 7 simplification: re-uses Device.run() per-grid for retire,
        but coordinates across streams at the scheduler level.
        """
        from gpusim.core.scheduler import ConcurrentStreamScheduler
        from gpusim.api import MultiStreamResult, Result
        from gpusim.frontend.parser import parse
        from gpusim.trace.recorder import Recorder

        weights = getattr(getattr(self.cfg, "scheduler", None), "priority_weights", None)
        sched = ConcurrentStreamScheduler(streams, priority_weights=weights)
        results_per_stream = {s.stream_id: [] for s in streams}

        # Phase 8 M4: register per-stream L2 windows so run() can apply them
        self._stream_windows = {}
        for s in streams:
            if s.l2_window is not None:
                self._stream_windows[s.stream_id] = s.l2_window

        while not all(s.is_idle() for s in streams):
            for s in streams:
                if not s.is_idle() and s.pending:
                    sched._ensure_inflight(s)
                if s.inflight is not None:
                    g = s.inflight
                    kernel = parse(g.ptx_src, "<inline>")
                    # Create a per-launch recorder so events are captured
                    per_launch_recorder = Recorder()
                    saved_recorder = self.recorder
                    self.recorder = per_launch_recorder
                    dev_res = self.run(
                        kernel=kernel,
                        grid=g.grid,
                        block=g.block,
                        params=g.params,
                        stream_id=s.stream_id,
                        kernel_name=g.kernel_name,
                    )
                    self.recorder = saved_recorder
                    result = Result(
                        outputs=dev_res.outputs,
                        mode="timing",
                        metrics={"cycles": dev_res.cycles,
                                 "occupancy": dev_res.occupancy},
                        _recorder=per_launch_recorder,
                        _occupancy=dev_res.occupancy,
                        _kernel_name=kernel.name,
                        _grid=g.grid,
                        _block=g.block,
                        stream_id=s.stream_id,
                        kernel_name=g.kernel_name,
                    )
                    results_per_stream[s.stream_id].append(result)
                    sched.mark_grid_retired(s)
                    # Phase 8 M3: signal pending record markers in this stream
                    if hasattr(s, "_pending_record_markers") and s._pending_record_markers:
                        end_cycle = result.metrics.get("cycles", 0)
                        for marker in s._pending_record_markers:
                            marker.event.signaled_at_cycle = end_cycle
                        s._pending_record_markers = []

        total_cycles = max((r.metrics.get("cycles", 0)
                              for results in results_per_stream.values()
                              for r in results), default=0)
        return MultiStreamResult(
            streams=results_per_stream,
            total_cycles=total_cycles,
            _recorder=getattr(self, "recorder", None),
            _stream_refs=list(streams),     # NEW
        )
