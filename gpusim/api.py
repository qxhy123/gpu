from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from gpusim.core.exec import functional_run
from gpusim.trace.recorder import Recorder
from gpusim.viz.html_report import save_html
from gpusim.viz.perfetto import save_perfetto
from gpusim.viz.notebook import warp_state_dataframe, stall_dataframe, warp_timeline_figure


@dataclass
class Result:
    outputs: dict[str, np.ndarray]
    mode: str
    metrics: dict[str, Any]
    _recorder: Recorder | None = field(default=None, repr=False)
    _kernel_name: str = field(default="", repr=False)
    _grid: tuple = field(default=(1,1,1), repr=False)
    _block: tuple = field(default=(1,1,1), repr=False)
    _occupancy: dict | None = field(default=None, repr=False)
    stream_id: int = 0    # NEW Phase 7 — single-kernel path uses default 0

    def summary(self) -> str:
        cyc = self.metrics.get("cycles", "?")
        bn = (self._occupancy or {}).get("bottleneck", "?")
        cache_part = ""
        if self._recorder is not None:
            try:
                cache_part = " | " + self.cache_summary()
            except Exception:
                pass
        return f"gpusim {self.mode}: {cyc} cycles, bottleneck={bn}{cache_part}"

    @property
    def events_df(self):
        return warp_state_dataframe(self._recorder) if self._recorder else None

    @property
    def stall_df(self):
        return stall_dataframe(self._recorder) if self._recorder else None

    def timeline(self, warp: int):
        return warp_timeline_figure(self._recorder, warp) if self._recorder else None

    @property
    def l1_events_df(self):
        from gpusim.viz.notebook import l1_events_dataframe
        return l1_events_dataframe(self._recorder) if self._recorder else None

    @property
    def l2_events_df(self):
        from gpusim.viz.notebook import l2_events_dataframe
        return l2_events_dataframe(self._recorder) if self._recorder else None

    @property
    def hbm_events_df(self):
        from gpusim.viz.notebook import hbm_events_dataframe
        return hbm_events_dataframe(self._recorder) if self._recorder else None

    @property
    def mma_events_df(self):
        from gpusim.viz.notebook import mma_events_dataframe
        return mma_events_dataframe(self._recorder) if self._recorder else None

    @property
    def wgmma_events_df(self):
        from gpusim.viz.notebook import wgmma_events_dataframe
        return wgmma_events_dataframe(self._recorder) if self._recorder else None

    @property
    def tma_events_df(self):
        from gpusim.viz.notebook import tma_events_dataframe
        return tma_events_dataframe(self._recorder) if self._recorder else None

    @property
    def mbarrier_events_df(self):
        from gpusim.viz.notebook import mbarrier_events_dataframe
        return mbarrier_events_dataframe(self._recorder) if self._recorder else None

    @property
    def tc_metrics(self) -> dict:
        if self._recorder is None:
            return {}
        from gpusim.analysis.metrics import (
            tc_utilization, precision_distribution, effective_tflops,
            async_overlap_ratio, mbarrier_wait_distribution,
            wgmma_queue_pressure, tma_bandwidth_utilization,
        )
        cycles = self.metrics.get("cycles", 1)
        mma = self.mma_events_df
        wgmma = self.wgmma_events_df
        tma = self.tma_events_df
        mbar = self.mbarrier_events_df
        warp_state = self.events_df
        return {
            "tc_utilization":     tc_utilization(mma, wgmma, cycles).to_dict() if mma is not None else {},
            "precision_dist":     precision_distribution(mma, wgmma).to_dict() if mma is not None else {},
            "effective_tflops":   effective_tflops(mma, wgmma, cycles, freq_ghz=1.0) if mma is not None else {},
            "async_overlap":      async_overlap_ratio(wgmma, warp_state) if wgmma is not None else 0.0,
            "wait_distribution":  mbarrier_wait_distribution(wgmma, mbar).to_dict() if wgmma is not None else {},
            "queue_pressure":     wgmma_queue_pressure(wgmma, cycles).to_dict() if wgmma is not None else {},
            "tma_bw_util":        tma_bandwidth_utilization(tma, cycles, total_hbm_bw=512.0) if tma is not None else 0.0,
        }

    def tc_summary(self) -> str:
        m = self.tc_metrics
        if not m:
            return "no recorder"
        flops = m.get("effective_tflops", {})
        flops_str = ", ".join(f"{k}: {v:.2f}" for k, v in flops.items())
        return f"TFLOPS [{flops_str}] | async_overlap={m.get('async_overlap', 0):.2f}"

    @property
    def cache_metrics(self) -> dict:
        if self._recorder is None:
            return {}
        from gpusim.analysis.metrics import (
            l1_hit_rate, l2_hit_rate, mshr_merge_rate,
            cache_hierarchy_breakdown,
            channel_utilization, row_buffer_hit_rate, wb_traffic_fraction,
        )
        l1 = self.l1_events_df
        l2 = self.l2_events_df
        hbm = self.hbm_events_df
        cycles = self.metrics.get("cycles", 1)
        return {
            "l1_hit_rate":     l1_hit_rate(l1)            if l1 is not None else 0.0,
            "l2_hit_rate":     l2_hit_rate(l2)            if l2 is not None else 0.0,
            "mshr_merge_rate": mshr_merge_rate(l1)        if l1 is not None else 0.0,
            "hierarchy":       cache_hierarchy_breakdown(l1, l2) if l1 is not None else {},
            "channel_util":    channel_utilization(hbm, cycles).tolist() if hbm is not None else [],
            "row_buffer_hit_rate": row_buffer_hit_rate(hbm) if hbm is not None else 0.0,
            "wb_traffic_fraction": wb_traffic_fraction(hbm) if hbm is not None else 0.0,
        }

    @property
    def bandwidth_df(self):
        from gpusim.analysis.metrics import bandwidth_per_channel
        if self._recorder is None:
            return None
        return bandwidth_per_channel(self.hbm_events_df,
                                       self.metrics.get("cycles", 1))

    def cache_summary(self) -> str:
        cm = self.cache_metrics
        if not cm:
            return "no recorder"
        return (f"L1 hit {cm['l1_hit_rate']*100:.1f}% / "
                f"L2 hit {cm['l2_hit_rate']*100:.1f}% / "
                f"MSHR merge {cm['mshr_merge_rate']*100:.1f}% / "
                f"row buffer hit {cm['row_buffer_hit_rate']*100:.1f}%")

    @property
    def cta_dispatch_events_df(self):
        from gpusim.viz.notebook import cta_dispatch_events_dataframe
        return cta_dispatch_events_dataframe(self._recorder) if self._recorder else None

    @property
    def l2_mshr_events_df(self):
        from gpusim.viz.notebook import l2_mshr_events_dataframe
        return l2_mshr_events_dataframe(self._recorder) if self._recorder else None

    @property
    def bulk_store_events_df(self):
        from gpusim.viz.notebook import bulk_store_events_dataframe
        return bulk_store_events_dataframe(self._recorder) if self._recorder else None

    @property
    def instr_issue_events_df(self):
        from gpusim.viz.notebook import instr_issue_dataframe
        return instr_issue_dataframe(self._recorder) if self._recorder else None

    @property
    def device_metrics(self) -> dict:
        if self._recorder is None:
            return {}
        from gpusim.analysis.metrics import (
            per_sm_utilization, cta_to_sm_mapping, cta_dispatch_latency,
            l2_cross_sm_hit_rate, l2_mshr_pressure, bulk_store_async_overlap_ratio,
        )
        cycles = self.metrics.get("cycles", 1)
        warp_state = self.events_df
        l2_events = self.l2_events_df
        l2_mshr = self.l2_mshr_events_df
        dispatch = self.cta_dispatch_events_df
        bulk = self.bulk_store_events_df
        if dispatch is not None and not dispatch.empty:
            n_sm = int(dispatch["sm_id"].max()) + 1
        else:
            n_sm = 1
        return {
            "per_sm_utilization": per_sm_utilization(warp_state, cycles, n_sm).to_dict() if warp_state is not None else {},
            "cta_to_sm_mapping": cta_to_sm_mapping(dispatch).to_dict() if dispatch is not None else {},
            "l2_cross_sm_hit_rate": l2_cross_sm_hit_rate(l2_events) if l2_events is not None else 0.0,
            "l2_mshr_pressure_peak": int(l2_mshr_pressure(l2_mshr, cycles).max()) if l2_mshr is not None and not l2_mshr.empty else 0,
            "bulk_store_async_overlap": bulk_store_async_overlap_ratio(bulk, warp_state) if bulk is not None else 0.0,
        }

    def device_summary(self) -> str:
        m = self.device_metrics
        if not m:
            return "no recorder"
        rate = m.get("l2_cross_sm_hit_rate", 0)
        peak = m.get("l2_mshr_pressure_peak", 0)
        overlap = m.get("bulk_store_async_overlap", 0)
        return (f"L2 cross-SM hit {rate*100:.1f}% / "
                 f"L2 MSHR peak {peak} / "
                 f"BulkStore overlap {overlap:.2f}")

    @property
    def cluster_dispatch_events_df(self):
        from gpusim.viz.notebook import cluster_dispatch_events_dataframe
        return cluster_dispatch_events_dataframe(self._recorder) if self._recorder else None

    @property
    def cluster_barrier_events_df(self):
        from gpusim.viz.notebook import cluster_barrier_events_dataframe
        return cluster_barrier_events_dataframe(self._recorder) if self._recorder else None

    @property
    def cluster_metrics(self) -> dict:
        if self._recorder is None:
            return {}
        from gpusim.analysis.metrics import (
            cluster_dispatch_latency, cluster_barrier_wait_distribution,
            dsmem_remote_access_rate,
        )
        cd = self.cluster_dispatch_events_df
        cb = self.cluster_barrier_events_df
        try:
            from gpusim.viz.notebook import instr_issue_dataframe
            ii = instr_issue_dataframe(self._recorder)
        except Exception:
            ii = None
        return {
            "cluster_count": len(cd) if cd is not None else 0,
            "avg_barrier_wait": float(
                cluster_barrier_wait_distribution(cb).mean()
            ) if cb is not None and not cb.empty else 0.0,
            "dsmem_remote_rate": dsmem_remote_access_rate(ii) if ii is not None else 0.0,
        }

    def cluster_summary(self) -> str:
        m = self.cluster_metrics
        if not m or m.get("cluster_count", 0) == 0:
            return "no clusters dispatched"
        return (f"clusters dispatched={m['cluster_count']} / "
                 f"avg barrier wait={m['avg_barrier_wait']:.1f} cyc / "
                 f"dsmem remote rate={m['dsmem_remote_rate']*100:.1f}%")

    @property
    def atomic_events_df(self):
        from gpusim.viz.notebook import atomic_events_dataframe
        return atomic_events_dataframe(self._recorder) if self._recorder else None

    @property
    def atomic_metrics(self) -> dict:
        if self._recorder is None:
            return {}
        from gpusim.analysis.metrics import (
            atomic_throughput_per_line, atomic_serialization_overhead,
            atom_vs_red_ratio, cooperative_epilogue_overlap,
        )
        cycles = self.metrics.get("cycles", 1)
        atomic_df = self.atomic_events_df
        if atomic_df is None or atomic_df.empty:
            return {"count": 0}
        per_line = atomic_throughput_per_line(atomic_df, cycles)
        peak_depth = int(per_line["atomic_count"].max()) if not per_line.empty else 0
        return {
            "count": len(atomic_df),
            "peak_queue_depth": peak_depth,
            "serialization_overhead": atomic_serialization_overhead(atomic_df, cycles),
            "atom_red_ratio": atom_vs_red_ratio(atomic_df),
            "cooperative_overlap": cooperative_epilogue_overlap(
                self.bulk_store_events_df, self.mma_events_df,
            ),
        }

    def atomic_summary(self) -> str:
        m = self.atomic_metrics
        if not m or m.get("count", 0) == 0:
            return "no atomic ops"
        return (f"atomic count={m['count']} / "
                 f"hot line peak depth={m['peak_queue_depth']} / "
                 f"serial overhead={m['serialization_overhead']*100:.1f}%")

    def html_report(self, path):
        if self._recorder is None:
            raise ValueError("no recorder; run in timing mode")
        save_html(self._recorder, path,
                  kernel_name=self._kernel_name, grid=self._grid, block=self._block,
                  cycles=self.metrics.get("cycles", 0),
                  occupancy=self._occupancy or {})

    def perfetto(self, path):
        if self._recorder is None:
            raise ValueError("no recorder; run in timing mode")
        save_perfetto(self._recorder, path)


def run(*, ptx_src: str | None = None, ptx_path: str | Path | None = None,
        grid: tuple[int,int,int], block: tuple[int,int,int],
        params: dict[str, np.ndarray | int],
        mode: str = "functional", config: Any = None, seed: int = 0,
        stream_id: int = 0, kernel_name: str = "<unnamed>") -> Result:
    """Run a PTX kernel under the simulator."""
    if ptx_src is None:
        if ptx_path is None:
            raise ValueError("provide ptx_src or ptx_path")
        ptx_src = Path(ptx_path).read_text()

    outputs = {k: v for k, v in params.items() if isinstance(v, np.ndarray)}

    if mode == "functional":
        # Resolve cluster_size from config (if provided)
        fn_cluster_size = 1
        if config is not None and not isinstance(config, (str, __import__("pathlib").Path)):
            fn_cluster_size = getattr(config, "cluster_size", 1)
        functional_run(ptx_src, params=params, grid=grid, block=block,
                       cluster_size=fn_cluster_size)
        return Result(outputs=outputs, mode="functional", metrics={})
    if mode == "timing":
        from gpusim.frontend.parser import parse
        from gpusim.config.loader import load_default, load_yaml
        from gpusim.core.device import Device
        from gpusim.config.schema import DeviceConfig, SMConfig
        cfg = load_default() if config is None else (
            load_yaml(config) if isinstance(config, (str, Path)) else config
        )
        if isinstance(cfg, SMConfig):
            cfg = DeviceConfig(n_sm=1, sm=cfg)
        k = parse(ptx_src, "<inline>")
        rec = Recorder()
        dev = Device(cfg, recorder=rec)
        res = dev.run(kernel=k, grid=grid, block=block, params=params,
                      stream_id=stream_id, kernel_name=kernel_name)
        result = Result(
            outputs=res.outputs, mode="timing",
            metrics={"cycles": res.cycles, "occupancy": res.occupancy},
            _recorder=rec, _kernel_name=k.name, _grid=grid, _block=block,
            _occupancy=res.occupancy,
            stream_id=stream_id,
        )
        return result
    raise NotImplementedError(f"mode={mode!r} not implemented yet")


_STREAM_ID_COUNTER = 0


def _next_stream_id() -> int:
    global _STREAM_ID_COUNTER
    sid = _STREAM_ID_COUNTER
    _STREAM_ID_COUNTER += 1
    return sid


def _reset_stream_id_counter() -> None:
    """Test-only helper to reset the global stream id counter."""
    global _STREAM_ID_COUNTER
    _STREAM_ID_COUNTER = 0


@dataclass
class GridLaunch:
    ptx_src: str
    kernel_name: str
    grid: tuple
    block: tuple
    params: dict
    config: object | None = None    # type: Config; lazy-typed to avoid circular


@dataclass
class Stream:
    stream_id: int = field(default_factory=_next_stream_id)
    pending: "deque[GridLaunch]" = field(default_factory=deque)
    inflight: GridLaunch | None = None
    completed: list = field(default_factory=list)

    def launch(self, ptx_src: str, grid: tuple, block: tuple,
               params: dict, *, kernel_name: str = "<unnamed>",
               config=None) -> None:
        self.pending.append(GridLaunch(
            ptx_src=ptx_src, kernel_name=kernel_name,
            grid=grid, block=block, params=params, config=config,
        ))

    def is_idle(self) -> bool:
        return self.inflight is None and not self.pending


@dataclass
class MultiStreamResult:
    streams: dict             # int -> list[Result]
    total_cycles: int = 0
    _recorder: object | None = None
