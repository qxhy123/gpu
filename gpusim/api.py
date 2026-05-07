from __future__ import annotations
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
        mode: str = "functional", config: Any = None, seed: int = 0) -> Result:
    """Run a PTX kernel under the simulator."""
    if ptx_src is None:
        if ptx_path is None:
            raise ValueError("provide ptx_src or ptx_path")
        ptx_src = Path(ptx_path).read_text()

    outputs = {k: v for k, v in params.items() if isinstance(v, np.ndarray)}

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
        rec = Recorder()
        sm = SM(cfg, recorder=rec)
        res = sm.run(kernel=k, grid=grid, block=block, params=params)
        return Result(
            outputs=res.outputs, mode="timing",
            metrics={"cycles": res.cycles, "occupancy": res.occupancy},
            _recorder=rec, _kernel_name=k.name, _grid=grid, _block=block,
            _occupancy=res.occupancy,
        )
    raise NotImplementedError(f"mode={mode!r} not implemented yet")
