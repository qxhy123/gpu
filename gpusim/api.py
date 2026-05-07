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
        sm = SM(cfg)
        res = sm.run(kernel=k, grid=grid, block=block, params=params)
        return Result(
            outputs=res.outputs, mode="timing",
            metrics={"cycles": res.cycles, "occupancy": res.occupancy},
        )
    raise NotImplementedError(f"mode={mode!r} not implemented yet")
