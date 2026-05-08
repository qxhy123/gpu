from __future__ import annotations
from pathlib import Path
import yaml
from .schema import SMConfig, SchedulerConfig, RegFileConfig, FUConfig, CacheConfig, HBMConfig

_DEFAULT_PATH = Path(__file__).parent / "default_hopper.yaml"


def _from_dict(d: dict) -> SMConfig:
    sched = SchedulerConfig(**(d.get("scheduler") or {}))
    rf = RegFileConfig(**(d.get("regfile") or {}))
    fu = FUConfig(**(d.get("fu") or {}))
    cache = CacheConfig(**(d.get("cache") or {}))     # NEW
    hbm = HBMConfig(**(d.get("hbm") or {}))           # NEW
    base = {k: v for k, v in d.items()
            if k not in ("scheduler", "regfile", "fu", "cache", "hbm")}
    return SMConfig(scheduler=sched, regfile=rf, fu=fu, cache=cache, hbm=hbm, **base)


def load_default() -> SMConfig:
    return load_yaml(_DEFAULT_PATH)


def load_yaml(path: str | Path) -> SMConfig:
    base = yaml.safe_load(_DEFAULT_PATH.read_text()) or {}
    over = yaml.safe_load(Path(path).read_text()) or {}
    merged = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return _from_dict(merged)
