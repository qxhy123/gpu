from __future__ import annotations
from pathlib import Path
import yaml
from .schema import (
    SMConfig, SchedulerConfig, RegFileConfig, FUConfig, CacheConfig, HBMConfig,
    TensorCoreConfig, DeviceConfig, CtaSchedulerConfig,
)

_DEFAULT_PATH = Path(__file__).parent / "default_hopper.yaml"


def _build_sm_config(sm_dict: dict) -> SMConfig:
    sched = SchedulerConfig(**(sm_dict.get("scheduler") or {}))
    rf = RegFileConfig(**(sm_dict.get("regfile") or {}))
    fu = FUConfig(**(sm_dict.get("fu") or {}))
    tensor_core = TensorCoreConfig(**(sm_dict.get("tensor_core") or {}))
    base = {k: v for k, v in sm_dict.items()
            if k not in ("scheduler", "regfile", "fu", "tensor_core",
                          "cache", "hbm")}
    return SMConfig(scheduler=sched, regfile=rf, fu=fu,
                     tensor_core=tensor_core, **base)


def _from_dict(d: dict) -> DeviceConfig:
    has_device = "device" in d
    has_sm_node = "sm" in d

    if has_device:
        device_d = d.get("device") or {}
        scheduler_d = device_d.get("scheduler") or {}
        scheduler = CtaSchedulerConfig(**scheduler_d)
        sm_dict = d.get("sm") or {}
        sm_cfg = _build_sm_config(sm_dict)
        cache = CacheConfig(**(d.get("cache") or {}))
        hbm = HBMConfig(**(d.get("hbm") or {}))
        n_sm = device_d.get("n_sm", 8)
        return DeviceConfig(n_sm=n_sm, sm=sm_cfg, cache=cache, hbm=hbm,
                             scheduler=scheduler)

    # Legacy
    if has_sm_node:
        sm_dict = d.get("sm") or {}
        cache = CacheConfig(**(d.get("cache") or {}))
        hbm = HBMConfig(**(d.get("hbm") or {}))
    else:
        sm_dict = {k: v for k, v in d.items() if k not in ("cache", "hbm")}
        cache = CacheConfig(**(d.get("cache") or {}))
        hbm = HBMConfig(**(d.get("hbm") or {}))
    sm_cfg = _build_sm_config(sm_dict)
    return DeviceConfig(n_sm=1, sm=sm_cfg, cache=cache, hbm=hbm,
                         scheduler=CtaSchedulerConfig())


def load_default() -> DeviceConfig:
    return load_yaml(_DEFAULT_PATH)


def load_yaml(path: str | Path) -> DeviceConfig:
    base = yaml.safe_load(_DEFAULT_PATH.read_text()) or {}
    over = yaml.safe_load(Path(path).read_text()) or {}
    merged = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return _from_dict(merged)
