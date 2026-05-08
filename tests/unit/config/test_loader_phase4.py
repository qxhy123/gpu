def test_cta_scheduler_config_default():
    from gpusim.config.schema import CtaSchedulerConfig
    cfg = CtaSchedulerConfig()
    assert cfg.cta_policy == "rr"


def test_device_config_default():
    from gpusim.config.schema import DeviceConfig
    cfg = DeviceConfig()
    assert cfg.n_sm == 8
    assert cfg.scheduler.cta_policy == "rr"
    assert cfg.cache.l2_size_bytes == 4 * 1024 * 1024
    assert cfg.hbm.channels == 8
    assert cfg.sm.sub_cores == 4


def test_cache_config_has_l2_mshr_slots():
    from gpusim.config.schema import CacheConfig
    cfg = CacheConfig()
    assert cfg.l2_mshr_slots == 32


def test_tensor_core_config_has_bulk_store_fields():
    from gpusim.config.schema import TensorCoreConfig
    cfg = TensorCoreConfig()
    assert cfg.bulk_store_queue_capacity == 16
    assert cfg.bulk_store_latency_per_line == 4
