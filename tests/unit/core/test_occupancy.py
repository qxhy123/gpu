from gpusim.core.occupancy import compute_occupancy, OccupancyResult
from gpusim.config.loader import load_default

def test_warps_bottleneck():
    cfg = load_default().sm
    r = compute_occupancy(cfg, threads_per_cta=256, regs_per_thread=8, smem_per_cta=1024)
    assert r.active_ctas <= cfg.max_ctas_per_sm
    assert r.bottleneck == "warps"
    assert r.active_ctas == 8

def test_regs_bottleneck():
    cfg = load_default().sm
    r = compute_occupancy(cfg, threads_per_cta=128, regs_per_thread=100, smem_per_cta=512)
    assert r.bottleneck == "regs"

def test_smem_bottleneck():
    cfg = load_default().sm
    r = compute_occupancy(cfg, threads_per_cta=128, regs_per_thread=8, smem_per_cta=32*1024)
    assert r.bottleneck == "smem"
    assert r.active_ctas == 1

def test_max_ctas_capped():
    cfg = load_default().sm
    r = compute_occupancy(cfg, threads_per_cta=32, regs_per_thread=4, smem_per_cta=128)
    assert r.active_ctas == cfg.max_ctas_per_sm
