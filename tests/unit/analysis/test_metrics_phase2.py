import pandas as pd
from gpusim.analysis.metrics import (
    l1_hit_rate, l2_hit_rate, mshr_merge_rate,
    cache_hierarchy_breakdown, bandwidth_per_channel,
    channel_utilization, row_buffer_hit_rate,
    queue_wait_distribution, wb_traffic_fraction,
)

def test_l1_hit_rate():
    df = pd.DataFrame([
        {"kind":"HIT"}, {"kind":"HIT"}, {"kind":"MISS_NEW"}, {"kind":"MISS_MERGE"},
    ])
    assert l1_hit_rate(df) == 0.5

def test_l2_hit_rate():
    df = pd.DataFrame([
        {"kind":"HIT"}, {"kind":"HIT"}, {"kind":"MISS_LOAD"},
        {"kind":"EVICT_CLEAN"}, {"kind":"EVICT_DIRTY"},
    ])
    # only HIT and MISS_* count as accesses (EVICT_* are side-effects)
    # 2/3 = 66.6...%
    assert abs(l2_hit_rate(df) - 2/3) < 1e-9

def test_mshr_merge_rate():
    df = pd.DataFrame([
        {"kind":"HIT"}, {"kind":"MISS_NEW"}, {"kind":"MISS_MERGE"},
        {"kind":"MISS_MERGE"}, {"kind":"MISS_NEW"},
    ])
    # 2 merges out of 4 total misses
    assert mshr_merge_rate(df) == 0.5

def test_cache_hierarchy_breakdown_sums_to_one():
    l1 = pd.DataFrame([{"kind":"HIT"}, {"kind":"HIT"}, {"kind":"MISS_NEW"}, {"kind":"MISS_NEW"}])
    l2 = pd.DataFrame([{"kind":"HIT"}, {"kind":"MISS_LOAD"}])
    out = cache_hierarchy_breakdown(l1, l2)
    assert "l1_hit" in out and "l2_hit" in out and "hbm" in out
    assert abs(sum(out.values()) - 1.0) < 1e-9

def test_bandwidth_per_channel_returns_list():
    hbm = pd.DataFrame([
        {"channel":0, "served_at":100, "queue_wait":0, "kind":"READ"},
        {"channel":0, "served_at":200, "queue_wait":0, "kind":"READ"},
        {"channel":1, "served_at":150, "queue_wait":0, "kind":"READ"},
    ])
    bw = bandwidth_per_channel(hbm, total_cycles=1000, line_bytes=128)
    assert len(bw) == 8   # default channels
    # ch 0: 2 transfers × 128 bytes / 1000 cycles
    assert bw.iloc[0] > 0
    assert bw.iloc[2] == 0      # no requests on channel 2

def test_channel_utilization():
    hbm = pd.DataFrame([
        {"channel":0, "cycle":0, "served_at":100, "kind":"READ", "queue_wait":0},
    ])
    cu = channel_utilization(hbm, total_cycles=1000, n_channels=8)
    assert len(cu) == 8
    assert cu.iloc[0] == 0.1     # 100/1000

def test_row_buffer_hit_rate():
    df = pd.DataFrame([
        {"row_kind":"ROW_HIT"}, {"row_kind":"ROW_HIT"},
        {"row_kind":"ROW_MISS"},
    ])
    assert abs(row_buffer_hit_rate(df) - 2/3) < 1e-9

def test_queue_wait_distribution():
    df = pd.DataFrame([
        {"queue_wait":0}, {"queue_wait":0}, {"queue_wait":50}, {"queue_wait":100},
    ])
    dist = queue_wait_distribution(df)
    assert len(dist) > 0   # some histogram bins

def test_wb_traffic_fraction():
    df = pd.DataFrame([
        {"kind":"READ"}, {"kind":"READ"}, {"kind":"WRITE_BACK"},
        {"kind":"READ"}, {"kind":"WRITE_BACK"},
    ])
    # 2 wb / 5 total = 0.4
    assert wb_traffic_fraction(df) == 0.4
