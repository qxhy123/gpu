import numpy as np, pathlib, gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/vector_add/kernel.ptx").read_text()


def test_full_pipeline_emits_html_and_perfetto(tmp_path):
    n = 1024
    rng = np.random.RandomState(0)
    a = rng.randn(n).astype(np.float32); b = rng.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    res = gpusim.run(ptx_src=PTX, grid=(8,1,1), block=(128,1,1),
                     params={"A":a,"B":b,"C":c,"N":n}, mode="timing")
    np.testing.assert_allclose(c, a + b, rtol=1e-5)

    res.html_report(tmp_path / "report.html")
    assert (tmp_path / "report.html").exists()
    res.perfetto(tmp_path / "trace.json")
    assert (tmp_path / "trace.json").exists()

    df = res.stall_df
    assert "state" in df.columns
    assert "cycles" in df.columns

    df2 = res.events_df
    assert "warp_id" in df2.columns

    fig = res.timeline(warp=0)
    assert fig is not None


def test_full_pipeline_exposes_cache_metrics(tmp_path):
    import numpy as np, pathlib, gpusim
    PTX = (pathlib.Path(__file__).parents[2] / "examples/vector_add/kernel.ptx").read_text()
    n = 1024
    rng = np.random.RandomState(0)
    a = rng.randn(n).astype(np.float32); b = rng.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    res = gpusim.run(ptx_src=PTX, grid=(8,1,1), block=(128,1,1),
                     params={"A":a,"B":b,"C":c,"N":n}, mode="timing")
    assert res.l1_events_df is not None
    assert res.l2_events_df is not None
    assert res.hbm_events_df is not None
    cm = res.cache_metrics
    assert "l1_hit_rate" in cm
    assert 0.0 <= cm["l1_hit_rate"] <= 1.0
