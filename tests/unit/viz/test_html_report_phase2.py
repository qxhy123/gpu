import numpy as np, pathlib, gpusim

PTX = (pathlib.Path(__file__).parents[3] / "examples/vector_add/kernel.ptx").read_text()


def test_html_report_includes_phase2_sections(tmp_path):
    n = 1024
    rng = np.random.RandomState(0)
    a = rng.randn(n).astype(np.float32); b = rng.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    res = gpusim.run(ptx_src=PTX, grid=(8,1,1), block=(128,1,1),
                     params={"A":a,"B":b,"C":c,"N":n}, mode="timing")
    html_path = tmp_path / "report.html"
    res.html_report(html_path)
    text = html_path.read_text()
    assert "Cache hierarchy hit rate" in text
    assert "HBM channel utilization" in text
    assert "Row buffer locality" in text
    assert "Write-back traffic" in text
    # Eviction heatmap may or may not be present (only on thrash kernels)
