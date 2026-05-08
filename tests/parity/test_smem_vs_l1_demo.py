import numpy as np, pathlib, gpusim

DIR = pathlib.Path(__file__).parents[2] / "examples/smem_vs_l1_demo"
PTX_SMEM = (DIR / "kernel_smem.ptx").read_text()
PTX_NOSMEM = (DIR / "kernel_no_smem.ptx").read_text()


def _run(ptx, A, B, C):
    gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(16,16,1),
               params={"A": A, "B": B, "C": C}, mode="functional")


def test_both_variants_compute_correct_matmul():
    rng = np.random.RandomState(0)
    A = rng.randn(16, 16).astype(np.float32)
    B = rng.randn(16, 16).astype(np.float32)
    C1 = np.zeros((16, 16), dtype=np.float32)
    C2 = np.zeros((16, 16), dtype=np.float32)
    _run(PTX_SMEM, A, B, C1)
    _run(PTX_NOSMEM, A, B, C2)
    np.testing.assert_allclose(C1, A @ B, rtol=1e-4)
    np.testing.assert_allclose(C2, A @ B, rtol=1e-4)
    np.testing.assert_allclose(C1, C2, rtol=1e-4)
