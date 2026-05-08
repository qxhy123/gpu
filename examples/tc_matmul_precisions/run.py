import numpy as np, pathlib, sys, gpusim

_DIR = pathlib.Path(__file__).parent
# Ensure the reference module can be imported by adding parent to path
sys.path.insert(0, str(_DIR.parent.parent))


def main():
    from examples.tc_matmul_precisions.reference import build_inputs, reference_output, output_dtype
    print("# tc_matmul_precisions: 6 dtype variants")
    print(f"{'variant':<8} {'cycles':<8} {'max diff vs numpy':<20}")
    for variant in ("fp32", "fp16", "bf16", "e4m3", "tf32", "int8"):
        A, B, C = build_inputs(variant, seed=0)
        out_dtype = output_dtype(variant)
        out = np.zeros(16 * 8, dtype=out_dtype)
        ptx = (_DIR / f"kernel_{variant}.ptx").read_text()
        res = gpusim.run(
            ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
            params={"A": A.flatten().copy(), "B": B.flatten().copy(),
                    "C": C.flatten().copy(), "OUT": out},
            mode="timing",
        )
        expected = reference_output(A, B, C, variant)
        diff = np.max(np.abs(out.reshape(16, 8).astype(np.float32) - expected.astype(np.float32)))
        cycles = res.metrics.get("cycles", "?")
        print(f"{variant:<8} {cycles:<8} {diff:<.2e}")


if __name__ == "__main__":
    main()
