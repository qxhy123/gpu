from typer.testing import CliRunner
from gpusim.cli import app

def test_cli_show_kernel_summary(tmp_path):
    p = tmp_path / "k.ptx"
    p.write_text(
        ".visible .entry k(.param .u32 N) { .reg .u32 %r<2>; mov.u32 %r1, 1; }"
    )
    res = CliRunner().invoke(app, ["show", str(p)])
    assert res.exit_code == 0, res.output
    assert "k" in res.output  # kernel name printed
    assert "params" in res.output.lower()

def test_cli_doctor():
    res = CliRunner().invoke(app, ["doctor"])
    assert res.exit_code == 0

def test_cli_run_functional_vector_add(tmp_path):
    # mini kernel that just stores tid into output
    ptx = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<3>; .reg .u64 %rd<4>;
        ld.param.u64 %rd1, [OUT];
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd3, %rd1, %rd2;
        st.global.u32 [%rd3], %r1;
    }
    """
    pf = tmp_path / "k.ptx"; pf.write_text(ptx)
    import numpy as np
    out = np.zeros(32, dtype=np.uint32)
    np.save(tmp_path / "out.npy", out)
    res = CliRunner().invoke(app, [
        "run", str(pf), "--grid", "1", "--block", "32",
        "--inputs", f"OUT:{tmp_path/'out.npy'}",
        "--mode", "functional",
    ])
    assert res.exit_code == 0, res.output
    out_after = np.load(tmp_path / "out.npy")
    assert list(out_after) == list(range(32))
