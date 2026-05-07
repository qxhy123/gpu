import json, base64, io, pathlib, pytest
import numpy as np
import gpusim

DATA_DIR = pathlib.Path(__file__).parent / "data"


def _load_npy_b64(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)))


def _decode_outputs(rec: dict) -> dict:
    return {k: _load_npy_b64(v) for k, v in rec.get("outputs", {}).items()}


def _ref_files() -> list[pathlib.Path]:
    return sorted(DATA_DIR.glob("*.ref.json"))


@pytest.mark.reference
@pytest.mark.parametrize("ref_file", _ref_files() or [pytest.param(None, marks=pytest.mark.skip(reason="no fixtures"))])
def test_simulator_matches_reference_numerics(ref_file):
    rec = json.loads(ref_file.read_text())
    expected = _decode_outputs(rec)
    if not expected:
        pytest.skip("no expected outputs")
    # simulator side: derive same inputs by seed + shape, then run
    rng = np.random.RandomState(rec.get("inputs_seed", 0))
    params = {}
    for name, shape in rec["inputs_shape"].items():
        if not shape:
            params[name] = 0
        else:
            params[name] = rng.randn(*shape).astype(np.float32)
    # also bind output buffers (zeros)
    for name, arr in expected.items():
        params[name] = np.zeros_like(arr)

    ptx = (pathlib.Path(__file__).parents[2] / rec["ptx_path"]).read_text()
    gpusim.run(ptx_src=ptx, grid=tuple(rec["launch"]["grid"]) + (1,)*(3-len(rec["launch"]["grid"])),
               block=tuple(rec["launch"]["block"]) + (1,)*(3-len(rec["launch"]["block"])),
               params=params, mode="functional")
    for name, exp in expected.items():
        np.testing.assert_allclose(params[name], exp, rtol=1e-5)
