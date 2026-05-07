from gpusim.config.loader import load_default, load_yaml
from gpusim.config.schema import SMConfig

def test_default_loads():
    c = load_default()
    assert isinstance(c, SMConfig)
    assert c.sub_cores == 4
    assert c.warps_per_sm == 64
    assert c.smem_banks == 32
    assert c.regfile.banks == 4
    assert c.scheduler.policy == "gto"

def test_overrides_via_yaml(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("scheduler:\n  policy: lrr\n")
    c = load_yaml(p)
    assert c.scheduler.policy == "lrr"
    assert c.sub_cores == 4
