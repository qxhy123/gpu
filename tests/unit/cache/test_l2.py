from gpusim.core.cache.l2 import L2Cache
from gpusim.config.schema import CacheConfig


class MockHBM:
    def __init__(self, latency=130):
        self.latency = latency
        self.requests: list[tuple[int, str]] = []

    def request(self, line_addr: int, now: int) -> int:
        self.requests.append((line_addr, "READ"))
        return now + self.latency

    def write_request(self, line_addr: int, now: int) -> int:
        self.requests.append((line_addr, "WRITE_BACK"))
        return now + self.latency


def test_l2_mock_returns_fixed_latency():
    cfg = CacheConfig()
    hbm = MockHBM()
    l2 = L2Cache(cfg=cfg, hbm=hbm)
    completion = l2.fetch(line_addr=0x100, now=0)
    assert completion > 0
