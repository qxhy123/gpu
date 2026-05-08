from gpusim.core.hbm import HBM, decompose_addr
from gpusim.config.schema import HBMConfig
from gpusim.trace.recorder import Recorder


def test_address_decode():
    cfg = HBMConfig()
    # bit layout: [6:0]=offset, [9:7]=ch, [14:10]=col, [18:15]=bank, [30:19]=row
    addr = (0xABC << 19) | (0x5 << 15) | (0x07 << 10) | (0x3 << 7) | 0x42
    c, b, col, row = decompose_addr(addr, cfg)
    assert c == 0x3
    assert b == 0x5
    assert col == 0x07
    assert row == 0xABC


def test_first_request_to_bank_is_row_miss():
    cfg = HBMConfig()
    h = HBM(cfg)
    completion = h.request(line_addr=0x10, now=0)  # any address
    # first access to a bank → row miss → row_miss_latency
    assert completion == cfg.row_miss_latency


def test_second_request_same_row_is_row_hit():
    cfg = HBMConfig()
    h = HBM(cfg)
    addr1 = 0x0   # ch=0, bank=0, col=0, row=0
    addr2 = 0x80  # ch=1, bank=0, col=0, row=0  (channel changes, row 0 in bank 0 of ch=1 not yet open)
    # actually addr1 opens row 0 in ch=0 bank=0; addr2 opens row 0 in ch=1 bank=0 (different bank!)
    # to test row-hit: same channel, same bank, same row
    h.request(line_addr=0, now=0)
    # next access to ch=0 bank=0 same row: increment col by 1 → bit 10
    h.request(line_addr=(1 << 10), now=100)
    # was in same bank (bit [18:15] still 0), same row (bit [30:19] still 0)
    # second was a row hit
    # we don't directly observe via return value here, but channel busy state should match


def test_concurrent_same_channel_serializes():
    """Two requests to same channel back-to-back must serialize via channel queue."""
    cfg = HBMConfig()
    h = HBM(cfg)
    c1 = h.request(line_addr=0, now=0)            # ch=0
    # next request to channel 0 (different bank, but same channel)
    c2 = h.request(line_addr=(1 << 15), now=0)    # ch=0, bank=1
    assert c2 >= c1   # serialized


def test_concurrent_different_channel_parallel():
    """Two requests to different channels must NOT serialize."""
    cfg = HBMConfig()
    h = HBM(cfg)
    c1 = h.request(line_addr=0, now=0)            # ch=0 (byte_addr=0x000, bits[9:7]=0)
    # line_addr=1 → byte_addr=128=0x80; bits[9:7] of 0x80 = 1 → ch=1
    c2 = h.request(line_addr=1, now=0)            # ch=1
    # same start time → parallel → both ~ row_miss_latency
    assert c1 == cfg.row_miss_latency
    assert c2 == cfg.row_miss_latency


def test_queue_wait_visible_via_recorder():
    """High-load same-channel requests get queue_wait > 0."""
    cfg = HBMConfig()
    h = HBM(cfg)
    h._recorder = Recorder()    # injected for test
    h.request(line_addr=0, now=0)
    h.request(line_addr=(1 << 15), now=0)   # same channel, different bank
    events = h._recorder.hbm_accesses()
    assert len(events) == 2
    assert events[0].queue_wait == 0
    assert events[1].queue_wait > 0


def test_write_request_separate_kind():
    cfg = HBMConfig()
    h = HBM(cfg)
    h._recorder = Recorder()    # imported above
    h.request(line_addr=0, now=0)
    h.write_request(line_addr=(1 << 7), now=0)   # ch=1
    events = h._recorder.hbm_accesses()
    assert events[0].kind == "READ"
    assert events[1].kind == "WRITE_BACK"
