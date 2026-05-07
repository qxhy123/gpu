from gpusim.trace.events import EventKind
from gpusim.trace.recorder import Recorder


def test_warp_state_rle_compresses_runs():
    r = Recorder()
    for c in range(5):
        r.warp_state(cycle=c, warp_id=0, state="ISSUED", pc=0)
    for c in range(5, 8):
        r.warp_state(cycle=c, warp_id=0, state="SCOREBOARD", pc=1)
    segs = list(r.warp_state_segments(warp_id=0))
    # one segment of (0..4, ISSUED) and one of (5..7, SCOREBOARD)
    assert len(segs) == 2
    assert segs[0].start == 0 and segs[0].end == 4 and segs[0].state == "ISSUED"
    assert segs[1].start == 5 and segs[1].end == 7 and segs[1].state == "SCOREBOARD"


def test_instr_issue_event_recorded():
    r = Recorder()
    r.instr_issue(cycle=10, warp_id=0, pc=5, op="add.f32",
                  src_loc=("k.ptx", 12), active_mask=0xFFFFFFFF)
    evs = list(r.instr_issues())
    assert len(evs) == 1 and evs[0].pc == 5 and evs[0].op == "add.f32"


def test_smem_access_event_recorded():
    r = Recorder()
    r.smem_access(cycle=20, warp_id=0, conflict_degree=4, addresses=[0]*32)
    evs = list(r.smem_accesses())
    assert len(evs) == 1 and evs[0].conflict_degree == 4


def test_gmem_access_event_recorded():
    r = Recorder()
    r.gmem_access(cycle=20, warp_id=0, n_transactions=2, efficiency=0.5,
                  addresses=[i*4 for i in range(32)])
    evs = list(r.gmem_accesses())
    assert len(evs) == 1 and evs[0].n_transactions == 2


def test_div_push_pop_recorded():
    r = Recorder()
    r.div_push(cycle=5, warp_id=0, pc=3, rpc=10, taken_mask=0xFFFF)
    r.div_pop(cycle=15, warp_id=0, pc=10)
    assert len(list(r.div_events())) == 2


def test_cta_lifecycle():
    r = Recorder()
    r.cta_launch(cycle=0, cta_id=0, warps=4, regs=16, smem_bytes=512)
    r.cta_retire(cycle=200, cta_id=0)
    evs = list(r.cta_events())
    assert len(evs) == 2
