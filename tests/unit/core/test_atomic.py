def test_l2_atomic_queue_first_arrival():
    from gpusim.core.atomic import L2AtomicQueue
    q = L2AtomicQueue(n_slots=32)
    completion = q.enqueue(line_addr=0x1000, sm_id=0, op="add", op_kind="atom",
                              arrival=10, atomic_op_latency=10, l2_hit_latency=20)
    # First arrival: arrival + l2_hit_latency + atomic_op_latency = 10 + 20 + 10 = 40
    assert completion == 40


def test_l2_atomic_queue_serializes_same_line():
    from gpusim.core.atomic import L2AtomicQueue
    q = L2AtomicQueue(n_slots=32)
    c1 = q.enqueue(line_addr=0x1000, sm_id=0, op="add", op_kind="atom",
                     arrival=0, atomic_op_latency=10, l2_hit_latency=20)
    # First: 0 + 20 + 10 = 30
    assert c1 == 30
    c2 = q.enqueue(line_addr=0x1000, sm_id=1, op="add", op_kind="atom",
                     arrival=5, atomic_op_latency=10, l2_hit_latency=20)
    # Second: max(5+20, 30) + 10 = max(25, 30) + 10 = 40
    assert c2 == 40
    c3 = q.enqueue(line_addr=0x1000, sm_id=2, op="add", op_kind="atom",
                     arrival=10, atomic_op_latency=10, l2_hit_latency=20)
    # Third: max(10+20, 40) + 10 = 50
    assert c3 == 50


def test_l2_atomic_queue_different_lines_parallel():
    from gpusim.core.atomic import L2AtomicQueue
    q = L2AtomicQueue(n_slots=32)
    c1 = q.enqueue(line_addr=0x1000, sm_id=0, op="add", op_kind="atom",
                     arrival=0, atomic_op_latency=10, l2_hit_latency=20)
    c2 = q.enqueue(line_addr=0x2000, sm_id=1, op="add", op_kind="atom",
                     arrival=0, atomic_op_latency=10, l2_hit_latency=20)
    # Different lines don't serialize
    assert c1 == c2 == 30


def test_l2_atomic_queue_depth_at_now():
    from gpusim.core.atomic import L2AtomicQueue
    q = L2AtomicQueue(n_slots=32)
    q.enqueue(line_addr=0x1000, sm_id=0, op="add", op_kind="atom",
                arrival=0, atomic_op_latency=10, l2_hit_latency=20)
    q.enqueue(line_addr=0x1000, sm_id=1, op="add", op_kind="atom",
                arrival=0, atomic_op_latency=10, l2_hit_latency=20)
    # At cycle 35, first completes (c=30) + second still in-flight (c=40)
    # → depth at 35 = 1
    assert q.queue_depth(0x1000, now=35) == 1
    # At cycle 50, both done → depth 0
    assert q.queue_depth(0x1000, now=50) == 0
