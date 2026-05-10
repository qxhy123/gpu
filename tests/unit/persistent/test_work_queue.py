def test_work_queue_push_pop_fifo():
    from gpusim.persistent.queue import WorkQueue
    q = WorkQueue()
    q.push("a")
    q.push("b")
    q.push("c")
    assert q.pop() == "a"
    assert q.pop() == "b"
    assert q.pop() == "c"
    assert q.pop() is None


def test_work_queue_stop():
    from gpusim.persistent.queue import WorkQueue
    q = WorkQueue()
    q.push(1)
    q.stop()
    assert q.is_stopped()
    # Pop still works for remaining items
    assert q.pop() == 1
    assert q.pop() is None


def test_work_queue_push_after_stop_raises():
    from gpusim.persistent.queue import WorkQueue
    import pytest
    q = WorkQueue()
    q.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        q.push(1)


def test_work_queue_is_empty():
    from gpusim.persistent.queue import WorkQueue
    q = WorkQueue()
    assert q.is_empty()
    q.push(1)
    assert not q.is_empty()
    q.pop()
    assert q.is_empty()
