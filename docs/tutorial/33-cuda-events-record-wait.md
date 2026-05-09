# Chapter 33 — CUDA Events: Record and Wait

## The Event Lifecycle

A CUDA event is a synchronization primitive that lets one stream wait for work in another stream to complete. The Phase 8 simulator models the full event lifecycle in four steps:

1. **Create** — instantiate an `Event` object. The event starts in the "unrecorded" state.
2. **Record** — call `stream.record(ev)`. This inserts a `_RecordMarker` into the stream's pending queue. When the scheduler drains the marker, it associates the event with that stream and notes the current cycle as the record cycle.
3. **Wait** — call `other_stream.wait(ev)`. This registers the event in `other_stream.event_waits`. The `ConcurrentStreamScheduler` checks `is_event_blocked()` each cycle and skips dispatching CTAs from `other_stream` until the event is signaled.
4. **Signal** — after the recorded grid retires, the scheduler calls `ev.signal(cycle)`. All waiting streams are unblocked on the next cycle.

```
Stream A:  [launch kernel_write] → [record ev] → retire → signal ev
Stream B:              [wait ev] ....blocked.... → [launch kernel_read]
```

The key invariant: `kernel_read` on stream B will never begin executing until `kernel_write` on stream A has fully retired. This matches the CUDA specification's guarantee for `cudaEventRecord` + `cudaStreamWaitEvent`.

## Event API in the Simulator

```python
from gpusim.api import Stream, Event

ev = Event()            # create
s_a.record(ev)          # record into stream A's pending queue
s_b.wait(ev)            # register wait on stream B
```

`Event` is a plain dataclass with two fields: `signal_cycle` (set when signaled, `None` until then) and `recorded_in_stream`. `is_signaled(cycle)` returns `True` if `signal_cycle is not None and signal_cycle <= cycle`.

## 看模拟器

Run the producer-consumer demo:

```bash
python examples/event_producer_consumer/run.py
```

```python
s_a = Stream()
s_b = Stream()
ev  = Event()

s_a.launch(ptx_src=write_ptx, kernel_name="write", ...)   # producer
s_a.record(ev)
s_b.wait(ev)
s_b.launch(ptx_src=read_ptx,  kernel_name="read",  ...)   # consumer

multi_res = gpusim.synchronize(streams=[s_a, s_b], config=cfg)
```

After synchronize, query the event wait cycles:

```python
wait_cycles = multi_res.event_wait_cycles_per_stream()
print(wait_cycles)
# {stream_b_id: <N>}  — cycles stream B was blocked waiting for the event
```

`multi_res.event_wait_cycles_per_stream()` returns a dict mapping stream ID to the number of cycles that stream spent blocked on event waits. A large value here means the consumer was idle for many cycles while the producer was running — an opportunity to pipeline (Chapter 36).

You can also inspect the HTML report for the event timeline section (§30 in the generated report):

```python
from gpusim.viz.html_report import save_html
save_html(multi_res.results[0], pathlib.Path("report.html"),
          kernel_name="write", grid=(1,1,1), block=(32,1,1),
          cycles=multi_res.total_cycles, occupancy={})
```

The §30 event-timeline table lists each `StreamEvent` record with its `event_id`, `kind` (`record` or `wait_satisfied`), `stream_id`, and `cycle`.

## 改一改

**Skip the event to observe the race condition.** Remove `s_a.record(ev)` and `s_b.wait(ev)`:

```python
s_a.launch(ptx_src=write_ptx, kernel_name="write", ...)
# s_a.record(ev)     <-- commented out
# s_b.wait(ev)       <-- commented out
s_b.launch(ptx_src=read_ptx,  kernel_name="read",  ...)

multi_res = gpusim.synchronize(streams=[s_a, s_b], config=cfg)
```

Without the event, the scheduler may dispatch the `read` kernel before the `write` kernel completes. The `OUT` array will contain stale or zero values — the read raced with the write. `event_wait_cycles_per_stream()` returns an empty dict, confirming no synchronization occurred.

This experiment makes the value of events concrete: the correctness guarantee costs some wait cycles, but those cycles prevent data races that produce incorrect results.

## 真机対照

CUDA event synchronization across streams uses two API calls:

```cpp
cudaEvent_t ev;
cudaEventCreate(&ev);
cudaEventRecord(ev, stream_a);       // record into stream A
cudaStreamWaitEvent(stream_b, ev, 0); // stream B waits for ev
```

After `cudaStreamWaitEvent`, all subsequent work submitted to `stream_b` is guaranteed to start only after the work preceding `cudaEventRecord` on `stream_a` has completed. This applies even across devices (with `cudaEventDisableTiming` and peer-to-peer access enabled).

The simulator models this semantics faithfully. The only difference is granularity: the simulator fires the signal after the recorded grid retires (at kernel granularity), while hardware fires it at a finer subcycle precision. For most use cases, kernel-granularity synchronization is sufficient and matches observable behavior in Nsight Systems timelines.
