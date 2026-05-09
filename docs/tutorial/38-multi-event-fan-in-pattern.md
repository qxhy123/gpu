# Chapter 38 — Multi-Event Fan-In Pattern

## The Fan-In Problem

Chapter 34 showed the fan-out pattern: one producer records an event, and multiple consumer streams call `stream.wait(ev)` to unblock simultaneously. Fan-in is the mirror image: multiple producers each signal their own event, and a single consumer stream must wait until **all** of them have signaled before it can proceed.

This is a common pattern in GPU pipelines:

- A prefetch stage runs on two streams (e.g., one fetching weights, one fetching activations). The compute stage must wait for both to complete before it can launch.
- An all-reduce operation splits across multiple streams; a reduce-scatter step must wait for all partial results.
- A multi-stage preprocessing pipeline branches into parallel transformations that must all finish before the next stage aggregates.

## `Stream.wait_all` Semantics

Phase 9 adds `Stream.wait_all(events)` to handle this case cleanly:

```python
s_c.wait_all([ev_a, ev_b])
```

This queues a barrier on stream `s_c` that does not resolve until **every event in the list has been signaled**. The semantics are AND-only — there is no OR variant. A stream blocked by `wait_all` will not dispatch any CTAs until the last outstanding event in its list has been recorded.

Internally, `wait_all` is implemented as a counter initialized to `len(events)`. Each time a depended-on event signals, the counter decrements. The consumer stream's block condition is `counter > 0`.

This is semantically identical to calling `stream.wait(ev)` multiple times:

```python
s_c.wait(ev_a)
s_c.wait(ev_b)
```

But `wait_all` is more readable when the list is long, and the single-call form also makes the intent clear in HTML timeline export — the visualizer renders a single multi-input arrow rather than two overlapping single-input arrows.

## Running the Demo

```bash
python examples/multi_event_fan_in/run.py
```

The demo has three streams: two producers (`s_a`, `s_b`) that each write 1s into their output arrays, and one consumer (`s_c`) that adds them together. The consumer is blocked by `wait_all([ev_a, ev_b])`.

```python
s_a = Stream(); s_b = Stream(); s_c = Stream()
ev_a = Event(); ev_b = Event()

s_a.launch(ptx_src=write_ptx, params={"OUT": A}, kernel_name="write_a", ...)
s_a.record(ev_a)

s_b.launch(ptx_src=write_ptx, params={"OUT": B}, kernel_name="write_b", ...)
s_b.record(ev_b)

s_c.wait_all([ev_a, ev_b])
s_c.launch(ptx_src=combine_ptx, params={"A": A, "B": B, "OUT": OUT},
           kernel_name="combine", ...)

multi_res = gpusim.synchronize(streams=[s_a, s_b, s_c], config=cfg)
```

After synchronization, `OUT[i]` should be 2 for all `i` (1 from A plus 1 from B).

## 看模拟器

**查看 HTML §30 的事件时间线：**

Open the HTML report and navigate to section §30 (event timeline). This section renders each event as a vertical tick on the timeline and draws arrows from `record` to `wait` edges. For a `wait_all` call, both `ev_a` and `ev_b` show arrows pointing into the consumer stream's wait node.

The key thing to verify: both producer events must be signaled *before* the consumer kernel's first CTA is dispatched. Check the cycle numbers:

```python
ev_a_cycle = multi_res.event_signal_cycle(ev_a)
ev_b_cycle = multi_res.event_signal_cycle(ev_b)
consumer_start = multi_res.streams[s_c.stream_id][0].start_cycle

print(f"ev_a signaled at cycle {ev_a_cycle}")
print(f"ev_b signaled at cycle {ev_b_cycle}")
print(f"consumer started at cycle {consumer_start}")
assert consumer_start >= max(ev_a_cycle, ev_b_cycle)
```

If the consumer started before the later of the two events, the wait_all semantics are violated — a useful invariant to check when debugging complex pipelines.

Also check `multi_res.stream_summary()` to see how long stream `s_c` spent waiting vs. executing:

```python
print(multi_res.stream_summary())
```

The wait cycles for `s_c` should be roughly equal to `max(s_a_duration, s_b_duration)` — it was blocked until the slower producer finished.

## 改一改

**删除一个事件 — 验证消费者永远无法解锁：**

Remove one of the events from the `wait_all` list and instead of passing both producers, only pass one:

```python
s_c.wait_all([ev_a])          # ev_b no longer required
# s_b still records ev_b, but s_c doesn't care
```

The consumer will unblock after `ev_a` signals, without waiting for `ev_b`. That's fine in this small demo (the data might be partially ready), but you can simulate the bug case: drop `s_b.record(ev_b)` entirely but keep `s_c.wait_all([ev_a, ev_b])`. Now `ev_b` is never signaled, and `s_c` blocks forever.

To make this observable without hanging the simulator, wrap the synchronize call with a timeout:

```python
import signal

def handler(sig, frame):
    raise TimeoutError("deadlock detected")

signal.signal(signal.SIGALRM, handler)
signal.alarm(5)
try:
    multi_res = gpusim.synchronize(streams=[s_a, s_b, s_c], config=cfg)
except TimeoutError:
    print("Consumer never unblocked — deadlock confirmed")
finally:
    signal.alarm(0)
```

This confirms that `wait_all` semantics require all events in the list to be recorded by active streams, and dropping any producer event causes a permanent stall.

## 真机对照

On real hardware, there is no single `cudaStreamWaitAllEvents` API call. Instead, you achieve fan-in by calling `cudaStreamWaitEvent` once per event:

```c
cudaStreamWaitEvent(s_c, ev_a, 0);
cudaStreamWaitEvent(s_c, ev_b, 0);
```

CUDA processes these in order: `s_c` first waits for `ev_a`, then for `ev_b`. The final effect is identical to AND-semantics: the next operation enqueued on `s_c` after both `cudaStreamWaitEvent` calls will not execute until both events have been recorded.

On H100, events are implemented via hardware-level semaphores in the GPC. When `cudaEventRecord` fires, it writes a completion token into the stream's work queue. `cudaStreamWaitEvent` inserts a wait instruction that spins the stream's CTA dispatcher until it reads the matching token.

The simulator's `wait_all` is a cleaner abstraction that compiles down to the same multi-wait semantics, but makes the fan-in intent explicit — the single method call documents that all events are required before the consumer proceeds.
