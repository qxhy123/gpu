# Chapter 34 — Event Fanout Pattern

## One Producer, Multiple Consumers

The previous chapter showed a 1-to-1 producer-consumer event: one stream records, one stream waits. A common production pattern generalizes this to **1-to-N**: a single producer event fans out to multiple consumer streams, all waiting on the same event. Each consumer can then proceed independently as soon as the producer completes.

```
Stream A:  [write shared buffer] → [record ev] → retire → signal ev
                                                              |
Stream B:                          [wait ev] ..................→ [read]
Stream C:                          [wait ev] ..................→ [read]
Stream D:                          [wait ev] ..................→ [read]
```

All three consumer streams (B, C, D) are blocked until stream A signals `ev`. The moment the event is signaled, all three consumers are unblocked simultaneously and their CTAs become eligible for dispatch on the next scheduler cycle. This is the GPU equivalent of a **broadcast** or **condvar signal**: one producer notifies all waiting consumers at once.

## Why Fanout Beats Chaining

The alternative to fanout is a serial chain:

```
A writes → signals ev1 → B reads → signals ev2 → C reads → signals ev3 → D reads
```

This turns O(1) concurrent consumer work into O(N) serial work. If each consumer takes T cycles, the chain takes 3T cycles versus the fanout's T cycles (plus A's overhead). Any time consumers are independent — they all read from the same shared buffer without modifying it — fanout is the correct pattern.

## 看模拟器

Run the fanout demo:

```bash
python examples/event_fanout/run.py
```

```python
s_a     = Stream()
streams = [Stream() for _ in range(3)]
ev      = Event()

s_a.launch(ptx_src=write_ptx, kernel_name="write", ...)
s_a.record(ev)

for s, out in zip(streams, outs):
    s.wait(ev)
    s.launch(ptx_src=read_ptx, kernel_name=f"read_{s.stream_id}", ...)

multi_res = gpusim.synchronize(streams=[s_a] + streams, config=cfg)
```

After synchronize, inspect the StreamEvent records to confirm fanout behavior. The trace will contain multiple `wait_satisfied` entries for the same `event_id`:

```python
# Via the analysis layer (if stream_event_events is accessible):
for sid, launches in multi_res.streams.items():
    for r in launches:
        print(f"  stream {sid}: {r.metrics.get('cycles', '?')} cycles")
```

In the Perfetto trace (Chapter 29), the producer stream's `record` marker appears as a flow arrow that fans out to three separate consumer streams. Each arrow connects the record cycle on stream A to the first CTA dispatch cycle on the corresponding consumer stream. The visual confirms simultaneity: all three arrows arrive at the same cycle on streams B, C, and D.

Look for multiple `wait_satisfied` entries that share the same `event_id` in the HTML §30 table — this is the direct evidence of fanout in the simulator's trace data.

## 改一改

**Convert fanout to a serial consumer pipeline.** Replace the fanout with a chain of events so consumers run in sequence:

```python
s_a = Stream()
s_b = Stream()
s_c = Stream()
ev1 = Event()
ev2 = Event()

s_a.launch(ptx_src=write_ptx, kernel_name="write", ...)
s_a.record(ev1)

s_b.wait(ev1)
s_b.launch(ptx_src=read_ptx, kernel_name="read_b", ...)
s_b.record(ev2)

s_c.wait(ev2)
s_c.launch(ptx_src=read_ptx, kernel_name="read_c", ...)

multi_res = gpusim.synchronize(streams=[s_a, s_b, s_c], config=cfg)
print(f"Serial chain: {multi_res.total_cycles} cycles")
```

Then restore the fanout and compare total cycle counts. The fanout version should complete in fewer total cycles because streams B and C overlap after the event fires. The `event_wait_cycles_per_stream()` metric will show higher wait times for the chain (C waits for both A and B to finish) versus the fanout (B and C both wait only for A).

## 真机対照

The CUDA fanout pattern uses a single `cudaStreamWaitEvent` call per consumer:

```cpp
cudaEvent_t ev;
cudaEventCreate(&ev);
cudaEventRecord(ev, stream_a);           // producer records

cudaStreamWaitEvent(stream_b, ev, 0);   // consumer B waits
cudaStreamWaitEvent(stream_c, ev, 0);   // consumer C waits
cudaStreamWaitEvent(stream_d, ev, 0);   // consumer D waits

// launch on stream_b, stream_c, stream_d — all start after ev fires
```

This is analogous to the pthread condvar `broadcast` pattern in CPU programming: `pthread_cond_broadcast` wakes all threads blocked on the condvar, just as the CUDA event simultaneously unblocks all streams that called `cudaStreamWaitEvent` with the same event handle.

On H100, the hardware event mechanism operates at sub-microsecond granularity. After the producer kernel's last CTA retires, the event fires and all consumer SM schedulers are notified within a few hundred nanoseconds. For large fanouts (many consumer streams) the overhead is constant: one signal, N unblocks, no chain of dependencies.
