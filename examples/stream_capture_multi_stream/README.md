# stream_capture_multi_stream — Phase 15

Two streams capture into a shared `CaptureSession`. Cross-stream `record`/`wait`
becomes a Graph edge.

Stream A: `k1 → record(ev) → k3`
Stream B: `wait(ev) → k2`

Demonstrates:
- `CaptureSession` shared across streams (M2)
- Cross-stream event sync becomes a graph dependency edge automatically

## Run
```bash
python run.py
```
