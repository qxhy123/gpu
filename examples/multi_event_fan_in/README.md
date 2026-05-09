# multi_event_fan_in

Phase 9 demo: 2 producers (s_a, s_b) → 1 consumer (s_c) using
s_c.wait_all([ev_a, ev_b]). Demonstrates multi-event fan-in pattern.
