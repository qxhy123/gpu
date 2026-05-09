# l2_window_demo

Phase 8 demo: high-priority stream with L2 set-window protection.
Mimics H100 cudaStreamAttributeAccessPolicyWindow.

Note: Phase 8 M4 ships the API + window registration; the actual L2
eviction integration is a Phase 9 follow-up. The window registration
is observable via Stream.l2_window and L2Cache._stream_windows.
