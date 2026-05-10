"""Phase 14: dynamic parallelism — parent kernel launches child."""

_pending_child_launches: list = []


def device_launch(parent_kernel_id: int, ptx_src: str, grid: tuple, block: tuple,
                  params: dict, *, kernel_name: str = "<child>") -> None:
    """Schedule a child kernel launch from a parent. Phase 14."""
    _pending_child_launches.append({
        "parent_kernel_id": parent_kernel_id,
        "ptx_src": ptx_src, "grid": grid, "block": block,
        "params": params, "kernel_name": kernel_name,
    })


def drain_pending_child_launches(config, recorder=None) -> list:
    """Process all pending child launches; return list of Results."""
    from gpusim.api import Stream, synchronize
    results = []
    while _pending_child_launches:
        item = _pending_child_launches.pop(0)
        s = Stream()
        s.launch(ptx_src=item["ptx_src"], grid=item["grid"], block=item["block"],
                 params=item["params"], kernel_name=item["kernel_name"],
                 config=config)
        multi_res = synchronize(streams=[s], config=config)
        if s.stream_id in multi_res.streams and multi_res.streams[s.stream_id]:
            res = multi_res.streams[s.stream_id][0]
            results.append(res)
            if recorder is not None:
                recorder.kernel_launch(
                    stream_id=s.stream_id,
                    kernel_name=item["kernel_name"],
                    grid=item["grid"], block=item["block"],
                    launch_cycle=0, complete_cycle=res.metrics.get("cycles", 0),
                    n_ctas=item["grid"][0] * item["grid"][1] * item["grid"][2],
                    parent_kernel_id=item["parent_kernel_id"],
                    is_persistent=False,
                )
    return results


def reset_pending_child_launches() -> None:
    """Test helper — clears the pending list (test hygiene)."""
    _pending_child_launches.clear()
