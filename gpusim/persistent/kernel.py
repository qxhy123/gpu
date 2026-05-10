"""Phase 14: PersistentKernel — long-running kernel that pulls from queue."""
from dataclasses import dataclass


@dataclass
class PersistentKernel:
    ptx_src: str
    grid: tuple
    block: tuple
    params_template: dict
    work_queue: object
    kernel_name: str = "<persistent>"

    def start(self, config, recorder=None) -> list:
        """Block until queue is stopped + empty. Returns Result per work item."""
        from gpusim.api import Stream, synchronize
        results = []
        while True:
            item = self.work_queue.pop()
            if item is None:
                # Queue empty — break
                break
            params = {**self.params_template, **item}
            s = Stream()
            s.launch(ptx_src=self.ptx_src, grid=self.grid, block=self.block,
                     params=params, kernel_name=self.kernel_name, config=config)
            multi_res = synchronize(streams=[s], config=config)
            if s.stream_id in multi_res.streams and multi_res.streams[s.stream_id]:
                res = multi_res.streams[s.stream_id][0]
                results.append(res)
                if recorder is not None:
                    cycles = res.metrics.get("cycles", 0)
                    recorder.kernel_launch(
                        stream_id=s.stream_id,
                        kernel_name=self.kernel_name,
                        grid=self.grid, block=self.block,
                        launch_cycle=0, complete_cycle=cycles,
                        n_ctas=self.grid[0]*self.grid[1]*self.grid[2],
                        parent_kernel_id=-1, is_persistent=True,
                    )
        return results
