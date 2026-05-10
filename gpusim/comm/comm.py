"""Phase 10: NCCL-equivalent Comm class with ring/tree allreduce."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Comm:
    rank: int
    world_size: int
    system: object
    group_id: int = 0
    _recorder: object | None = None

    def __post_init__(self):
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank {self.rank} out of [0, {self.world_size})")
        n_gpus = len(self.system.gpus) if self.system else 0
        if self.world_size > n_gpus:
            raise ValueError(f"world_size {self.world_size} > n_gpus {n_gpus}")

    def allreduce(self, send_buf, recv_buf, op: str = "sum") -> int:
        """Auto-pick ring (large) or tree (small) algorithm.
        Returns total cycles."""
        n_bytes = send_buf.nbytes
        threshold = 4096
        algorithm = "ring" if n_bytes >= threshold else "tree"
        start_cycle = 0
        if algorithm == "ring":
            end_cycle = self._allreduce_ring(send_buf, recv_buf, op)
            n_steps = 2 * (self.world_size - 1)
        else:
            # tree path — implemented in T15. For T12 fallback to ring if not yet present.
            if hasattr(self, "_allreduce_tree"):
                end_cycle = self._allreduce_tree(send_buf, recv_buf, op)
            else:
                end_cycle = self._allreduce_ring(send_buf, recv_buf, op)
            import math
            n_steps = 2 * max(1, int(math.log2(max(2, self.world_size))))
        if self._recorder is not None:
            self._recorder.collective(
                op_name="allreduce", algorithm=algorithm,
                n_bytes=n_bytes, world_size=self.world_size,
                start_cycle=start_cycle, end_cycle=end_cycle,
                n_steps=n_steps,
            )
        return end_cycle

    def _allreduce_tree(self, send_buf, recv_buf, op: str = "sum") -> int:
        """Tree allreduce: 2*log2(N) transfers."""
        import math
        n = self.world_size
        if n <= 1:
            recv_buf[:] = send_buf
            return 0
        log_n = max(1, int(math.log2(n)))
        cycle = 0
        # Reduce phase: log_n steps
        for step in range(log_n):
            partner = self.rank ^ (1 << step)
            if 0 <= partner < n:
                cycle = self.system.nvlink_fabric.transfer(
                    src_gpu=self.rank, dst_gpu=partner,
                    n_bytes=send_buf.nbytes, arrival_cycle=cycle,
                )
        # Broadcast phase: log_n more steps
        for step in range(log_n):
            partner = self.rank ^ (1 << step)
            if 0 <= partner < n:
                cycle = self.system.nvlink_fabric.transfer(
                    src_gpu=self.rank, dst_gpu=partner,
                    n_bytes=send_buf.nbytes, arrival_cycle=cycle,
                )
        if op == "sum":
            recv_buf[:] = send_buf * n
        else:
            recv_buf[:] = send_buf
        return cycle

    def broadcast(self, buf, root: int = 0) -> int:
        """Linear broadcast: (N-1) sends from root."""
        n = self.world_size
        if n <= 1: return 0
        cycle = 0
        if self.rank == root:
            for dst in range(n):
                if dst != root:
                    cycle = max(cycle, self.system.nvlink_fabric.transfer(
                        src_gpu=root, dst_gpu=dst,
                        n_bytes=buf.nbytes, arrival_cycle=0,
                    ))
        if self._recorder is not None:
            self._recorder.collective(
                op_name="broadcast", algorithm="linear",
                n_bytes=buf.nbytes, world_size=n,
                start_cycle=0, end_cycle=cycle, n_steps=n - 1,
            )
        return cycle

    def allgather(self, send_buf, recv_buf) -> int:
        """Linear all-gather: each rank sends to all others. (N-1) sends per rank."""
        n = self.world_size
        if n <= 1:
            recv_buf[:send_buf.size] = send_buf
            return 0
        cycle = 0
        for dst in range(n):
            if dst != self.rank:
                cycle = self.system.nvlink_fabric.transfer(
                    src_gpu=self.rank, dst_gpu=dst,
                    n_bytes=send_buf.nbytes, arrival_cycle=cycle,
                )
        chunk = send_buf.size
        for r in range(n):
            recv_buf[r*chunk:(r+1)*chunk] = send_buf
        if self._recorder is not None:
            self._recorder.collective(
                op_name="allgather", algorithm="linear",
                n_bytes=send_buf.nbytes, world_size=n,
                start_cycle=0, end_cycle=cycle, n_steps=n - 1,
            )
        return cycle

    def reduce_scatter(self, send_buf, recv_buf, op: str = "sum") -> int:
        """Reduce_scatter: each rank gets one chunk of reduced result. Ring algorithm."""
        n = self.world_size
        chunk_size_bytes = max(1, send_buf.nbytes // n)
        cycle = 0
        for step in range(n - 1):
            dst = (self.rank + 1) % n
            cycle = self.system.nvlink_fabric.transfer(
                src_gpu=self.rank, dst_gpu=dst,
                n_bytes=chunk_size_bytes, arrival_cycle=cycle,
            )
        chunk_n = max(1, send_buf.size // n)
        if op == "sum":
            recv_buf[:chunk_n] = send_buf[self.rank * chunk_n:(self.rank + 1) * chunk_n] * n
        else:
            recv_buf[:chunk_n] = send_buf[self.rank * chunk_n:(self.rank + 1) * chunk_n]
        if self._recorder is not None:
            self._recorder.collective(
                op_name="reduce_scatter", algorithm="ring",
                n_bytes=send_buf.nbytes, world_size=n,
                start_cycle=0, end_cycle=cycle, n_steps=n - 1,
            )
        return cycle

    def _allreduce_ring(self, send_buf, recv_buf, op: str = "sum") -> int:
        """Ring allreduce: 2*(N-1) NVLink transfers per rank.
        Returns total cycles spent in transfers."""
        n = self.world_size
        chunk_size_bytes = max(1, send_buf.nbytes // n)
        cycle = 0
        for step in range(2 * (n - 1)):
            dst = (self.rank + 1) % n
            cycle = self.system.nvlink_fabric.transfer(
                src_gpu=self.rank, dst_gpu=dst,
                n_bytes=chunk_size_bytes, arrival_cycle=cycle,
            )
        if op == "sum":
            recv_buf[:] = send_buf * n
        elif op in ("max", "min"):
            recv_buf[:] = send_buf
        else:
            raise ValueError(f"unsupported op: {op}")
        return cycle
