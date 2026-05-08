"""wgmma async queue + warp-group functional execution."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from gpusim.core.exec import WarpFnState
from gpusim.frontend.ir import RegGroup, PtxType


@dataclass
class InflightWgmma:
    issued_at: int
    completion_at: int
    dst_regs: tuple[tuple[str, ...], ...]   # 4 warps × N regs each
    commit_group_id: int = -1


@dataclass
class WgmmaQueue:
    capacity: int = 16
    in_flight: list[InflightWgmma] = field(default_factory=list)
    committed_groups: list[int] = field(default_factory=list)
    next_group_id: int = 0

    def try_push(self, f: InflightWgmma) -> bool:
        if len(self.in_flight) >= self.capacity:
            return False
        self.in_flight.append(f)
        return True

    def commit_group(self) -> int:
        gid = self.next_group_id
        self.next_group_id += 1
        for f in self.in_flight:
            if f.commit_group_id < 0:
                f.commit_group_id = gid
        self.committed_groups.append(gid)
        return gid

    def must_wait(self, target_n: int) -> bool:
        return len(self.committed_groups) > target_n

    def drain_completed_groups(self, now: int) -> list[int]:
        """Drain committed groups whose all in_flight wgmmas have completed.
        Returns drained group ids; mutates committed_groups + in_flight."""
        drained: list[int] = []
        # process committed groups in order — must drain oldest first (FIFO)
        while self.committed_groups:
            gid = self.committed_groups[0]
            in_group = [f for f in self.in_flight if f.commit_group_id == gid]
            if not all(f.completion_at <= now for f in in_group):
                break
            drained.append(gid)
            self.in_flight = [f for f in self.in_flight if f.commit_group_id != gid]
            self.committed_groups.pop(0)
        return drained


# Functional execution function added in T17
from gpusim.core.tensor_core.precision import cast_array
from gpusim.core.tensor_core.mma_spec import MmaSpec


def execute_wgmma_for_group(
    *, spec: MmaSpec, warps: list[WarpFnState],
    a_smem_array: np.ndarray, b_smem_array: np.ndarray,
    dst_per_warp: tuple[RegGroup, ...], c_per_warp: tuple[RegGroup, ...],
) -> None:
    """Functionally execute one wgmma. Reads A from smem (M×K matrix) and
    B from smem (K×N matrix) directly as ndarrays (caller resolves descriptors
    to ndarrays). Distributes D into 4 warps × 32 lanes × N regs per spec §4.2.

    Layout (fictional, spec §11):
        warp w, lane i, reg j -> D[w*16 + i/2][(i%2)*(N/2) + j]
    """
    M, N, K = spec.m, spec.n, spec.k
    A_typed = cast_array(a_smem_array.astype(np.float32) if a_smem_array.dtype != np.float32
                          else a_smem_array.copy(), src=PtxType.f32, dst=spec.dtype_a)
    B_typed = cast_array(b_smem_array.astype(np.float32) if b_smem_array.dtype != np.float32
                          else b_smem_array.copy(), src=PtxType.f32, dst=spec.dtype_b)

    # Collect C from 4 warps × 32 lanes × N_REGS regs
    n_regs_per_lane = N // 2
    half_N = N // 2
    C = np.zeros((M, N), dtype=np.float32)
    for warp_w in range(4):
        for lane in range(32):
            row = warp_w * 16 + lane // 2
            col_base = (lane % 2) * half_N
            for j in range(n_regs_per_lane):
                reg = c_per_warp[warp_w].regs[j].name
                C[row, col_base + j] = warps[warp_w].threads[lane].get_f32(reg)

    # Compute D = A @ B + C (accumulate in f32)
    D = (A_typed.astype(np.float32) @ B_typed.astype(np.float32)) + C
    D_typed = cast_array(D, src=PtxType.f32, dst=spec.dtype_d)

    # Distribute D to dst regs (4 warps × 32 lanes × N_REGS)
    D_f32 = D_typed.astype(np.float32)
    for warp_w in range(4):
        for lane in range(32):
            row = warp_w * 16 + lane // 2
            col_base = (lane % 2) * half_N
            for j in range(n_regs_per_lane):
                reg = dst_per_warp[warp_w].regs[j].name
                warps[warp_w].threads[lane].set_f32(reg, float(D_f32[row, col_base + j]))
