"""Phase 12: PyTorch-distributed-equivalent adapter."""
from __future__ import annotations
import numpy as np


_system = None
_comm = None
_world_size = 0
_rank = 0


def init_process_group(world_size: int, rank: int, n_gpus: int = None,
                         config=None) -> None:
    global _system, _comm, _world_size, _rank
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.comm.comm import Comm
    from gpusim.config.loader import load_default
    if config is None:
        config = load_default()
    if n_gpus is None:
        n_gpus = world_size
    config.n_gpus = n_gpus
    _system = MultiGpuSystem.from_config(config)
    _comm = Comm(rank=rank, world_size=world_size, system=_system)
    _world_size = world_size
    _rank = rank


def destroy_process_group() -> None:
    global _system, _comm, _world_size, _rank
    _system = None
    _comm = None
    _world_size = 0
    _rank = 0


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def _to_numpy(t):
    if isinstance(t, np.ndarray):
        return t
    try:
        import torch
        if isinstance(t, torch.Tensor):
            return t.numpy()
    except ImportError:
        pass
    raise TypeError(f"expected numpy.ndarray or torch.Tensor, got {type(t)}")


def _copy_back(t, arr):
    if isinstance(t, np.ndarray):
        t[:] = arr
        return
    try:
        import torch
        if isinstance(t, torch.Tensor):
            t.copy_(torch.from_numpy(arr))
            return
    except ImportError:
        pass


def barrier() -> None:
    """No-op in simulator (single-process)."""
    pass


def all_reduce(tensor, op: str = "sum") -> None:
    arr = _to_numpy(tensor)
    recv = np.empty_like(arr)
    _comm.allreduce(arr, recv, op=op)
    _copy_back(tensor, recv)


def all_gather(tensor_list, tensor) -> None:
    arr = _to_numpy(tensor)
    recv = np.empty(arr.size * _world_size, dtype=arr.dtype)
    _comm.allgather(arr, recv)
    chunk = arr.size
    for i, t in enumerate(tensor_list):
        _copy_back(t, recv[i*chunk:(i+1)*chunk].reshape(arr.shape))


def reduce_scatter(output, input_list, op: str = "sum") -> None:
    arrs = [_to_numpy(t) for t in input_list]
    full = np.concatenate(arrs)
    recv = np.empty_like(_to_numpy(output))
    _comm.reduce_scatter(full, recv, op=op)
    _copy_back(output, recv)


def broadcast(tensor, src: int = 0) -> None:
    arr = _to_numpy(tensor)
    _comm.broadcast(arr, root=src)
    _copy_back(tensor, arr)


def send(tensor, dst: int) -> None:
    arr = _to_numpy(tensor)
    _comm.send(arr, dst_rank=dst)


def recv(tensor, src: int) -> None:
    arr = _to_numpy(tensor)
    _comm.recv(arr, src_rank=src)
