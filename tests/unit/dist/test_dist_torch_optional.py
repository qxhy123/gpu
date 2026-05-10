def test_torch_tensor_path_if_available():
    """If torch is installed, dist API should accept torch tensors."""
    try:
        import torch
    except ImportError:
        import pytest
        pytest.skip("torch not installed — skipping torch-tensor path")
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    t = torch.full((16,), 1.0, dtype=torch.float32)
    dist.all_reduce(t, op="sum")
    expected = torch.full((16,), 4.0, dtype=torch.float32)
    assert torch.allclose(t, expected)
    dist.destroy_process_group()


def test_invalid_input_type_raises():
    """Non-numpy non-torch input should raise TypeError."""
    import gpusim.dist as dist
    import pytest
    dist.init_process_group(world_size=4, rank=0)
    with pytest.raises(TypeError):
        dist.all_reduce([1, 2, 3], op="sum")   # plain Python list
    dist.destroy_process_group()
