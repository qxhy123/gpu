def test_pytorch_dist_simple_correctness():
    """Use gpusim.dist API: init → all_reduce → barrier."""
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)

    # Each rank's "loss"
    loss = np.full(8, 1.0, dtype=np.float32)
    dist.all_reduce(loss, op="sum")
    np.testing.assert_array_equal(loss, np.full(8, 4.0, dtype=np.float32))

    dist.barrier()
    dist.destroy_process_group()
