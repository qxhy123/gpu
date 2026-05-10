import numpy as np
import gpusim.dist as dist


def main():
    dist.init_process_group(world_size=4, rank=0)
    loss = np.full(8, 1.0, dtype=np.float32)
    print(f"Before all_reduce: loss[0:4] = {list(loss[0:4])}")
    dist.all_reduce(loss, op="sum")
    print(f"After all_reduce:  loss[0:4] = {list(loss[0:4])} (expected [4.0]*4)")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
