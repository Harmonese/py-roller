"""Isolated optional separator, deterministic seed and bounded CPU threads."""
import random
import sys


def main():
    import numpy as np
    import torch
    from demucs.separate import main as separate
    torch.set_num_threads(int(sys.argv[1]))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    separate(sys.argv[2:])


if __name__ == '__main__': main()
