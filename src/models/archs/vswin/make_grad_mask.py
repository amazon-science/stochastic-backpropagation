import numpy as np


def make_block_checkboard(D, H, W, block_size, downsample):
    h = H // block_size
    w = W // block_size
    checkboard = (np.indices((D, h, w)).sum(0) % downsample == 0).astype(np.float32)
    tar_board = np.zeros((D, H, W), dtype=np.float32)
    for i in range(D):
        for j in range(h):
            for k in range(w):
                tar_board[
                    i,
                    j * block_size : (j + 1) * block_size,
                    k * block_size : (k + 1) * block_size,
                ] = checkboard[i, j, k]
    return tar_board
