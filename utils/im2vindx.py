import numpy as np


def im2vindx(sz, ksz):
    ix, iy = np.meshgrid(np.arange(sz[1]), np.arange(sz[0]))
    nx, ny = np.meshgrid(np.arange(-(ksz[1] - 1) / 2, (ksz[1] + 1) / 2, 1),
                         np.arange(-(ksz[0] - 1) / 2, (ksz[0] + 1) / 2, 1))
    gx = np.clip(ix.reshape(-1, 1) + nx.flatten().reshape(1, -1), 0, sz[1] - 1)
    gy = np.clip(iy.reshape(-1, 1) + ny.flatten().reshape(1, -1), 0, sz[0] - 1)
    indx = (gx + gy * sz[1]).astype(int)
    return indx
