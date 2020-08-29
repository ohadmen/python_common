import numpy as np


def dxdy_ker(n=7):
    nn = np.arange(-(n - 1) / 2, (n + 1) / 2, 1)
    gx, gy = np.meshgrid(nn, nn)
    kx = gx / (gx * gx + gy * gy + np.finfo(float).eps)
    ky = gy / (gx * gx + gy * gy + np.finfo(float).eps)
    return kx, ky
