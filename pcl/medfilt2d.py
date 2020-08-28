import numpy as np

from common.utils.im2vindx import im2vindx


def medfilt2d(xyz, ksz):
    ind = im2vindx(xyz.shape, ksz)
    xyz_filt = xyz.reshape(-1, 3).copy()
    for i in range(3):
        xyz_filt[:, i] = np.nanmedian(xyz_filt[ind, i], axis=1)
    return xyz_filt.reshape(xyz.shape)
