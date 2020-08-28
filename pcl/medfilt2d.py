import numpy as np

from common.utils.im2vindx import im2vindx


def medfilt2d(xyz, ksz):
    """
    apply 2d median filter with nan removeal
    :param xyz: NxMx3 point cloud
    :param ksz: two element tuple/list, represents size of kernel, should be odd
    :return: filtered point cloud
    """
    ind = im2vindx(xyz.shape, ksz)
    xyz_filt = xyz.reshape(-1, 3).copy()
    for i in range(3):
        xyz_filt[:, i] = np.nanmedian(xyz_filt[ind, i], axis=1)
    return xyz_filt.reshape(xyz.shape)
