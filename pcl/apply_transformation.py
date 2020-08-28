import numpy as np


def apply_transformation(t: np.ndarray, pcl: np.ndarray):
    if t.shape == (4, 4):
        dim = 3
    elif t.shape == (3, 3):
        dim = 2
    else:
        raise RuntimeError("Bad transformation size {}".format(t.shape))
    pcl=pcl.copy()
    if len(pcl.shape) == 2:
        if pcl.shape[1] != dim:
            raise RuntimeError("input size is not consistent with transformation")
        pcl_out = (np.c_[pcl, np.ones((pcl.shape[0], 1))] @ t.T)[:, :dim]

    elif len(pcl.shape) == 3:
        if pcl.shape[2] != dim:
            raise RuntimeError("input size is not consistent with transformation")
        pcl_out = ((np.c_[pcl.reshape(-1, dim), np.ones((pcl.shape[0] * pcl.shape[1], 1))] @ t.T)[:, :dim]).reshape(
            pcl.shape)
    else:
        raise RuntimeError("Unknown input size")
    return pcl_out
