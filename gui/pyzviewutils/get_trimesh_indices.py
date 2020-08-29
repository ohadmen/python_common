import numpy as np


def get_trimesh_indices(sz):
    indx = np.arange(sz[0] * sz[1],dtype=np.int32).reshape(sz[:2])
    triA = np.stack((indx[:-1, 1:], indx[:-1, :-1], indx[1:, 1:]), axis=2).reshape(-1, 3)
    triB = np.stack((indx[:-1, :-1], indx[1:, :-1], indx[1:, 1:]), axis=2).reshape(-1, 3)
    tri = np.r_[triA, triB]
    return tri
