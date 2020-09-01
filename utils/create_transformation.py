import numpy as np

from common.utils.rotation_matrix import rotatiom_matrix


def create_transformation(rot_vec, tran_vec, scale_vec=None):
    dim = len(tran_vec)
    if scale_vec is None:
        scale_vec = np.ones(dim)
    assert len(rot_vec) == dim
    t = np.eye(dim + 1)

    if dim == 2:
        c, s = np.cos(rot_vec), np.sin(rot_vec)
        t[:dim, :dim] = np.array(((c, -s), (s, c)))
    elif dim == 3:
        assert len(rot_vec) == dim
        t[:dim, :dim] = rotatiom_matrix(rot_vec)
    else:
        raise RuntimeError("unknown input dim {dim}")
    t[:dim, :dim] *= scale_vec
    t[:dim, -1] = tran_vec
    return t
