import numpy as np


def cpm(x):
    return np.array([[0, x[2], -x[1]],
                     [-x[2], 0, x[0]],
                     [x[1], -x[0], 0]
                     ])


def line_set_intersection_point(p: np.ndarray, n: np.ndarray):
    """
    :param p: Nx3 matrix, line origins
    :param n: Nx3 matrix, line normals
    """

    assert p.shape[1] == 3, "data dim should be 3"
    assert (np.all(n.shape == p.shape))
    b = np.cross(n, p).flatten()
    a_mat = -np.vstack([cpm(x) for x in n])
    q = np.linalg.pinv(a_mat) @ b
    e = np.sqrt(np.mean((a_mat@q-b)**2))
    return q,e
