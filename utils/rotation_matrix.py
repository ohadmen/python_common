import numpy as np


def cpm(v):
    # generate the cross product matrix of vector v
    m = np.array(((0, -v[2], v[1]),
                  (v[2], 0, -v[0]),
                  (-v[1], v[0], 0)))
    return m


def rotation_matrix(rot_vec):
    rot_vec = np.asarray(rot_vec)
    if rot_vec.shape == (3, ):
        angle = np.linalg.norm(rot_vec)
        if angle == 0:
            return np.eye(3)
        axis = rot_vec / angle
        c = cpm(axis)
        r = np.eye(3) + c * np.sin(angle) + (1 - np.cos(angle)) * c @ c
    elif rot_vec.shape == (3,3):
        g, v = np.linalg.eig(rot_vec)
        axis = np.real(v[:, np.argmin(np.abs(g - 1))])
        angle = np.arccos((np.trace(rot_vec) - 1) * 0.5)
        r = axis * angle
    else:
        raise RuntimeWarning("unknown input shape")
    return r
