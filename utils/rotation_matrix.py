import numpy as np


def cpm(v):
    # generate the cross product matrix of vector v
    m = np.array(((0, -v[2], v[1]),
                  (v[2], 0, -v[0]),
                  (-v[1], v[0], 0)))
    return m


def rotation_matrix(rot_vec):
    rot_vec = np.array(rot_vec)
    angle = np.linalg.norm(rot_vec)
    axis = rot_vec / angle
    c = cpm(axis)
    r = np.eye(3) + c * np.sin(angle) + (1 - np.cos(angle)) * c @ c
    return r
