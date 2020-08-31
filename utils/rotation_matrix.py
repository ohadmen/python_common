import numpy as np
from cv2.cv2 import Rodrigues


def cpm(v):
    # generate the cross product matrix of vector v
    m = np.array(((0, -v[2], v[1]),
                  (v[2], 0, -v[0]),
                  (-v[1], v[0], 0)))
    return m


def rotatiom_matrix(rot_vec):
    r = Rodrigues(rot_vec)[0]
    # Rodrigues' rotation formul
    # angle = np.linalg.norm(rot_vec)
    # axis = rot_vec / angle
    # r = np.cos(angle) * np.eye(3) + np.sin(angle) * cpm(axis) + (1 - np.cos(angle)) * axis * axis.transpose()
    return r
