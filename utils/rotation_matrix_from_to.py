import numpy as np

from .rotation_matrix import rotation_matrix


def rotation_matrix_from_to(src: np.ndarray, dst: np.ndarray):
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    ax = np.cross(src,dst)
    asin = np.linalg.norm(ax)
    acos = src@dst
    angle = np.arctan2(asin,acos)
    rot_vec = ax/asin*angle
    rmat = rotation_matrix(rot_vec)
    return rmat

