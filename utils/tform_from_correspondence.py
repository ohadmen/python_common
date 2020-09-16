import numpy as np


def tform_from_correspondence(src: np.ndarray, dst: np.ndarray):
    assert (np.all(src.shape == dst.shape))
    msrc = np.mean(src, axis=0)
    mdst = np.mean(dst, axis=0)

    h = (src - msrc).T @ (dst - mdst)
    u, s, vt = np.linalg.svd(h)

    matR = (u @ vt).T

    if np.linalg.det(matR) < 0:
        vt[:, -1] = -1
        matR = (u @ vt).T
    tform = np.eye(4)
    tform[:3, -1] = mdst - matR @ msrc
    tform[:3, :3] = matR
    return tform
