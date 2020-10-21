import numpy as np


def tform_from_correspondence(src: np.ndarray, dst: np.ndarray):
    assert (np.all(src.shape == dst.shape))
    msrc = np.mean(src, axis=0)
    mdst = np.mean(dst, axis=0)

    h = (src - msrc).T @ (dst - mdst)
    v, s, w = np.linalg.svd(h)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0
    if d:
        s[-1] *= -1
        v[:, -1] *= -1
    matR = (v @ w).T

    tform = np.eye(4)
    tform[:3, -1] = mdst - matR @ msrc
    tform[:3, :3] = matR
    return tform
