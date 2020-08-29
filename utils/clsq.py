import numpy as np


def clsq(mat: np.ndarray, dim: int) -> [np.ndarray, np.ndarray]:
    """
    solves the constrained least squares Problem
    A (c n)' ~ 0 subject to norm(n,2)=1
    length(n) = dim
    [c,n] = clsq(A,dim)

    solves the following problem:
    A * x = 0
    with sum(x(end-dim:end).^2)=1

    :param mat: input matrix A
    :param dim: dimension of normalised vector n
    :return: c: unnormalised part of the solution
    :return: n: normalised part of the solution
    """

    m, p = mat.shape
    k = p - dim
    if p < dim + 1:
        raise Exception('not enough unknowns')
    if m < dim:
        raise Exception('not enough equations')

    q, r = np.linalg.qr(mat)
    r_ = r[k:, k:]
    u, s, v = np.linalg.svd(r_)
    n = v[- 1, :].T
    c = np.dot(np.dot(-np.linalg.pinv(r[0:k, 0:k]), r[0:k, k:]), n)

    return c, n
