import numpy as np

from .clsq import clsq
from .rotation_matrix_from_to import rotation_matrix_from_to


class Plane3d:
    def __init__(self, normal=np.array([0, 0, 1]), point=np.zeros(3)):
        self.normal = normal
        self.d = -normal.T @ point

    def get_points(self, s=1):
        v = np.array([[-1, -1, 0],
                      [-1, +1, 0],
                      [+1, +1, 0],
                      [+1, -1, 0]])
        f = np.array([[0, 2, 1],
                      [0, 3, 3]])

        rr = rotation_matrix_from_to(np.array((0, 0, 1)), self.normal)
        v = v @ rr.T + self.normal * self.d

        return v, f

    def __repr__(self):
        return "n=({},{},{}), d = {}".format(*self.normal, self.d)

    def project(self, x):
        e = self.dist(x)
        xp = x - e.reshape(-1,1)*self.normal
        return xp

    def dist(self, x):
        d = x @ self.normal + self.d
        return d

    @staticmethod
    def fit(xyz: np.ndarray):
        a = np.c_[np.ones(xyz.shape[0]), xyz]
        th = np.r_[clsq(a, 3)]
        err = a @ th
        return Plane3d(th[1:], -th[1:] * th[0]), err
