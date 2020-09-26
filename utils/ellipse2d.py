import numpy as np
import matplotlib.pyplot as plt


class Ellipse2d:
    def _get_axes(self):
        return np.array([[np.cos(self.angle), -np.sin(self.angle)],
                         [np.sin(self.angle), np.cos(self.angle)]])

    def __init__(self, center=np.zeros(2), angle: float = 0.0, length=np.ones(2)):
        self.center = np.array(center)
        self.angle = angle
        length = np.array(length)
        o = np.argsort(-length)
        self.length = length[o]

    def get_points(self, n=64):
        axes = self._get_axes() * self.length.reshape(1, 2)
        t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        xy = np.c_[np.cos(t), np.sin(t)]
        pts = xy @ axes.T + self.center.reshape(1, 2)
        return pts

    def __repr__(self):
        return "a:{} l0={} l1={} cx={} cy={}".format(self.angle, self.length[0], self.length[1], self.center[0],
                                                     self.center[1])

    @staticmethod
    def fit(xy: np.ndarray):
        assert len(xy.shape) == 2
        assert xy.shape[1] == 2
        mx = np.mean(xy, axis=0)
        xy = xy - mx
        x = xy[:, 0]
        y = xy[:, 1]
        mat = np.c_[x ** 2, 2 * x * y, y ** 2, -2 * x, -2 * y]
        th = np.linalg.inv(mat.T @ mat) @ mat.T @ np.ones_like(x)
        sigma = np.array([th[0:2], th[1:3]])
        err = mat @ th - 1
        c = np.linalg.inv(sigma) @ th[3:] + mx
        g, axes = np.linalg.eig(sigma)

        length = np.sqrt(1 / g)
        o = np.argsort(-length)
        length = length[o]
        main_axis = axes[:, o[0]]
        angle = np.arctan2(main_axis[1], main_axis[0])
        return Ellipse2d(c, angle, length), err

    def dist(self, p, eps: float = np.finfo(float).eps * 1e3):
        n_max_itr = 1000
        # source: https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/
        p_ = p - self.center
        axes = self._get_axes()
        p_ = p_ @ axes

        t = np.arctan2(np.sign(p_[:, 1]), np.sign(p_[:, 0]))
        a = self.length[0]
        b = self.length[1]
        det = a * a - b * b
        i = 0

        # plt.plot(*((self.get_points(64) - self.center) @ axes).T)
        # plt.plot(*p_.T,'+r')

        while True:
            ct, st = np.cos(t), np.sin(t)
            xy = np.c_[a * ct, b * st]

            # plt.plot(*xy[32].T, '*b')

            # Centre of curvature at current estimate
            ee = np.c_[ct ** 3 / a, -st ** 3 / b] * det

            rv = xy - ee
            qv = p_ - ee

            rn = np.linalg.norm(rv, axis=1)
            qn = np.linalg.norm(qv, axis=1)
            delta_c = rn * np.arcsin(np.cross(rv, qv) / (rn * qn))

            delta_t = delta_c / np.sqrt((a * st) ** 2 + (b * ct) ** 2)
            # to increase numerical stability decrease step size
            delta_t = delta_t *0.25

            if np.all(np.abs(delta_t) < eps):
                break
            t += delta_t

            if i == n_max_itr:
                raise RuntimeError("ellispe distance could not converge!")
            else:
                i = i + 1

        d = (p_ - xy) @ axes.T
        return d
