import numpy as np


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
        # source: https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/
        p_ = p - self.center
        axes = self._get_axes()
        p_ = p_ @ axes
        nz = np.zeros(p.shape[0])
        px = p_[:, 0]
        py = p_[:, 1]

        is_inside = nz == 1
        t = np.pi / 4 + nz
        t[~is_inside] = np.arctan2(py, px)
        a = self.length[0]
        b = self.length[1]

        while True:
            x = a * np.cos(t)
            y = b * np.sin(t)
            # Centre of curvature at current estimate
            ex = (a * a - b * b) * np.cos(t) ** 3 / a
            ey = (b * b - a * a) * np.sin(t) ** 3 / b

            rx = x - ex
            ry = y - ey

            qx = px - ex
            qy = py - ey

            r = np.hypot(ry, rx)
            q = np.hypot(qy, qx)

            delta_c = r * np.arcsin((rx * qy - ry * qx) / (r * q))
            delta_t = delta_c / np.sqrt(a ** 2 + b ** 2 - x ** 2 - y ** 2)
            if np.all(np.abs(delta_t) < eps):
                break
            t += delta_t
            # t = np.clip(0, np.pi / 2, t)

        # q = np.c_[np.copysign(x, p[:, 0]), np.copysign(y, p[:, 1])]
        q = np.c_[x, y]
        q = q @ axes.T + self.center
        d = p - q
        return d
