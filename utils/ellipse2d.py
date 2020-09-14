import numpy as np


class Ellipse2d:

    def __init__(self, center=np.zeros(2), angle: float = 0.0, length=np.ones(2)):
        self.center = center
        self.angle = angle
        o = np.argsort(-length)
        self.length = length[o]

    def get_points(self, n=64):
        axes = np.array([[np.cos(self.angle), -np.sin(self.angle)],
                         [np.sin(self.angle), np.cos(self.angle)]]) * self.length.reshape(1, 2)
        t = np.linspace(0, 2 * np.pi, n)
        xy = np.c_[np.cos(t), np.sin(t)]
        pts = xy @ axes.T + self.center.reshape(1, 2)
        return pts
    def __repr__(self):
        return "a:{} l0={} l1={} cx={} cy={}".format(self.angle,self.length[0],self.length[1],self.center[0],self.center[1])
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
