import unittest
import numpy as np

from common.utils.ellipse2d import Ellipse2d


class TestEllipse2d(unittest.TestCase):

    @staticmethod
    def _random_ellipse(seed):
        np.random.seed(seed)
        c = np.random.randn(2)
        angle = np.random.rand() * np.pi
        ll = np.random.rand(2)

        elps = Ellipse2d(c, angle, ll)
        return elps

    def test_fit(self):
        n = 100
        for i in range(1000):
            ellipse_ref = self._random_ellipse(i)
            pts = ellipse_ref.get_points(n)
            ellipse_hat, err = Ellipse2d.fit(pts)

            eps = 1e-3
            angle_diff = np.angle(np.exp(1j * (ellipse_hat.angle - ellipse_ref.angle)) ** 2) / 2
            if np.abs(angle_diff) > eps:
                self.fail()
            if np.any(np.abs(ellipse_hat.length - ellipse_ref.length) > eps):
                self.fail()
            if np.any(np.abs(ellipse_hat.center - ellipse_ref.center) > eps):
                self.fail()
            return True

    def test_dist(self):
        n = 64
        for i in range(100):
            elps = self._random_ellipse(i)
            pts = elps.get_points(n)
            d = elps.dist(pts)
            dn = np.linalg.norm(d,axis=1)
            if np.any(dn > np.finfo(float).eps * 1e6):
                self.fail()
            v = np.random.randn(*pts.shape) * 0.1
            d = elps.dist(pts + v)
            e = elps.dist(pts + v - d)
            en = np.linalg.norm(e, axis=1)
            if np.any(en > np.finfo(float).eps * 1e6):
                self.fail()


if __name__ == "__main__":
    unittest.main()
