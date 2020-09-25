import unittest
import numpy as np

from common.utils.ellipse2d import Ellipse2d


class TestEllipse2d(unittest.TestCase):
    def test_fit(self):
        n = 100
        for i in range(1000):
            np.random.seed(i)
            c = np.random.randn(2)
            angle = np.random.rand() * np.pi
            ll = np.random.rand(2)
            ellipse_ref = Ellipse2d(c, angle, ll)
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


if __name__ == "__main__":
    unittest.main()
