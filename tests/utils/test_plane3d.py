import unittest
import numpy as np

from common.utils.plane3d import Plane3d
from common.utils.rotation_matrix_from_to import rotation_matrix_from_to


class TestEllipse2d(unittest.TestCase):
    def test_fit(self):
        n = 100
        for i in range(1000):
            np.random.seed(i)
            nrml = np.random.randn(3)
            nrml = nrml / np.linalg.norm(nrml)
            p = nrml * np.random.randn()
            pts = np.c_[np.random.randn(n, 2), np.zeros(n)]
            pts = pts @ rotation_matrix_from_to(np.array((0, 0, 1)), nrml).T + p
            plane_hat, err = Plane3d.fit(pts)

            eps = 1e-3
            if np.abs(plane_hat.normal.T @ nrml - 1) > eps:
                self.fail()
            if np.any(np.abs(plane_hat.dist(pts)) > eps):
                self.fail()
            return True


if __name__ == "__main__":
    unittest.main()
