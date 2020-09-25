import unittest

import numpy as np

from common.utils.apply_transformation import apply_transformation
from common.utils.rotation_matrix import rotation_matrix
from common.utils.tform_from_correspondence import tform_from_correspondence


def create_tform(r, t):
    tform = np.eye(4)
    tform[:3, -1] = t
    tform[:3, :3] = rotation_matrix(r)
    return tform


class Test(unittest.TestCase):
    def test_line_set_intersection_point(self):
        n_points = 1000
        np.random.seed(0)
        for noise_sigma in np.linspace(0, 0.1, 10):


            tform = create_tform(np.random.randn(3), np.random.rand(3))

            p = np.random.rand(n_points, 3)
            q = apply_transformation(tform,p)
            tform_h = tform_from_correspondence(p, q)
            err = np.linalg.norm(tform - tform_h)
            if err > noise_sigma + np.finfo(float).eps * 1e3:
                self.fail()


if __name__ == "__main__":
    unittest.main()
