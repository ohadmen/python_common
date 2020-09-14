import unittest

import numpy as np
from cv2.cv2 import Rodrigues

from common.utils.rotation_matrix_from_to import rotation_matrix_from_to


class Test(unittest.TestCase):
    @staticmethod
    def random_normal():
        x = np.random.randn(3)
        x = x / np.linalg.norm(x)
        return x

    def test_rotation_matrix_from_to(self):

        for i in range(100):
            np.random.seed(i)
            src = self.random_normal()
            dst = self.random_normal()

            r = rotation_matrix_from_to(src, dst)
            dst_h = r @ src
            err = np.linalg.norm(dst - dst_h)
            if err > np.finfo(float).eps * 1e3:
                self.fail()


if __name__ == "__main__":
    unittest.main()
