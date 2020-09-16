import unittest

import numpy as np
from cv2.cv2 import Rodrigues

from common.utils.rotation_matrix import rotation_matrix


class Test(unittest.TestCase):
    def test_line_set_intersection_point(self):

        for i in range(100):
            np.random.seed(i)
            rot_vec = np.random.randn(3)
            rot_mat = Rodrigues(rot_vec)[0]
            rot_mat_hat = rotation_matrix(rot_vec)
            err = np.linalg.norm(rot_mat-rot_mat_hat)
            if err>np.finfo(float).eps*1e3:
                self.fail()




if __name__ == "__main__":
    unittest.main()
