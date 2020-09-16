import unittest

import numpy as np

from common.utils.line_set_intersection_point import line_set_intersection_point


class Test(unittest.TestCase):
    def test_line_set_intersection_point(self):
        n_points = 100
        for noise_sigma in np.linspace(0,0.1,10):
            q = np.random.rand(3)
            p = np.random.rand(n_points,3)
            n = p-q
            n = n/np.linalg.norm(n,axis=1).reshape(-1,1)
            p =  p + np.random.randn(*p.shape)*noise_sigma
            q_hat,e_hat =line_set_intersection_point(p,n)
            err = np.linalg.norm(q_hat-q)
            if err>noise_sigma+np.finfo(float).eps*1e3:
                self.fail()




if __name__ == "__main__":
    unittest.main()
