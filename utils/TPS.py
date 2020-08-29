# written by Ohad Menashe Mar.2013 (Python port on Sep.2018)
# ==============================
# K-dimnetional Thin plate spline object
# usage:
#   construction:
#        t=TPS_kd(source_points,dest_points)
#           -source_points - Nxdim source points
#           -dest_points - Nxdim destination points
#   apply:
#       t.at(pts)
#           -pts - Mxdim points to transform
#       t.inv() - inverse transformation

import numpy as np


class TPS:
    def __init__(self, src, dst):
        self.src = src.astype(np.float)
        self.dst = dst.astype(np.float)
        self.mat = np.array([])
        self.priv_calc_i_mat()

    def priv_calc_i_mat(self):
        n = self.src.shape[0]
        d_in = self.src.shape[1]
        d_ot = self.dst.shape[1]

        assert n == self.dst.shape[0], ' src and dst should have the same number of points'

        k = TPS.priv_k(self.src, self.src)
        p = np.concatenate((np.ones((n, 1)), self.src), axis=1)
        w = np.concatenate(
            (np.concatenate((k, p), axis=1), np.concatenate((p.T, np.zeros((d_in + 1, d_in + 1))), axis=1)), axis=0)
        self.mat = np.matmul(np.linalg.pinv(w), np.concatenate((self.dst, np.zeros((d_in + 1, d_ot))), axis=0))

    @staticmethod
    def priv_k(a, b):
        dim = a.shape[1]
        assert (dim == b.shape[1])
        a_ = np.transpose(np.expand_dims(a, axis=2), (0, 2, 1))
        b_ = np.transpose(np.expand_dims(b, axis=2), (0, 2, 1))
        k = (np.transpose(a_, (1, 0, 2)) - b_)
        k = k ** 2
        k = np.sum(k, axis=2)
        k = np.maximum(k, np.finfo(np.float).eps * 1e6)
        k = TPS.priv_ufunc(k, dim)
        return k

    @staticmethod
    def priv_ufunc(nrm2, dim):
        if dim == 1:
            d = nrm2 * np.sqrt(nrm2)
        elif dim == 2:
            d = nrm2 * np.log(nrm2)
        elif dim == 3:
            d = np.sqrt(nrm2)
        else:
            d = np.power(nrm2, 1 - dim / 2)
        return d

    def at(self, pt_in):
        MAX_SET_SIZE = 1000
        assert pt_in.shape[1] == self.src.shape[1], 'Input points dim is different than TPS dim'
        data_n = pt_in.shape[0]

        if data_n > MAX_SET_SIZE:
            pt_out = np.zeros((pt_in.shape[0], self.dst.shape[1]))
            for i0 in range(0, data_n, MAX_SET_SIZE):
                i1 = np.min((data_n, i0 + MAX_SET_SIZE))
                pt_out[i0:i1, :] = self.at(pt_in[i0:i1])
            return pt_out

        k = TPS.priv_k(pt_in, self.src)
        p = np.concatenate((np.ones((data_n, 1)), pt_in), axis=1)
        z = np.matmul(np.concatenate((k.T, p), axis=1), self.mat)
        return z

    def add(self, src, dst):
        self.src = np.concatenate((self.src, src), axis=0)
        self.dst = np.concatenate((self.dst, dst), axis=0)
        self.priv_calc_i_mat()

    def inv(self):
        tps_inv = TPS(self.dst, self.src)
        return tps_inv

    def __mul__(self, other):
        src_a = self.src
        src_b = self.inv().at(other.src)

        dst_a = other.at(self.dst)
        dst_b = other.dst

        tps_c = TPS(np.concatenate((src_a, src_b), axis=0), np.concatenate((dst_a, dst_b), axis=0))
        return tps_c
