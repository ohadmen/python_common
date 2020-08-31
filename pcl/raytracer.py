import numpy as np

from common.pcl.bresenham3d import bresenham3d


class Raytracer:

    def _ray_trimesh_intersection(self,rp, rn, v):
        u = v[1] - v[0]
        v = v[2] - v[0]
        n = np.cross(u, v)
        if np.linalg.norm(n) == 0:  # bad mesh
            return np.inf, 0
        b = n @ rn  # mesh perpendiicual to ray
        if b == 0:
            return np.inf, 0
        a = n @ (v[0] - rp)
        r = a / b
        if r < 0:  # ray go away from triangle
            return np.inf, 0
        pt = rp + rn * r
        uu = u @ u
        uv = u @ v
        vv = v @ v
        w = pt - v[0]
        wu = w @ u
        wv = w @ v
        d = uv * uv - uu * vv
        if d == 0:
            return np.inf, 0
        s = (uv * wv - vv * wu) / d
        if s < 0 or s > 1:
            return np.inf, 0
        t = (uv * wu - uu * wv) / d
        if t < 0 or (s + t) > 1:
            return np.inf, 0

        return r, b

    def _ray_vol_intersection(self, q, n):

        tlist = []
        # xy
        txy1 = -q[2] / n[2]
        pt = q + txy1 * n
        okxy1 = (0 < pt[0] < self.dim[0]) and (0 < pt[1] < self.dim[1])
        if okxy1:
            tlist.append(txy1)

        # xz
        txz1 = -q[1] / n[1]
        pt = q + txz1 * n
        oktx1 = (0 < pt[0] < self.dim[0]) and (0 < pt[2] < self.dim[2])
        if oktx1:
            tlist.append(txz1)

        # yz
        tyz1 = -q[0] / n[0]
        pt = q + tyz1 * n
        okyz1 = (0 < pt[1] < self.dim[1]) and (0 < pt[2] < self.dim[2])
        if okyz1:
            tlist.append(tyz1)

        if len(tlist) == 0:
            return None
        t = max(tlist)
        pt = q + t * n
        return pt

    def __init__(self, v, f, n=64):
        self.v = v
        self.f = f
        self.x0 = np.min(v, axis=0)
        d = np.max(v, axis=0) - self.x0
        self.sz = np.max(d) / n
        self.dim = np.ceil(d / self.sz).astype(int)
        self.vol = [[[[] for _ in range(self.dim[2])] for _ in range(self.dim[1])] for _ in range(self.dim[0])]
        vq = (v - self.x0) / self.sz
        for ii in range(f.shape[0]):
            mx = np.ceil(np.max(vq[f[ii]], axis=0)).astype(int)
            mn = np.floor(np.min(vq[f[ii]], axis=0)).astype(int)
            for x in range(mn[0], mx[0]):
                for y in range(mn[1], mx[1]):
                    for z in range(mn[2], mx[2]):
                        self.vol[x][y][z].append(ii)

    def trace(self, p, n):
        r = np.inf
        i = 0
        q1 = (p - self.x0) / self.sz
        q2 = self._ray_vol_intersection(q1, n)
        if q2 is None:
            return r, i
        q1 = q1.astype(int)
        q2 = q2.astype(int)
        qlist = bresenham3d(q1, q2)

        for q in qlist:
            if np.any(q < 0) or np.any(q >= self.dim):
                continue
            f_list = self.vol[q[0]][q[1]][q[2]]
            if len(f_list)==0:
                continue
            rr, ii = zip(*[self._ray_trimesh_intersection(p, n, self.v[self.f[f_list[j]]]) for j in range(len(f_list))])
            ind = np.argmin(rr)
            if rr[ind] != np.inf:
                r = rr[ind]
                i = ii[ind]
                break
        return r, i