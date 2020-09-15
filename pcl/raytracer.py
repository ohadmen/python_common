import numpy as np



from .bresenham3d import bresenham3d


# zv = zview.interface()
class Raytracer:

    def _ray_trimesh_intersection(self, rp, rn, triv):
        u = triv[1] - triv[0]
        v = triv[2] - triv[0]
        n = np.cross(u, v)
        if np.linalg.norm(n) == 0:  # bad mesh
            return np.inf, 0
        b = n @ rn  # mesh perpendiicual to ray
        if b == 0:
            return np.inf, 0
        a = n @ (triv[0] - rp)
        r = a / b
        if r < 0:  # ray go away from triangle
            return np.inf, 0
        pt = rp + rn * r
        uu = u @ u
        uv = u @ v
        vv = v @ v
        w = pt - triv[0]
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
        intensity = max(0,-b / np.linalg.norm(n))
        return r, intensity

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
        d = (np.max(v, axis=0) - self.x0)
        d *= 1.001  # liast point gets into last bin
        self.sz = np.max(d) / n
        self.dim = np.ceil(d / self.sz).astype(int)
        self.vol = [[[[] for _ in range(self.dim[2])] for _ in range(self.dim[1])] for _ in range(self.dim[0])]
        vq = (v - self.x0) / self.sz
        for ii in range(f.shape[0]):
            mn = np.floor(np.min(vq[f[ii]], axis=0)).astype(int)
            mx = np.floor(np.max(vq[f[ii]], axis=0)).astype(int)

            for x in range(mn[0], mx[0] + 1):
                for y in range(mn[1], mx[1] + 1):
                    for z in range(mn[2], mx[2] + 1):
                        self.vol[x][y][z].append(ii)

        # for x in range(self.dim[0]):
        #     for y in range(self.dim[1]):
        #         for z in range(self.dim[2]):
        #             zv.addColoredMesh("cube{}{}{}".format(x,y,z), *add_rectangle(t=np.array([x, y, z]) * self.sz, s=(self.sz, self.sz, self.sz),c=(1,1,0,0.2)))
        #             zv.addMesh("mesh{}{}{}".format(x,y,z),self.v,self.f[self.vol[x][y][z]])

    def trace(self, p, n):

        # zv.addEdges("ray", np.c_[p, p + n].T.copy(), np.array([[0, 1]]))
        r = np.inf
        i = 0
        q1 = (p - self.x0) / self.sz
        q2 = self._ray_vol_intersection(q1, n)
        if q2 is None:
            return r, i

        qlist = bresenham3d(q1, q2)
        qlist = np.array(qlist)
        qlist = (qlist[1:]+qlist[:-1])/2
        qlist = qlist.astype(int)
        checked_faces = set()
        for q in qlist:
            if np.any(q < 0) or np.any(q >= self.dim):
                continue
            f_list = self.vol[q[0]][q[1]][q[2]]
            f_list = [x for x in f_list if x not in checked_faces] #remove check
            if len(f_list) == 0:
                continue
            # zv.addColoredMesh('mesh', addColor(self.v, 'r'), self.f[f_list])
            # zv.addColoredMesh("cube", *add_rectangle(t=q * self.sz, s=(self.sz, self.sz, self.sz), c=[1, 0, 0, 0.2]))
            rr, ii = zip(*[self._ray_trimesh_intersection(p, n, self.v[self.f[ii]]) for ii in f_list ])
            ind = np.argmin(rr)
            if rr[ind] != np.inf:
                r = rr[ind]
                i = ii[ind]
                break
            [checked_faces.add(x) for x in f_list]
        return r, i
