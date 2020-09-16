import common.gui.pyzview as m
import numpy as np

from common.gui.pyzviewutils.get_trimesh_indices import get_trimesh_indices


class Zview:

    @staticmethod
    def _rgba2floatcol(rgba):
        rgba = np.clip(rgba, 0, 1)
        rgbafloat = (rgba * 255).astype(np.uint8)
        rgbafloat = rgbafloat.flatten()
        rgbafloat = rgbafloat.view(np.float32)
        return rgbafloat

    @staticmethod
    def _str2rgb(str):
        if str == 'r':
            col = [1, 0, 0]
        elif str == 'g':
            col = [0, 1, 0]
        elif str == 'b':
            col = [0, 0, 1]
        elif str == 'y':
            col = [1, 1, 0]
        elif str == 'w':
            col = [1, 1, 1]
        elif str == 'k':
            col = [0, 0, 0]
        elif str == 'R':
            col = list(np.random.rand(3))
        else:
            raise RuntimeError("unknown color name")
        return col

    @classmethod
    def _get_pts_arr(cls, xyz, color, alpha):
        if len(xyz.shape) == 3:
            xyz = xyz.reshape(-1, xyz.shape[2])
        n, ch = xyz.shape
        assert ch >= 3
        if ch == 3:
            xyzrgba = np.c_[xyz, np.ones((n, 4))]
        elif ch == 4:
            xyzrgba = np.c_[xyz[:, :3], xyz[:, 3:4] * [1, 1, 1], np.ones((n, 1))]
        elif ch == 6:
            xyzrgba = np.c_[xyz, np.ones((n, 1))]
        elif ch == 7:
            xyzrgba = xyz
        else:
            raise RuntimeError("unknown number of channels")

        assert (xyzrgba.shape[1] == 7)

        if alpha:
            xyzrgba[:, -1] = alpha
        if color:
            if isinstance(color, str):
                xyzrgba[:, 3:6] = np.array(cls._str2rgb(color))
            if hasattr(color, '__len__'):
                if len(color) == 3:
                    xyzrgba[:, 3:6] = np.array(color)
                if len(color) == n:
                    xyzrgba[:, 3:6] = np.array(color)
                else:
                    raise RuntimeError("unknonw color option")
        rgba = cls._rgba2floatcol(xyzrgba[:, 3:])
        return np.c_[xyz[:, :3].astype(np.float32), rgba]

    def __init__(self):
        self.zv = m.interface()  # get interface

    def _set_cam_look_at(self, e, c, u):
        e, c, u = [x.tolist() if isinstance(x, np.ndarray) else x for x in [e, c, u]]
        return self.zv.setCameraLookAt(e, c, u)

    def remove_shape(self, k):
        return self.zv.removeShape(k)

    def add_mesh(self, string, xyz, color=None, alpha=None):
        if len(xyz.shape) != 3:
            raise RuntimeError("expecting nxmxD for D>=3")
        xyzf = self._get_pts_arr(xyz, color, alpha)
        faces = get_trimesh_indices(xyz.shape)
        k = self.zv.addColoredMesh(string, xyzf, faces)
        if k==-1:
            raise RuntimeWarning("could not get response from zview app")
        return k

    def add_points(self, string, xyz, color=None, alpha=None):
        xyzf = self._get_pts_arr(xyz, color, alpha)
        k = self.zv.addColoredPoints(string, xyzf)
        return k

    def update_points(self, handle, xyz, color=None, alpha=None):
        xyzf = self._get_pts_arr(xyz, color, alpha)
        self.zv.updateColoredPoints(handle, xyzf)


    KEY_ESC = 16777216
    KEY_ENTER = 16777220

    def get_last_keystroke(self):
        return self.zv.getLastKeyStroke()