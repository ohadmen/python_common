import numpy as np

from ..apply_colormap import apply_colormap
from .add_color import add_color
from .get_trimesh_indices import get_trimesh_indices


def add_mesh(xyz, col=None):
    assert len(xyz.shape) == 3
    r, c, d = xyz.shape
    if col:
        xyzc = add_color(xyz, col)
    else:
        if d == 3:
            xyzc = add_color(xyz, 'r')
        elif d == 4:
            rgb = apply_colormap('jet', xyz[:, :, 3:])
            xyzc = np.concatenate([xyz[:, :, :3], rgb])
        elif d == 6:
            xyzc = add_color(xyz[:, :, :3], xyz[:, :, 3:])
        elif d == 7:
            xyzc = add_color(xyz[:, :, :3], xyz[:, :, 3:])
        else:
            raise RuntimeError("unknown input dimension {}".format(d))

    f = get_trimesh_indices([r, c])
    v = xyzc.reshape(-1, 4)
    return v, f
