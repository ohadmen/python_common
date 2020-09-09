import numpy as np

from common.gui.pyzviewutils.rgba2floatcol import rgba2floatcol


def colvec2rgba(n,col):
    colmat = np.ones((n,len(col)))
    col = np.array(col).reshape(1,1,-1)
    colmat = colmat*col
    return colmat

def add_color(xyz, col):
    xyz = xyz.astype(np.float32)
    if isinstance(col, str):
        if col == 'r':
            col = [1, 0, 0]
        elif col == 'g':
            col = [0, 1, 0]
        elif col == 'b':
            col = [0, 0, 1]
        elif col == 'y':
            col = [1, 1, 0]
        elif col == 'w':
            col = [1, 1, 1]
        elif col == 'k':
            col = [0, 0, 0]
        else:
            raise RuntimeError("unknown color name")
        col = colvec2rgba(xyz.shape[0],col)
        pass
    elif len(col) == 3 or len(col) == 4:
        col = colvec2rgba(xyz.shape[0],col)
    elif isinstance(col, np.ndarray) and col.shape[2] == 3:
        col=np.concatenate([col,np.ones((col.shape[0],col.shape[1],1))],axis=2)
    elif isinstance(col, np.ndarray) and col.shape[2] == 4:
        pass
    else:
        raise RuntimeError("unknown color option")
    rgba = rgba2floatcol(col)
    xyzc = np.c_[xyz.reshape(-1, 3), rgba.reshape(-1, 1)]
    return xyzc
