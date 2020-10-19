import numpy as np


def interp2(xg, yg, v, xn, yn):

    xg_0 = xg[0, 0]
    xg_d = xg[0, 1]-xg[0, 0]

    yg_0 = yg[0, 0]
    yg_d = yg[1, 0]-yg[0, 0]

    # check that x is griddata
    assert (np.all(np.abs(np.diff(xg, axis=1) - xg_d) < np.finfo(float).eps * 1e6))
    assert(np.all(np.abs(xg[:,0]-xg_0)<np.finfo(float).eps*1e3))

    # check that y is griddata
    assert (np.all(np.abs(np.diff(yg, axis=0) - yg_d) < np.finfo(float).eps * 1e6))
    assert (np.all(np.abs(yg[0, :] - yg_0) < np.finfo(float).eps * 1e3))

    x_ = (xn-xg_0)/xg_d
    y_ = (yn-yg_0)/yg_d

    out_of_roi = (x_ < 0) + (y_ < 0) +  (x_ > v.shape[1]-1) + (y_ > v.shape[0]-1)

    x_[out_of_roi]=0
    y_[out_of_roi]=0

    x0 = np.floor(x_).astype(int)
    y0 = np.floor(y_).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    dx1 = (x_ - x0).reshape(-1,1)
    dx0 = (x1 - x_).reshape(-1,1)
    dy1 = (y_ - y0).reshape(-1,1)
    dy0 = (y1 - y_).reshape(-1,1)

    q00 = v[y0, x0]
    q01 = v[y0, x1]
    q10 = v[y1, x0]
    q11 = v[y1, x1]

    vn =q00 * dy0 * dx0 + \
        q01 * dy0 * dx1 + \
        q10 * dy1 * dx0 + \
        q11 * dy1 * dx1
    vn[out_of_roi]=np.nan
    return vn
