import numpy as np
import matplotlib.cm


def apply_colormap(cmap_name, data):
    cmap = matplotlib.cm.get_cmap(cmap_name)
    cmap = np.asanyarray([cmap(i - 1) for i in range(cmap.N)])

    data_n = data - np.nanmin(data)
    mx = np.nanmax(data_n)
    data_n /= mx if ~np.isnan(mx) else 1
    data_n *= cmap.N
    data_n = data_n.astype(int)
    data_out = cmap[data_n]
    return data_out
