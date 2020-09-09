import numpy as np


def rgba2floatcol(rgba):
    if (rgba.shape[2] == 3):
        rgba = (np.concatenate((rgba, np.ones((list(rgba.shape[:2]) + [1]))), axis=2))

    rgba = np.clip(rgba,0,1)
    rgbafloat = (rgba * 255).astype(np.uint8)
    rgbafloat = rgbafloat.flatten()
    rgbafloat = rgbafloat.view(np.float32).reshape(rgba.shape[:2])
    return rgbafloat
