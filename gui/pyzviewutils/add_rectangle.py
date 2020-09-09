from .add_color import add_color
from ...utils.apply_transformation import apply_transformation
from ...utils.create_transformation import create_transformation
import numpy as np


def add_rectangle(t=np.zeros(3), r=np.zeros(3), s=np.zeros(3),c='r'):
    tform = create_transformation(r, t, s)
    v = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
    f = np.array(
        [[3, 1, 0], [3, 1, 2], [3, 6, 2], [3, 7, 6], [0, 1, 5], [0, 5, 4], [0, 7, 4], [0, 3, 7], [1, 2, 6], [1, 6, 5],
         [5, 6, 7], [4, 5, 7]])
    v = apply_transformation(tform, v)
    v = add_color(v,c)
    return v, f
