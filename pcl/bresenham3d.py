import numpy as np


def bresenham3d(q1, q2):
    q1 = q1.copy()
    pt_list = [q1.copy()]
    d = np.abs(q2-q1)
    if (q2[0] > q1[0]):
        xs = 1
    else:
        xs = -1
    if (q2[1] > q1[1]):
        ys = 1
    else:
        ys = -1
    if (q2[2] > q1[2]):
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis" 
    if (d[0] >= d[1] and d[0] >= d[2]):
        p1 = 2 * d[1] - d[0]
        p2 = 2 * d[2] - d[0]
        while (q1[0] != q2[0]):
            q1[0] += xs
            if (p1 >= 0):
                q1[1] += ys
                p1 -= 2 * d[0]
            if (p2 >= 0):
                q1[2] += zs
                p2 -= 2 * d[0]
            p1 += 2 * d[1]
            p2 += 2 * d[2]
            pt_list.append(q1.copy())

            # Driving axis is Y-axis" 
    elif (d[1] >= d[0] and d[1] >= d[2]):
        p1 = 2 * d[0] - d[1]
        p2 = 2 * d[2] - d[1]
        while (q1[1] != q2[1]):
            q1[1] += ys
            if (p1 >= 0):
                q1[0] += xs
                p1 -= 2 * d[1]
            if (p2 >= 0):
                q1[2] += zs
                p2 -= 2 * d[1]
            p1 += 2 * d[0]
            p2 += 2 * d[2]
            pt_list.append(q1.copy())

            # Driving axis is Z-axis" 
    else:
        p1 = 2 * d[1] - d[2]
        p2 = 2 * d[0] - d[2]
        while (q1[2] != q2[2]):
            q1[2] += zs
            if (p1 >= 0):
                q1[1] += ys
                p1 -= 2 * d[2]
            if (p2 >= 0):
                q1[0] += xs
                p2 -= 2 * d[2]
            p1 += 2 * d[1]
            p2 += 2 * d[0]
            pt_list.append(q1.copy())
    return pt_list 
