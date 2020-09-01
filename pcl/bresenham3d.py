import numpy as np

def floor_q(v):
    fv =np.floor(v)
    is_round = v==fv
    v = v.copy()
    v[~is_round] =fv[~is_round]
    v[is_round]-=1
    return v
def ceil_q(v):
    fv =np.ceil(v)
    is_round = v==fv
    v = v.copy()
    v[~is_round] =fv[~is_round]
    v[is_round]+=1
    return v

def bresenham3d(q1, q2):
    n = q2 - q1
    n = n / np.linalg.norm(n)
    pt_list = []
    q_next = q1.copy()
    max_t = np.mean((q2-q1)/n)
    acc_t = 0
    while True:

        # p_a = floor_q(q_next)
        # p_b = ceil_q(q_next)
        p_a = np.floor(q_next-np.finfo(float).eps*1e3)
        p_b = np.ceil(q_next+np.finfo(float).eps*1e3)
        posible_t = np.r_[(p_a - q_next) / n, (p_b - q_next) / n]
        posible_t = [x for x in posible_t if x > 0]
        if len(posible_t)==0:
            break
        posible_t = np.min(posible_t)
        acc_t +=posible_t
        q_next = q_next + posible_t * n
        if acc_t>max_t:
            break
        pt_list.append(q_next)

    return pt_list
