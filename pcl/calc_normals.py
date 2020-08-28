import numpy as np


def norm(v):
    nn = np.linalg.norm(v, axis=2)
    vn = v / nn.reshape(list(nn.shape) + [1])
    return vn

def crossN(a,b):
    c = np.cross(a,b)
    return norm(c)

def calc_normals(xyz):
    """
    calc normals of input pcl
    :param xyz: NxMx3 point cloud, each coordinate in a different layer
    :return: MxNx3 point normals
    """
    assert(xyz.shape[2]==3)
    vu = np.roll(xyz,-1,axis=0)-xyz
    vr = np.roll(xyz, 1, axis=1)-xyz
    vd = np.roll(xyz, 1, axis=0)-xyz
    vl = np.roll(xyz, -1, axis=1)-xyz
    n = np.stack((crossN(vu,vl),crossN(vl,vd),crossN(vd,vr),crossN(vr,vu)))
    n = np.stack([np.nanmedian(n[:,:,:,i],axis=0) for i in range(3)],axis=2)
    n = norm(n)
    #reorient = np.sum(n * xyz, axis=2) > 0
    #n[reorient] *=-1
    return n