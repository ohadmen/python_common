import numpy as np
def readobj(fn):
    v=np.zeros((0,3))
    f = np.zeros((0, 3))
    with open(fn, 'r') as fp:
        lines = fp.readlines()
    lines = [x.strip() for x in lines] #strip white
    lines = [x for x in lines if len(x)!=0 and x[0]!='#'] # remove comments
    v = np.stack([ np.fromstring(x[1:], dtype=np.float32, sep=' ') for x in lines if x[:2]=='v '])
    f = np.stack([np.fromstring(x[1:], dtype=np.float32, sep=' ') for x in lines if x[:2] == 'f '])
    f = f-1#one based
    f = f.astype(int)
    return v,f

