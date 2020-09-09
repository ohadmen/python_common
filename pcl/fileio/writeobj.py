import getpass
import numpy as np
import socket
from datetime import datetime


def writeobj(fn,v,f):
    with open(fn, 'w') as fp:
        fp.write("#Written by python write obj\n")
        fp.write("#User: {}\n".format( getpass.getuser()))
        fp.write("#Machine: {}\n".format(socket.gethostname()))
        fp.write("#Date: {}\n".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        fp.write("#\n")
        fp.write("# vertics: {}\n".format(v.shape[0]))
        fp.write("# faces: {}\n".format(f.shape[0]))

        np.apply_along_axis(lambda x: fp.write('v {} {} {}\n'.format(x[0], x[1], x[2])), axis=1, arr=v)
        np.apply_along_axis(lambda x: fp.write('f {:d} {:d} {:d}\n'.format(x[0], x[1], x[2])), axis=1, arr=f.astype(int)+1)
