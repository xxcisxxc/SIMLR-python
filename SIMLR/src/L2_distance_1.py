import numpy as np

def L2_distance_1(a, b):
    if a.shape != b.shape:
        raise ValueError("The dimensions of a and b don't agree")
    
    if a.shape[0] == 1:
        a = np.concatenate((a, np.zeros(a.shape)), axis=0)
        b = np.concatenate((b, np.zeros(b.shape)), axis=0)
    elif len(a.shape) == 1:
        a = a.reshape(1,-1)
        b = b.reshape(1,-1)
        a = np.concatenate((a, np.zeros(a.shape)), axis=0)
        b = np.concatenate((b, np.zeros(b.shape)), axis=0)

    aa = np.sum(a*a, axis=0)
    bb = np.sum(b*b, axis=0)
    ab = np.dot(a.T, b)
    d = np.tile(aa.reshape(-1,1), len(bb)) + np.tile(bb, (len(aa), 1)) - 2 * ab
    d = np.real(d)
    d[d<0] = 0

    d = d * (1-np.eye(d.shape[0]))
    return d
