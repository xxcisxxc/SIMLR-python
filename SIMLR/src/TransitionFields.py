import numpy as np
from .NE_dn import NE_dn

def TransitionFields(W):
    zeroindex = np.nonzero(np.sum(W, axis=1) == 0)[0]
    W = W * len(W)
    W = NE_dn(W, 'ave')
    w = np.sqrt(np.sum(np.abs(W), axis=0)+np.finfo(np.double).eps)
    W = W / np.tile(w, (len(W),1))
    W = np.dot(W, W.T)
    Wnew = W
    Wnew[zeroindex, :] = 0
    Wnew[:, zeroindex] = 0
    return Wnew
    