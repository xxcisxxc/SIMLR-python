import numpy as np
from scipy import linalg

def eig1(A, c=None, isMax=True, isSym=True):
    if c is None or c > A.shape[0]:
        c = A.shape[0]

    if isSym:
        A = np.maximum(A, A.T)

    d, v = np.linalg.eig(A)
    if isMax:
        d1 = -np.sort(-d)
        idx = np.argsort(-d)
    else:
        d1 = np.sort(d)
        idx = np.argsort(d)
    
    idx1 = idx[0:c]
    eigval = d[idx1]
    eigvec = np.real(v[:, idx1])
    eigval_full = d[idx].reshape(-1,1)

    return eigvec, eigval, eigval_full
