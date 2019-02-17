import numpy as np
from scipy.sparse import dia_matrix

def NE_dn(w, type):
    w = w * len(w)
    D = np.sum(np.abs(w), axis=1) + np.finfo(np.double).eps

    if type == 'ave':
        D = 1 / D
        D = dia_matrix(np.diag(D))
        wn = D.dot(w)
    elif type == 'gph':
        D = 1 / np.sqrt(D)
        D = dia_matrix(np.diag(D))
        wn = D.dot((D.T.dot(w.T)).T)
    else:
        raise ValueError("Please input right strs")

    return wn
