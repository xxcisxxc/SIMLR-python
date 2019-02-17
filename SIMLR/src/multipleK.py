import numpy as np
from scipy.stats import norm
from .dist2 import dist2

eps = np.finfo(np.double).eps

def multipleK(x):
    if type(x) != np.ndarray:
        raise TypeError("Please input 'numpy.ndarray' type variable into function multipleK")

    N = x.shape[0]
    Kernels = []
    sigma = np.arange(2, 1-0.25, -0.25)
    Diff = dist2(x)
    T = np.sort(Diff)
    INDEX = np.argsort(Diff)
    m, n = Diff.shape
    allk = np.arange(10, 30+2, 2)
    for allk_l in allk:
        if allk_l < N-1:
            TT = np.mean(T[:,2-1:allk_l+1], axis=1, keepdims=True) + eps
            Sig = (np.tile(TT, n) + np.tile(TT.T, (n, 1))) / 2
            Sig = Sig * (Sig>eps) + eps
            for sigma_j in sigma:
                W = norm.pdf(Diff, 0, sigma_j*Sig)
                Kernels.append((W+W.T)/2)
                
    Kernels = np.array(Kernels)
   
    D_Kernels = []
    for K in Kernels:
        k = 1 / np.sqrt(np.diag(K)+1)
        G = K
        G_diag = np.diag(G).reshape(-1,1)
        D_temp = (np.tile(G_diag, len(G)) + np.tile(G_diag.T, (len(G), 1)) - 2*G) / 2
        D_temp = D_temp - np.diag(np.diag(D_temp))
        D_Kernels.append(D_temp)

    D_Kernels = np.array(D_Kernels)
    return D_Kernels
