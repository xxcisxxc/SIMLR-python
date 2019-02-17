import numpy as np
from sklearn.cluster import KMeans
import time
import warnings
from src import *

eps = np.finfo(np.double).eps

def SIMLR(X, c, k=10, ifimpute=False, normalize=False):
    # the type of X must be np.ndarray or list(double)
    X = np.double(np.array(np.mat(X)))

    if ifimpute:
        Xmean = np.mean(X, axis=1)
        sub2ind = (X == 0)
        J = np.nonzero(sub2ind)[0]
        X[sub2ind] = Xmean[J]

    if normalize:
        X = X - np.min(X)
        X = X / np.max(X)
        X = X - np.mean(X, axis=0, keepdims=True)

    t0 = time.time()
    order = 2
    no_dim = np.array([c])

    NITER = 30
    num = X.shape[0]
    r = -1
    beta = 0.8
    D_Kernels = multipleK(X)
    del X
    alphaK = 1/D_Kernels.shape[0] * np.ones(D_Kernels.shape[0])
    distX = np.mean(D_Kernels, axis=0)
    distX1 = np.sort(distX)
    idx = np.argsort(distX)
    A = np.zeros((num, num))
    di = distX1[:, 2-1:k+2]
    rr = 0.5 * (k*di[:, k+1-1] - np.sum(di[:, 1-1:k], axis=1));
    id_ = idx[:,2-1:k+2]
    temp = (np.tile(di[:,k+1-1].reshape(-1,1), di.shape[1])-di) / np.tile((k*di[:,k+1-1]-np.sum(di[:,1-1:k], axis=1)).reshape(-1,1)+eps, di.shape[1])
    a = np.tile(np.arange(1, num+1).reshape(-1,1), id_.shape[1])
    A[a.flatten('F')-1, id_.flatten('F')] = temp.flatten('F')
    if r <= 0:
        r = np.mean(rr)
    lambda_ = max(np.mean(rr), 0)
    A[np.isnan(A)] = 0.
    S0 = np.max(distX) - distX
    S0 = Network_Diffusion(S0, k)
    S0 = NE_dn(S0, 'ave')
    S = (S0 + S0.T) / 2
    D0 = np.diag(np.sum(S, axis=order-1))
    L0 = D0 - S
    F, temp, evs = eig1(L0, c, False)
    F = NE_dn(F, 'ave')
    converge = []
    #NITER=1
    for iter_ in range(NITER):
        distf = L2_distance_1(F.T, F.T)
        A = np.zeros((num, num))
        b = idx[:, 1:]
        a = np.tile(np.arange(1, num+1).reshape(-1,1), b.shape[1])
        inda = (a.flatten('F')-1, b.flatten('F'))
        ad = ((distX[inda]+lambda_*distf[inda])/2/r).flatten('F').reshape(b.shape[1], num)
        ad = projsplx_c(-ad).T
        A[inda] = ad.flatten('F')
        A[np.isnan(A)] = 0.
        S = (1-beta) * A + beta * S
        S = Network_Diffusion(S, k)
        S = (S + S.T) / 2
        D = np.diag(np.sum(S, axis=order-1))
        L = D - S
        F_old = F
        F, temp, ev = eig1(L, c, 0)
        F = NE_dn(F, 'ave')
        F_new = F
        F = (1-beta) * F_old + beta * F
        evs = np.append(evs, ev, axis=1)
        DD = np.zeros(D_Kernels.shape[0])
        for i in range(D_Kernels.shape[0]):
            temp = (eps+D_Kernels[i, :, :]) * (eps+S)
            DD[i] = np.mean(np.sum(temp, axis=0))
        alphaK0 = umkl_bo(DD)
        alphaK0 = alphaK0 / np.sum(alphaK0)
        alphaK = (1-beta)*alphaK + beta*alphaK0
        alphaK = alphaK / np.sum(alphaK)
        fn1 = np.sum(ev[0:c])
        fn2 = np.sum(ev[0:c+1])
        converge.append(fn2 - fn1)
        if iter_ < 9 and ev[-1] > 0.000001:
            lambda_ = 1.5 * lambda_
            r = r / 1.01
        elif converge[iter_] > 1.01*converge[iter_-1]:
            S = S_old
            if converge[iter_-1] > 0.2:
                warnings.warn("Maybe you should set a larger value of c", RuntimeWarning)
            break
        S_old = S
        distX = Kbeta(D_Kernels, alphaK)
        distX1 = np.sort(distX)
        idx = np.argsort(distX)
    
    LF = F
    S = np.real(S)
    D = np.diag(np.sum(S, axis=order-1))
    L = D - S
    D, U = np.linalg.eig(L)
    if no_dim.size == 1:
        F = tsne_p_bo(S, U[:, 0:no_dim[0]])
    else:
        F = []
        for i in no_dim:
            F.append(tsne_p_bo(S, U[:, 0:i]))
    
    center = KMeans(n_clusters=c, n_init=20).fit(LF).cluster_centers_
    center = np.argmin(dist2(center,LF), axis=1)
    center = F[center, :]
    yKmeans = KMeans(n_clusters=c, init=center).fit(F)
    ydata = tsne_p_bo(S)
    t1 = time.time()
    timeOurs = t1 - t0
    return yKmeans, ydata, timeOurs
