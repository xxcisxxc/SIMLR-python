import numpy as np
from sklearn.cluster import KMeans
import time
import sys
import warnings
from scipy import sparse
from src import *
import pdb
eps = np.finfo(np.double).eps


def SIMLR_large_scale(X, c, k=10, ifimpute=False, normalize=False):
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

    # order = 2
    # no_dim = np.array([c])

    NITER = 5
    r = -1
    beta = 0.8

    start = time.time()
    ind, val = KNN_Annoy(X,2*k);
    val = np.double(np.abs(val));
    #construct multiple kernels
    D_Kernels = large_multipleK(val,ind,k)
    del val
    alphaK = 1/D_Kernels.shape[0] * np.ones(D_Kernels.shape[0])
    distX = np.mean(D_Kernels, axis=0)

    di = distX[:, 1:k+2]
    rr = 0.5*(k*di[:,k]-np.sum(di[:,0:k], axis=1))

    if r <= 0:
        r = np.mean(rr)

    lambda_ = max(np.mean(rr), 0)
    del rr 
    del di

    S0 = np.max(distX) - distX

    S0 = NE_dn(S0, 'ave')

    F,evalues = top_eig(S0, ind.copy(), c);

    d = evalues/np.max(evalues);
    d = (1-beta)*d / (1-beta*(d**2))
    F = F*(np.tile((d+eps).T,(len(F),1)))

    F = NE_dn(F, 'ave')
    F0 = F
    for iter_ in range(NITER):
        distf = large_L2_distance_1(F, ind);
        distf = (distX+lambda_*distf)/2/r;
        distf = projsplx_c(-distf.T).T;
        S0 = (1-beta)*S0+(beta)*distf;
        F,evalues = top_eig(S0, ind.copy(), c);
        d = evalues/np.max(evalues);
        d = (1-beta)*d / (1-beta*(d**2))
        F = F*(np.tile((d+eps).T,(len(F),1)))
        F = NE_dn(F, 'ave')
        F = (1-beta)*F0+(beta)*F
        F0 = F
        DD = np.zeros(D_Kernels.shape[0])
        for i in range(D_Kernels.shape[0]):
            temp = (eps+D_Kernels[i, :, :]) * (eps+S0)
            DD[i] = np.mean(np.sum(temp, axis=0))

        alphaK0 = umkl_bo(DD)
        alphaK0 = alphaK0 / np.sum(alphaK0)
        alphaK = (1-beta)*alphaK + beta*alphaK0
        alphaK = alphaK / np.sum(alphaK)
        lambda_ = 1.5 * lambda_
        r = r / 1.1

        distX = Kbeta(D_Kernels, alphaK)
    val = S0
    I = np.tile(np.arange(0, S0.shape[0]).reshape(-1,1),S0.shape[1])
    S0 = sparse.coo_matrix((S0.flatten()/2.0,(I.flatten(),ind.flatten())),shape=(S0.shape[0], S0.shape[0])) + sparse.coo_matrix((S0.flatten()/2.0,(ind.flatten(),I.flatten())),shape=(S0.shape[0],S0.shape[0]))
    end = time.time()
    print("simlr passing time : ",end-start)
    start = time.time()
    yKmeans = KMeans(n_clusters=c, n_init=50,random_state=0).fit(F)
    end = time.time()
    print("kmeans passing time : ",end-start)
    # pdb.set_trace()
    ydata = SIMLR_embedding_tsne(S0,1,2,F[:,0:2])
    # print(ydata.shape)
    
    return yKmeans, S0, ydata
