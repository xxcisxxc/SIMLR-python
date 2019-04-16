import numpy as np
from scipy import sparse
from .compute_wtsne_obj_grad_repulsive_barneshut import compute_wtsne_obj_grad_repulsive_barneshut
import pdb
eps = np.finfo(np.double).eps

def tSNE_embed(P, Y0, attr, theta, max_iter, check_step, tol, max_time, verbose, recordYs):

    Y = Y0
    position = np.nonzero(P)
    Pnz = np.array(P[position[0],position[1]])
    weights = np.sum(P,axis=0)
    t = 1
    if recordYs:
        pass
    else:
        Ys = []
    n = P.shape[0]

    for iter in range(2,max_iter+1):
        Y_old = Y
        qnz = 1/(1+np.sum((Y[position[0],:]-Y[position[1],:])**2,1))
        Pq = attr * sparse.coo_matrix(((Pnz*qnz).flatten(),(position[0].flatten(),position[1].flatten())),shape=(n, n))
        _, repu = compute_wtsne_obj_grad_repulsive_barneshut(Y, weights, theta, 2)
        Y = (np.dot(np.array(Pq.todense()),Y)-repu/4)/(np.sum(np.array(Pq.todense()),axis=1,keepdims=True)+eps)

        if (recordYs) & (iter%check_step==0):
            t = t + 1
            Ys.append(Y)

    if recordYs:
         Ys = np.array(Ys) 

    return Y

def SIMLR_embedding_tsne(P, do_init,DD, Y0):
    P = 0.5*(P+P.T)
    P = P / np.sum(P)
    theta = 2
    check_step = 1
    tol = 1e-4
    max_time = np.inf
    verbose = False
    optimizer = 'fphssne'
    recordYs = False
    attr = 1
    max_iter = 300
    Y1 = tSNE_embed(P, Y0, attr, theta, max_iter, check_step, tol, max_time, verbose, recordYs)
    return tSNE_embed(P, Y1, attr, theta, max_iter, check_step, tol, max_time, verbose, recordYs)
