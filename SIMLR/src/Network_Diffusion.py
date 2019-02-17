import numpy as np
from .dominateset import dominateset
from .TransitionFields import TransitionFields
from scipy.sparse import dia_matrix

def Network_Diffusion(A, K):
    A = A - np.diag(np.diag(A))
    P = dominateset(np.abs(A), min(K, len(A)-1)) * np.sign(A)
    DD = np.sum(np.abs(P.T), axis=0)
    P = P + np.eye(len(P)) + np.diag(DD)
    P = TransitionFields(P)
    D, U = np.linalg.eig(P)
    d = np.real(D + np.finfo(np.double).eps)
    alpha = 0.8
    beta = 2
    d = (1-alpha)*d / (1-alpha*(d**beta))

    D = np.diag(np.real(d))
    #U = np.real(U)
    W = np.dot(U, np.dot(D, U.T))
    W = (W*(1-np.eye(len(W)))) / np.tile(1-np.diag(W).reshape(-1,1), len(W))
    D = dia_matrix(np.diag(DD))
    W = D.dot(W)
    W = (W + W.T) / 2
    return W
