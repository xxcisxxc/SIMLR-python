import numpy as np

def umkl_bo(D, beta=None):
    if beta is None:
        beta = 1/len(D)

    tol = 1e-4
    u = 20
    logU = np.log(u)
    H, thisP = Hbeta(D, beta)
    betamin = -np.inf
    betamax = np.inf
    Hdiff = H - logU
    tries = 0
    while np.abs(Hdiff) > tol and tries < 30:
        if Hdiff > 0:
            betamin = beta
            if np.isinf(betamax):
                beta =  beta * 2
            else:
                beta = (beta + betamax) / 2
        else:
            betamax = beta
            if np.isinf(betamin):
                beta = beta / 2
            else:
                beta = (beta + betamin) / 2

        H, thisP = Hbeta(D, beta)
        Hdiff = H - logU
        tries = tries + 1

    return thisP

def Hbeta(D, beta):
    D = (D-np.min(D)) / (np.max(D)-np.min(D)+np.finfo(np.double).eps)
    P = np.exp(-D*beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta*np.sum(D*P)/sumP
    P = P / sumP
    return H, P
