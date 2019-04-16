import numpy as np
from scipy.stats import norm
from .dist2 import dist2
import pdb 
eps = np.finfo(np.double).eps

def large_multipleK(val,ind,KK):
    # if type(x) != np.ndarray:
    #     raise TypeError("Please input 'numpy.ndarray' type variable into function multipleK")

    val = val*val
    sigma = np.arange(2, 1-0.25, -0.25)
    allk = np.arange(np.ceil(KK/2), np.ceil(KK*1.5)+np.ceil(KK/10), np.ceil(KK/10))
    D_Kernels = []
    for allk_l in allk:
        if allk_l < val.shape[1]:
            temp = np.mean(val[:,0:int(allk_l)], axis=1, keepdims=True)
            temp0 = 0.5*(np.tile(temp,(1,val.shape[1])) + temp[ind].squeeze())+ eps

            for sigma_j in sigma:
                temp = norm.pdf(val,0,sigma_j*temp0)
                temptemp = temp[:,0]
                temp = 0.5*(np.tile(temptemp[:,np.newaxis],(1,val.shape[1])) + temptemp[ind]) - temp;
                D_Kernels.append(temp+eps)

    D_Kernels = np.array(D_Kernels)

    return D_Kernels
