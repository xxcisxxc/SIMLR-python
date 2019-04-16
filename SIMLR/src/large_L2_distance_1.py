import numpy as np
import pdb
def large_L2_distance_1(F, ind):
    # if a.shape != b.shape:
    #     raise ValueError("The dimensions of a and b don't agree")
    m,n = ind.shape;
    I = np.tile(np.arange(0, m)[:,np.newaxis],(1,n))
    temp = np.sum((F[I,:]-F[ind,:])**2,axis=2)

    return temp
