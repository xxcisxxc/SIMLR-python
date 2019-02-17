import numpy as np

def dominateset(aff_matrix, NR_OF_KNN):
    A = -np.sort(-aff_matrix)
    B = np.argsort(-aff_matrix)
    res = A[:, 1-1:NR_OF_KNN]
    inds = np.tile(np.arange(1, len(aff_matrix)+1).reshape(-1,1), NR_OF_KNN)
    loc = B[:, 1-1:NR_OF_KNN]
    PNN_matrix1 = np.zeros(aff_matrix.shape)
    PNN_matrix1[inds.flatten('F')-1, loc.flatten('F')] = res.flatten('F')
    PNN_matrix = (PNN_matrix1 + PNN_matrix1.T) / 2

    return PNN_matrix
