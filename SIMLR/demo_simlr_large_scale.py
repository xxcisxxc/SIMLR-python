import time
import numpy as np
from sklearn.utils.extmath import randomized_svd as rsvd
from SIMLR_large_scale import SIMLR_large_scale
from src import Cal_NMI
from loadpkl import loadpkl
import matplotlib.pyplot as plt
import pdb 

def fast_pca(in_X, K):
    
    in_X = in_X - np.tile(np.mean(in_X,axis=0,keepdims=True),(in_X.shape[0],1))
    U, S, _ = rsvd(in_X, K);
    K = min(S.shape[0],K);
    X = np.dot(U[:,0:K],np.diag(np.sqrt(S)));
    X = X/np.tile(np.sqrt(np.sum(X**2,axis=1,keepdims=True)),(1,K));
    return X 

datasets = ['Zeisel']

for ds in datasets:
    data = loadpkl(ds)
    X = np.array(data['in_X']).astype('double')
    true_labs = np.array(data['true_labs']).astype('double')
    C = true_labs.max()

    in_X = np.log10(1+X)
    start = time.time()
    X = fast_pca(in_X,500)
    end = time.time()
    print("pca passing time : ",end-start)

    yKmeans, _, ydata = SIMLR_large_scale(X, int(C), k=30)
    print(Cal_NMI(true_labs, yKmeans.labels_))
    plt.scatter(ydata[:,0], ydata[:,1], c=true_labs.flatten(), alpha=0.5)
    plt.show()


