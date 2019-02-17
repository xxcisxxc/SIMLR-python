import numpy as np

def Cal_NMI(true_labels, cluster_labels):
    true_labels = np.double(true_labels)
    cluster_labels = np.double(cluster_labels)
    true_labels = true_labels - np.min(true_labels)
    cluster_labels = cluster_labels -np.min(cluster_labels)

    true_labels = true_labels.flatten()
    cluster_labels = cluster_labels.flatten()

    n = len(true_labels)
    cat = spconvert(np.array([np.arange(n), true_labels, np.ones(n)]))
    clss = spconvert(np.array([np.arange(n), cluster_labels, np.ones(n)]))
    cmat = np.dot(clss, cat.T)
    n_i = np.sum(cmat, axis=0, keepdims=True)
    n_j = np.sum(cmat, axis=1, keepdims=True)

    row, col = cmat.shape
    product = np.tile(n_i, (row, 1)) * np.tile(n_j, col)
    index = product > 0
    n = np.sum(cmat)
    product[index] = (n*cmat[index]) / product[index]
    index = product > 0
    product[index] = np.log(product[index])
    product = cmat * product
    score = np.sum(product)
    index = n_i > 0
    n_i[index] = n_i[index] * np.log(n_i[index]/n)
    index = n_j > 0
    n_j[index] = n_j[index] * np.log(n_j[index]/n)
    denominator = np.sqrt(np.sum(n_i) * np.sum(n_j))

    if denominator == 0:
        score = 0
    else:
        score = score / denominator

    return score

def spconvert(D):
    D = np.int64(D)
    S = np.zeros((np.max(D[1, :])+1, D.shape[1]))
    S[D[1,:], D[0,:]] = D[2,:]
    return S
