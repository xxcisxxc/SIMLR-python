import numpy as np

def LaplacianScore(X, W):
	nSmp, nFea = X.shape
	a1, a2 = W.shape

	if a1 != a2 or a1 != nSmp:
		raise ValueError("The dimension of W is not accepted")

	D = np.sum(W, axis=1)
	L = W
	allone = np.ones((nSmp, 1))

	tmp1 = np.dot(D, X)

	DPrime = np.sum(np.dot(np.diag(D).T, X)*X, axis=0) - tmp1 * tmp1 / np.sum(D)
	LPrime = np.sum(np.dot(L.T, X)*X, axis=0) - tmp1 * tmp1 / np.sum(D)

	DPrime[DPrime<1e-12] = 10000

	Y = LPrime / DPrime
	Y = Y.T
	return Y