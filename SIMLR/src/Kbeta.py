import numpy as np

def Kbeta(D_K, alpha):
	if len(D_K.shape) == 2 or len(D_K.shape) == 1:
		return alpha * D_K
	K = np.zeros((D_K.shape[1], D_K.shape[2]))
	for i, a in enumerate(alpha):
		K += D_K[i] * a
	return K