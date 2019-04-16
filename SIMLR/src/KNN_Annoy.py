from annoy import AnnoyIndex
import numpy as np
import sys
def KNN_Annoy(X, KK):
	NK = KK
	NN, NF = X.shape
	if KK > NF:
		raise ValueError("KK should be less than 2th-dim of X")
	
	t = AnnoyIndex(NF,metric='euclidean')
	for i, v in enumerate(X):
		t.add_item(i, v)

	t.build(100)
	ind = []
	val = []

	for i in range(NN):
		closest = t.get_nns_by_item(i, NK)
		ind.append(closest)
		val.append([t.get_distance(i, j) for j in closest])

	return np.array(ind), np.array(val)
