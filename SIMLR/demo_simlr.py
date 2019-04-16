from SIMLR import SIMLR
from src import Cal_NMI
from loadpkl import loadpkl
import numpy as np
import matplotlib.pyplot as plt

#datasets = ['mECS', 'Kolod', 'Pollen', 'Usoskin']
datasets = ['Pollen']

for ds in datasets:
	data = loadpkl(ds)
	X = data['in_X']
	true_labs = np.array(data['true_labs']).astype('double')
	C = true_labs.max()
	yKmeans, ydata, time = SIMLR(X, int(C))
	NMI_i = Cal_NMI(yKmeans.labels_, true_labs);
	print('The NMI value for dataset %s: %s' % (ds, NMI_i))

	scale = 20
	plt.scatter(ydata[:,0], ydata[:,1], c=true_labs.flatten(), alpha=0.5)
	plt.show()

