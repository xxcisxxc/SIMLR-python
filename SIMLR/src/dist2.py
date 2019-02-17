import numpy as np

def dist2(*args):
    if len(args) == 1:
        x = args[0]
        c = x
    elif len(args) == 2:
        x = args[0]
        c = args[1]
    else:
        raise ValueError("The number of input is uncorrect")

    if type(x) != np.ndarray or type(c) != np.ndarray:
        raise TypeError("Please input numpy.ndarray variable")

    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    if dimx != dimc:
        raise ValueError("Data dimension does not match dimension of centres")

    n2 = np.dot(np.ones((ncentres, 1)), np.sum(x**2, axis=1).reshape(1, -1)).T + \
        np.dot(np.ones((ndata, 1)), np.sum(c**2, axis=1).reshape(1, -1)) - \
        2*(np.dot(x, c.T))

    if True in (n2<0).flatten():
        n2[n2<0] = 0

    return n2
