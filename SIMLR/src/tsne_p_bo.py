import numpy as np
tiny = np.finfo(np.double).tiny

def tsne_p_bo(P, no_dims=np.array([2])):
    if no_dims.size > 1:
        initial_solution = True
        ydata = no_dims
        no_dims = len(no_dims)
    elif no_dims == []:
        no_dims = 2
    else:
        initial_solution = False
        no_dims = no_dims[0]

    n = P.shape[0]
    momentum = 0.08
    final_momentum = 0.1
    mom_switch_iter = 250
    stop_lying_iter = 100
    max_iter = 1000
    epsilon = 500
    min_gain = 0.01

    P -= np.diag(np.diag(P))
    P = 0.5 * (P + P.T)
    P = P / np.sum(P)
    P[P<tiny] = tiny
    const = np.sum(P * np.log(P))
    if not initial_solution:
        P = P * 4
        ydata = 0.0001 * np.random.standard_normal((n, no_dims))
    y_incs = np.zeros(ydata.shape)
    gains = np.ones(ydata.shape)
    for iter_ in range(max_iter):
        sum_ydata = np.sum(ydata**2, axis=1, keepdims=True)
        num = 1 / (1+sum_ydata+sum_ydata.T-2*np.dot(ydata, ydata.T))
        num -= np.diag(np.diag(num))
        Q = num / np.sum(num)
        Q[Q<tiny] = tiny
        L = (P - Q) * num
        y_grads = 4 * np.dot(np.diag(np.sum(L, axis=0))-L, ydata)
        gains = (gains+0.2)*(np.sign(y_grads)!=np.sign(y_incs)) + (gains*0.8)*(np.sign(y_grads)==np.sign(y_incs))
        gains[gains<min_gain] = min_gain
        y_incs = momentum * y_incs - epsilon * (gains*y_grads)
        ydata = ydata + y_incs
        ydata = ydata - np.mean(ydata, axis=0, keepdims=True)
        ydata[ydata<-100] = -100
        ydata[ydata>100] = 100
        if iter_+1 == mom_switch_iter:
            momentum = final_momentum
        if iter_+1 == stop_lying_iter and not initial_solution:
            P = P / 4
        if not (iter_+1) % 10:
            cost = const - np.sum(P*np.log(Q))
            print('Iteration %s: error is %s' % (iter_+1, cost))

    return ydata
