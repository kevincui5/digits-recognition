# -*- coding: utf-8 -*-
import numpy as np
from lrCostFunction import lrCostFunction1, lrCostFunction2, gradient2
from scipy.optimize import minimize

def oneVsAll(X, y, num_labels, lmbda):
    m = X.shape[0]
    n = X.shape[1]
    X = np.hstack((X, np.ones((m, 1))))
    all_theta = np.zeros((num_labels, n + 1))
    
    for i in range(1, num_labels + 1):
        theta = np.zeros(n + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (m, 1))
        
        # minimize the objective function
        fmin = minimize(fun=lrCostFunction1, x0=theta, args=(X, y_i, lmbda), method='TNC', jac=gradient2)
        all_theta[i-1,:] = fmin.x
    
    return all_theta
        