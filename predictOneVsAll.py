# -*- coding: utf-8 -*-
import numpy as np
from sigmoid import sigmoid

def predictOneVsAll(Theta, X):
    m = X.shape[0]
    X = np.hstack((X, np.ones((m, 1))))
    z = np.matmul(Theta, X.T)
    g_of_z = sigmoid(z)
    maxInd = g_of_z.argmax(axis = 0)
    return maxInd