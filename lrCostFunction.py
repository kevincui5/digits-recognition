# -*- coding: utf-8 -*-
import numpy as np
from sigmoid import sigmoid

def lrCostFunction1(theta, X, y, l):
    m = X.shape[0]
    z = X.dot(theta.T)
    g_of_z = sigmoid(z)[:, None]
    reg = 0.5 * l / m * (np.sum(theta * theta) - theta[0] * theta[0])
    first = -y * np.log(g_of_z)
    second = (1 - y) * np.log(1 - g_of_z)
    J = np.sum(first - second) / m + reg
    return J

def gradient(theta, X, y, l):
    m = X.shape[0]
    z = X.dot(theta.T)
    g_of_z = sigmoid(z)[:, None]
    first = np.sum(X * (g_of_z - y), axis=0)
    second = theta.T * l
    grad = (first + second) / m 
    grad[0] = grad[0] - l / m * theta[0]

# from other's blog
def gradient2(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y
    
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    
    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    
    return np.array(grad).ravel()

# matrix implementation, not recommanded
def lrCostFunction2(theta, X, y, l):
    m = X.shape[0]
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    z = X * theta.T
    g_of_z =sigmoid(z)
    first = np.multiply(-y, np.log(g_of_z))
    second = np.multiply((1 - y), np.log(1 - g_of_z))
    reg = 0.5*l/m*(np.sum(np.multiply(theta,theta))-theta.item(0)*theta.item(0))
    J = np.sum(first - second) / m + reg
    
    first = np.sum(np.multiply((g_of_z - y), X), axis=0) / m
    second = l / m * theta.T
    grad = first.T + second
    grad[0, 0] = grad.item(0) - l / m * theta.item(0)
    return J, grad