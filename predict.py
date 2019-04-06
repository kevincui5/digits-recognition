import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):

# Useful values
    m = X.shape[0]  #5000
    X = np.hstack((np.ones((m, 1)), X)) #5000x401

    A1 = X
    Z2 = np.matmul(Theta1, A1.T) #25x5000
    A2 = sigmoid(Z2) #25x5000
    A2 = np.vstack((np.ones((1, m)), A2)) #26x5000
    Z3 = np.matmul(Theta2, A2) #10x5000
    A3 = sigmoid(Z3)
    y_index = A3.argmax(axis = 0)
    return y_index



