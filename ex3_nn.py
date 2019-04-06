# -*- coding: utf-8 -*-

# Part 2: Neural Networks

import scipy.io
import numpy as np
from lrCostFunction import lrCostFunction1, lrCostFunction2, gradient2, gradient
from oneVsAll import oneVsAll
from predict import predict

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)


# Load Training Data
print('Loading and Visualizing Data ...\n')

data = scipy.io.loadmat('ex3data1.mat');
X = data.get('X')
y = data.get('y')
m = X.shape[0]

# Randomly select 100 data points to display
#sel = randperm(size(X, 1))
#sel = sel(1:100)

#not implemented; displayData(X(sel, :))

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
data = scipy.io.loadmat('ex3weights.mat');
Theta1 = data.get('Theta1')
Theta2 = data.get('Theta2')


## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)
correct = np.array([1 if (a + 1 == b) else 0 for (a, b) in zip(pred, y)])
print('\nTraining Set Accuracy: %f\n', correct.mean(axis=0) * 100)
# =============================================================================
# print('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100)
# 
# #  To give you an idea of the network's output, you can also run
# #  through the examples one at the a time to see what it is predicting.
# 
# #  Randomly permute examples
# rp = randperm(m)
# 
# for i = 1:m
#     # Display 
#     print('\nDisplaying Example Image\n')
#     displayData(X(rp(i), :))
# 
#     pred = predict(Theta1, Theta2, X(rp(i),:))
#     print('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10))
#     
#     # Pause with quit option
#     s = input('Paused - press enter to continue, q to exit:','s')
#     if s == 'q'
#       break
#     end
# end
# =============================================================================

