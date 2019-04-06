# One-vs-all

import scipy.io
import numpy as np
from lrCostFunction import lrCostFunction1, lrCostFunction2, gradient2, gradient
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

data = scipy.io.loadmat('ex3data1.mat');
X = data.get('X')
y = data.get('y')

#displayData not implemented

# =============================================================================
#  ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
# =============================================================================

# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization');

theta_t = np.array([-2, -1, 1, 2])
intercept = np.ones((5,1))
X_t = np.arange(1,16).reshape(3,5).transpose()/10
X_t = np.hstack((intercept, X_t))
y_t = np.array([[1],[0],[1],[0],[1]])
l_t = 3

J = lrCostFunction1(theta_t, X_t, y_t, l_t)
grad = gradient(theta_t, X_t, y_t, l_t)

print('\nCost: %f\n', J);
print('Expected cost: 2.534819\n');
print('Gradients:\n');
print(' %f \n', grad);
print('Expected gradients:\n');
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

#============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

lmbda = 0.1
num_labels = 10 #the number of digits in the image
all_theta = oneVsAll(X, y, num_labels, lmbda)

# ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X);
correct = np.array([1 if (a + 1 == b) else 0 for (a, b) in zip(pred, y)])
#prediction method followed andrew's ML course, picking the highest (max)
# among the 10 results, instead of picking the one with > 50%, which is the method
# many online blog used.  andrew's method assumes the numeric in the example must belong to
# one of the ten digits
print('\nTraining Set Accuracy: %f\n', correct.mean(axis=0) * 100)
