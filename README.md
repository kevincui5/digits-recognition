
The problem come from Andrew Ng's machine learning course projects from Coursera, and 
I'd like to implement them in python instead of matlab/octave

We like to implement logistic regression and neural network to recognize 
hand-written digits and compare the result at the end.
The dataset is in the file ex3data1.mat.  It is a subset of MNIST hand-written 
digits, containing 5000 training sets. Each digit image is a 20x20 gray scale 
image but in ex3data1.mat image is converted to float64, each row represent a image,
and there are 400 columns, each as a pixel, a feature.
To try, just execute ex3.py
For the first part of the exercise, I implemented the regularized logistic regression
just like the logistic regression exercise, except it is multi-class.  Again, I 
didn't use any library or framework, just train the model 10 times, each time on
different digits, and pick the largest probability from the 10 results for each
training set.

For the second part, we are to implement a simple nural network to recognize the digits.
We don't need to implement the back-propagation, saving for another exercise.  
We are given the pre-trained parameters and they are saved in ex3weights.mat.  
I implemented the forward-propagation in the predict.py. so no optimization objection
algorithm library needed for this exercise.
run ex3_nn.py to try

DO NOT USE THIS SOURCE CODE FOR THE EXERCISES/PROJECTS IN COURSERA MACHINE
 LEARNING COURSE.