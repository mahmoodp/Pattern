import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression as logreg
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import scale
from sklearn.svm import SVC
import glob
import os

################---------------Q3--------------################


with open('X.csv') as X_file:
    X = np.loadtxt(X_file, delimiter=',')
        
with open('y.csv') as y_file:
    y = np.loadtxt(y_file)
    

print(X.shape, y.shape)  
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# log_loss calculator function
def log_loss(w, x, y):
    # loss value
    L = 0

    for n in range(x.shape[0]):
        L += np.log(1+np.exp(-y[n]*np.dot(w, x[n])))
    return L


# gradient calucator function
def gradient_log_loss(w, x, y):
    # gradient
    gradient = 0

    for n in range(x.shape[0]):
        num = -np.exp(-y[n] * np.dot(w, x[n])) * y[n] * x[n]
        den = 1 + np.exp(-y[n] * np.dot(w, x[n]))
        gradient += num / den
    return gradient


w = np.random.rand(2)

# set step size to a small positive value
step = .0001

clf = logreg()


weight_history = []
acc_history = []


for _ in range(100):
    # Apply the gradient descent rule.
    w = w - step * gradient_log_loss(w, X, y)

    # Print the current state.
    #print("Iteration %d: w = %s (log-loss = %.2f)" %
    #      (iteration, str(w), log_loss(w, x, y)))

    # Compute the accuracy:
    y_prob = 1 / (1 + np.exp(-np.dot(X, w)))
            # Threshold at 0.5 (results are 0 and 1)
    y_pred = (y_prob > 0.5).astype(int)
            # Transform [0,1] coding to [-1,1] coding
    y_pred = 2*y_pred - 1

    accuracy = np.mean(y_pred == y)
    acc_history.append(accuracy)
    
    weight_history.append(w)

weight_history = np.array(weight_history)


fig, ax = plt.subplots(1, 2)
ax[0].plot(weight_history[:, 0], weight_history[:, 1], 'ro-')
ax[0].set_title('Weights')
ax[0].set_xlabel('w[0]')
ax[0].set_ylabel('w[1]')

ax[1].plot(acc_history)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')

plt.show


            
            
