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


################---------------Q4--------------################

def load_data(folder):
    """ 
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """
    
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
          
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]
            

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def extract_lbp_features(X, P = 8, R = 5):
    """
    Extract LBP features from all input samples.
    - R is radius parameter
    - P is the number of angles for LBP
    """
    
    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)

X, y = load_data("GTSRB_subset")
F = extract_lbp_features(X)
#F= Normalizer().fit(F)
F = scale(F)

X_train, X_test, y_train, y_test = train_test_split(F, y, test_size=0.2)

names = [
         "LogisticRegression",
         "SVC"]


classifiers = [
    logreg(),
    SVC()
        ]

C_range = 10.0 ** np.arange(-5, 0)

for name, clf in zip(names, classifiers):
    for C in C_range:
        for penalty in ["l1", "l2"]:
            clf.C = C
            clf.penalty = penalty
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print ('The score of '+ name + ' for C = %.2e and penalty = %s is %.3f' % (C, penalty, score))
            
            
