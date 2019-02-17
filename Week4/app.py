import glob
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC as SVC
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.model_selection import cross_val_score
from simplelbp import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier




#########---------------Q3---------------#########

digits = load_digits()

print(digits.data.shape)

print(digits.keys())

#This section is just for test, can be removed
#data = digits.data
#data1 = data[24].reshape((8, 8))

#target = digits.target

#image = digits.images


#plt.gray()
#plt.matshow(data1)
#plt.show()

plt.gray()
plt.matshow(digits.images[0]) 
plt.show()

print('The label of corresponding digit is: ' + str(digits.target[0]))

X = digits.data
y = digits.target


# train_test_split splits arrays or matrices into random train and test subsets.
# That means that everytime you run it without specifying random_state, you will get a different result and 
# this is expected behavior.
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

names = [
        "Nearest Neighbors",
        "LinearDiscriminant",
        "Linear SVM",
        "LogisticRegression"]


classifiers = [
    KNN(),
    LDA(),
    SVC(),
    logreg()
    ]



for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('The score of '+ name + ' classifier is '+ str(score))


#########---------------Q4---------------#########


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

# Test our loader

X, y = load_data("GTSRB_subset")
F = extract_lbp_features(X)
print("X shape: " + str(X.shape))
print("F shape: " + str(F.shape))

#X_train, X_test, y_train, y_test = train_test_split(F, y, test_size=0.02)

names = [
        "Nearest Neighbors",
        "LinearDiscriminant",
        "Linear SVM",
        "LogisticRegression"]


classifiers = [
    KNN(),
    LDA(),
    SVC(),
    logreg()
    ]



for name, clf in zip(names, classifiers):
   
    scores = cross_val_score(clf, F, y, cv=5)
    #scores = cross_val_score(clf, X_train, y_train, cv=5)
    
    
    print('The scores of '+ name + ' classifier using cross validation is'+ str(scores) + 
          ' and average is ' + str(scores.mean()))

#########---------------Q5---------------#########


X_train, X_test, y_train, y_test = train_test_split(F, y, test_size=.2)

names = ["Random Forest", "AdaBoost", "Extra Trees", "Gradient Boosting" ]

classifiers = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier()
        ]

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('The score of '+ name + ' classifier is '+ str(score))
   
 
