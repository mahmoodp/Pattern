'''
import numpy as np
import matplotlib.pyplot as plt
import csv


data = np.loadtxt('locationData.csv', dtype='float', delimiter=" ", usecols=range(0, 3))
print(data.shape)

plt.plot(data[:, 0], data[:, 1], 'ro')
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')

plt.plot(data[:, 0], data[:, 1], data[:, 2], 'gray')
plt.show()


print("----------------------------------------> part 2")


with open('locationData.csv', 'r') as data_csv:
    reader = csv.reader(data_csv, delimiter=" ")

    for line in reader:
        values = (line[0], line[1], line[2])
        print(values)
        #print(values.shape)
'''


import numpy as np

import matplotlib.pyplot as plt
#
from mpl_toolkits.mplot3d import Axes3D 
from scipy.io import loadmat
####------------------------------------ 1 ------------------------

print("-----------------------------> part 1")

with open("locationData.csv", "r") as fp:
    data = np.loadtxt(fp)

    print ("Result size is %s" % (str(data.shape)))

####------------------------------------ 2 ------------------------


print("-----------------------------> part 2")

print("Plot the figures")

fig1 = plt.subplot(2, 2, 1)
fig1.plot(data[:, 0], data[:, 1], 'ro')

fig2 = plt.subplot(2, 2, 2, projection = "3d")
fig2.plot(data[:, 0], data[:, 1], data[:, 2], 'gray')


####------------------------------------ 3 ------------------------
 
print("-----------------------------> part 3")

X=[]

with open("locationData.csv","r") as fp:
    for line in fp:
        values = line.split(" ")
        values = [float(v) for v in values]
        X.append(values)

X= np.array(X)

print("Result size is %s" % (str(X.shape)))

print ("Result of numpy.all test: %s" % (str(np.all(data==X))))

print ("Result of numpy.any test: %s" % (str(np.any(data!=X))))

####------------------------------------ 4 ------------------------
 
print("-----------------------------> part 4")

mat = loadmat("twoClassData.mat")

print(mat.keys())

X= mat["X"]

y = mat["y"].ravel()

fig3 = plt.subplot(2, 2, 3)

fig3.plot(X[y == 0, 0], X[y == 0, 1], 'ro')

fig3.plot(X[y == 1, 0], X[y == 1, 1], 'bo')


####------------------------------------ 5 ------------------------
 
print("-----------------------------> part 5")

x = np.load("x.npy")

y = np.load("y.npy")

A = np.vstack([x, np.ones(len(x))]).T

m, c = np.linalg.lstsq(A, y)[0]

print(m, c)

fig4 = plt.subplot(2, 2, 4)

fig4.plot(x,y,'bo')

fig4.plot(x, m*x + c, 'r')


plt.show()
