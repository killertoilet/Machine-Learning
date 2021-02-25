# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:55:07 2021

@author: kesuiker
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.datasets import load_iris
import pandas as pd



# Task 1 ---------------------------------------------------------------------


def error_func(x, y):
    return np.sin(x) + np.cos(y)

def x_gradient(x, y):

    return np.cos(x)

def y_gradient(x, y):

    return - np.sin(y)

def gradients(x,y):
    return x_gradient(x, y), y_gradient(x, y)

val = [-4, -2, 0, 2, 4]
x1 = []
y1 = []

for i in range(len(val)):
    for k in range(len(val)):
        x1.append(val[i])
        y1.append(val[k])
x1 = 0
y1 = 0
print(x1)
sinx = np.sin(x1)
cosy = np.cos(y1)

learning_rate = 0.1
grads = []
test = []
test2 = []

x1 = [0]
y1 = [0]

# arbitrary starting point
error_x = [0.1]
error_y = [0.1]

for i in range(50):
    
    grads = gradients(error_x[i], error_y[i])
    increment = -np.array(grads) * learning_rate
    error_x.append(error_x[i] + increment[0])
    error_y.append(error_y[i] + increment[1])
    
    
    
xax = np.linspace(-5, 5, 500)
yax = np.linspace(-5, 5, 500)

x,y = np.meshgrid(xax, yax)
z = error_func(x, y)


plt.contourf(x,y,z)
plt.plot(error_x, error_y, '-r')


# Task 2 ----------------------------------------------------------------------

def cost(x, y):
    return (x - y)**2
    
def gradient_x(x):
    return 2 * x

def gradient_y(y):
    return -2 * y

def gradients(x, y):
    return gradient_x(x), gradient_y(y)

iris = load_iris()

df = pd.DataFrame(iris.data, None, iris.feature_names)
df['target'] = iris.target
df['target_names'] = df['target'].map({i:name for i,name in enumerate(iris.target_names)})

x = np.array([0,0,0,0], dtype=float)

learning_rate = 0.5


for i in range(50):
    y = np.array(df[iris.feature_names].iloc[i])

    grads = gradients(x, y)
    increment = -np.array(grads) * learning_rate
    x =+ increment
    
print("Mean of setosa class is: " + str(iris.data[:50].mean(axis=0)) + "\n")
print("Gradiet Descent Output Results: " + str(x[len(x)-1]))





























