# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:28:48 2021

@author: kesuiker
"""

from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error




mnist = fetch_openml('mnist_784', version=1)
iris = load_iris()

#pull data and target from mnist
x, y = mnist["data"], mnist["target"]

#convert y to ints
y = y.astype(np.uint8)

#seperate training data and test data
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]


#iterate through each number and everage the images
for i in range(10):
    avg_img = np.average(x_train[y_train==i],0)
    plt.subplot(2, 5, i+1)
    plt.imshow(avg_img.reshape((28,28))) 
    plt.axis('off')
    
    
#-------------------------------------------------------------------------------------------------

 
    df = pd.DataFrame(iris.data, None, iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = df['target'].map({i:name for i,name in enumerate(iris.target_names)})
    
    
    lr = LogisticRegression()
    
    lr.fit(df[iris.feature_names], df['target'])
    df.loc[:,'prediction'] = lr.predict(df[iris.feature_names])
    
    
    print(confusion_matrix(df['target'], df['prediction']))
    print(classification_report(df['target'], df['prediction']))
    
    
    lr = LinearRegression()
    lr.fit(df[iris.feature_names], df['target'])
    df.loc[:,'prediction'] = lr.predict(df[iris.feature_names]).round()
    
    #
    print(confusion_matrix(df['target'], df['prediction']))
    print(classification_report(df['target'], df['prediction']))

#------------------------------------------------------------------------------------------
#