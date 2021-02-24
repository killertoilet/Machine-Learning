# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:48:05 2021

@author: kesuiker
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


# Task 1
#choosing to go with the boston data set to predice house prices with a linear 
# regression model
#linear  regression is needed because the target is numerical and can take ony value 
# number range as opposed to logistic that is used as a classifier when the target is
# of a discrete number of potential values to guess.

# load boston data set and place into data frame

print('TASK 1')
boston = load_boston()
df = pd.DataFrame(boston.data, None, boston.feature_names)
df['target'] = boston.target

feature_names = boston.feature_names
print(feature_names)

# this array will hold the scores that we will average at the end
scores = []

# 10 splits, with a shuffle of the indicies 
kf = KFold(n_splits = 10, shuffle = True)

for train_index, test_index in kf.split(df):
    # create a train and test data frame with the inicies generated from KFold
    train_df, test_df = df.loc[train_index], df.loc[test_index]
    
    lr = LinearRegression()
    lr.fit(train_df[feature_names], train_df[['target']])
    scores.append(lr.score(test_df[feature_names], test_df[['target']]))
    print(lr.score(test_df[feature_names], test_df[['target']]))

print('average score: ' + str(np.mean(scores)))


print('-----------------------------------------------------------------')

# Task 2
print('TASK 2')
# first without shuffle
kf = KFold(n_splits = 5, shuffle = False)

print('KFold with shuffle = False')
for train_index, test_index in kf.split(df):
    # create a train and test data frame with the inicies generated from KFold
    train_df, test_df = df.loc[train_index], df.loc[test_index]
    
    lr = LinearRegression()
    lr.fit(train_df[feature_names], train_df[['target']])
    scores.append(lr.score(test_df[feature_names], test_df[['target']]))
    
    print('Coef: ' + str(lr.coef_))
    print('Intercept: ' +str(lr.intercept_))



# second with shuffle
print('-----------------------------------------------------------------')
print('KFold with shuffle = True')

kf = KFold(n_splits = 5, shuffle = True)
for train_index, test_index in kf.split(df):
    # create a train and test data frame with the inicies generated from KFold
    train_df, test_df = df.loc[train_index], df.loc[test_index]
    
    lr = LinearRegression()
    lr.fit(train_df[feature_names], train_df[['target']])
    scores.append(lr.score(test_df[feature_names], test_df[['target']]))
    
    print('Coef: ' + str(lr.coef_))
    print('Intercept: ' +str(lr.intercept_))