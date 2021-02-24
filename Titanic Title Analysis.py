# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:57:47 2021

@author: kesuiker


# week 3 lab
# 1. apply simple train test splut on either the titanic and/or MNIST dataset when applying 
# logistic regression for different test sizes, 0.5, 0.75 and 0.9
# 2. Apply cross validation on the titanic dataset where you are trying to predict the non NaN
# values of the age column using the remaining numerical columns
"""


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import r2_score


#-------------------------------------------------------------------------------------------------
# Part 1

train_data = "titanic_train.csv"
df = pd.read_csv(train_data)

target = ['Survived']
feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare', 'female', 'male', 'S', 'C', 'Q']

# create new binary columns for male/ female
df_dummy = pd.get_dummies(df['Sex'])
df = pd.concat([df, df_dummy], axis=1)

# create new binary columns for embarked location
df_dummy = pd.get_dummies(df['Embarked'])
df = pd.concat([df, df_dummy], axis=1)

test_sizes = [0.5, 0.7, 0.9]

# 
#for i in range(len(test_sizes)):
#    print("Training with " + str(round((1 - test_sizes[i]) *100)) + "% data")
#    train_df, test_df = train_test_split(df, test_size = test_sizes[i])
#
#    lr = LogisticRegression()
#    lr.fit(train_df[feature_names], train_df[target])
#    
#    prediction = 'Prediction' + str(test_sizes[i])
#    df.loc[:, prediction] = lr.predict(df[feature_names])
#
#    print(confusion_matrix(df[target], df[prediction]))
#    print(classification_report(df[target], df[prediction]))


#-------------------------------------------------------------------------------------------------
# Part 2
# to guess the age of the missing data, will need to separte the data into a training set that
# contains poassangers with a valid age and those with NaN. Use cross validation to trian the model
# using an 80/20 split. with the last instance of the model create, try to predict the ages f the 
# reaimaing NaN aged passengers

def get_title(name):
    if "." in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return "no title"
    
def replace_title(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:
        return 'Mr'
    elif title in ['the Countess', 'Mme', 'Lady']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
    
    
new_indices = random.sample(list(df.index), len(df))
n_folds = 5

feature_names = ['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'female', 'male', 'S', 'C', 'Q', \
                  'Master', 'Miss', 'Mr', 'Mrs']
accuracies = {}

titles = sorted(set([x for x in df.Name.map(lambda x: get_title(x))]))

df['Title'] = df['Name'].map(lambda x: get_title(x))
df['Title'] = df.apply(replace_title, axis=1)

df_dummy = pd.get_dummies(df['Title'])
df = pd.concat([df, df_dummy], axis=1)

target = ["Age"]

idx = df.loc[pd.isna(df["Age"]), :].index

df_train_age = df.drop(idx)
df_test_age = df.loc[idx, :]

new_indices = random.sample(list(df_train_age.index), len(df_train_age))

batch_size = len(df_train_age)//n_folds

for i in range(n_folds):
    test_indices = new_indices[i*batch_size:(i+1)*batch_size]
    train_indices = set(new_indices).difference(test_indices)
    cross_train_df = df_train_age.loc[train_indices]
    cross_test_df = df_train_age.loc[test_indices]
    
    lr = LinearRegression()
    lr.fit(cross_train_df[feature_names], cross_train_df[target])
    
    print(lr.score(cross_test_df[feature_names], cross_test_df[target]))

    

df_test_age.loc[:,'Prediction'] = lr.predict(df_test_age[feature_names])


    
    
    
    
    
    
    