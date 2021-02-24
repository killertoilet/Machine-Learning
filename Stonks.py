# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:33:54 2021

@author: kesuiker
"""

# do some cool shit

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

file = 'C:\\DEV\\Skewl\\DataMining\\Data\\ETFs\\voo.us.txt'

df = pd.read_csv(file)

feature_names = ['Open', 'High', 'Low', 'Volume', 'Open2', 'Open3']
poly_names = ['Open2', 'High2', 'Low2', 'Volume2']

poly_features = PolynomialFeatures(degree=3, include_bias=False)

open2 = poly_features.fit_transform(df[['Open']])

df['Open2'] = open2[:,1]
df['Open3'] = open2[:,2]
target = ['Close']

lr = LinearRegression()

lr.fit(df[feature_names], df[target])

print(lr.predict([[359.71, 360.1, 356.86, 2071520, 129391.2841, 46543338.8]]))