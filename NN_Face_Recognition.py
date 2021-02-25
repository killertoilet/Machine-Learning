# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:34:52 2021

@author: Kevin


Task 1:

Train a neural network on the Facial Keypoints dataset from last week's lab to predict the 15 keypoints of the face.

You are welcome to use 2,3 or 4 hidden layers of varying sizes (it is standard practice to set each hidden layer as an increasing powers of 2, e.g. for 3 layers the number of hidden nodes per layer could be 64, 128, 256)

Task 2:

Load your own face and run it through the trained network
"""

from tensorflow import keras
import pandas as pd
import numpy as np


file = 'FacialKeypoints.csv'



df = pd.read_csv(file)

col_names = []

for col in df.columns:
    col_names.append(col)

feature_names = col_names[:(len(col_names)-1)]
target = col

for i in range(len(df)):
    df['Image'][i] = np.array(list(map(float, df['Image'][i].split(' '))))
    df['Image'][i] = [[j] for j in df['Image'][i]]


    


# model = keras.models.Sequential()
# #model.add(keras.layers.Flatten(input_shape=[len(df['Image'][0])]))

# model.add(keras.layers.Dense(64, input_dim=9216, activation='relu'))
# model.add(keras.layers.Dense(64, activation='sigmoid'))
# model.add(keras.layers.Dense(31, activation='sigmoid'))

# model.add(keras.layers.Dense(len(feature_names)))

# model.summary()

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# model.fit(df[['Image']], df[feature_names], epochs=1)
