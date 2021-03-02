# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:26:51 2021

@author: kesuiker
"""

# -*- coding: utf-8 -*-
"""

Task 1:
Train a neural network on the Facial Keypoints dataset from last week's lab to predict the 15 keypoints of the face.

Task 2:
Load your own face and run it through the trained network
"""

from tensorflow import keras
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

file = 'FacialKeypoints.csv'



# read the face data to data frame
df = pd.read_csv(file)

col_names = []

for col in df.columns:
    col_names.append(col)

feature_names = col_names[:(len(col_names)-1)]
target = col


df = df.dropna()
df_train = df.Image.str.split(" ", expand=True,).astype(float)

model = keras.models.Sequential()
 #model.add(keras.layers.Flatten(input_shape=[len(df['Image'][0])]))

model.add(keras.layers.Dense(16, input_dim=9216, activation='relu'))
model.add(keras.layers.Dense(64, activation='sigmoid'))
model.add(keras.layers.Dense(64, activation='sigmoid'))
model.add(keras.layers.Dense(30, activation=None))
model.add(keras.layers.Dense(len(feature_names)))

model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# fit the model using face data, run for 10 epochs
model.fit(df_train[df_train.columns[0:9216]], df[feature_names], epochs=10)


# import mat damon and reshpae the image
color_im = Image.open('mattdamon.jpg')
black_white_im = color_im.convert('L')
cropped_bw_np = np.array(black_white_im)[350:1150, 50:950]
resized_bw = Image.fromarray(cropped_bw_np).resize((96,96))

resized_bw_np = np.array(resized_bw)

resized_bw_shape = resized_bw_np.reshape(1,9216)

# predict the coordinatets and store them into mat variable
mat = model.predict(resized_bw_shape)

plt.imshow(resized_bw)


# plot the predicted coordinates
for k in range(0,30, 2):        
    plt.plot(mat[0][k], mat[0][k+1], 'r+')
    plt.axis('off')
