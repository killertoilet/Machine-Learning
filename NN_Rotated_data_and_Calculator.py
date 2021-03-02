# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:01:02 2021

@author: kesuiker



"""
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import copy
from tensorflow import keras
import random




def create_model():
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=784))
    model.add(keras.layers.Dense(32, activation='sigmoid'))
    model.add(keras.layers.Dense(32, activation='sigmoid'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy'\
                   , optimizer='adam', metrics=['accuracy'])
    
    return model

#K._get_available_gpus()
#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} )
#sess = tf.compat.v1.Session(config=config) 
#K.set_session(sess)



#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} ) 
#sess = tf.Session(config=config) 
#K.set_session(sess)

mnist = fetch_openml('mnist_784', version=1)

#pull data and target from mnist
x, y = mnist["data"], mnist["target"]

#convert y to ints
y = y.astype(np.uint8)

#seperate training data and test data
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

x_train_90 = copy.deepcopy(x[:30000])
x_train_180 = copy.deepcopy(x[:30000])
x_train_270 = copy.deepcopy(x[:30000])



for i in range(len(x_train_90)):
    #temp = x_train[i].reshape(28,28)
    x_train_90[i] = np.rot90(x_train_90[i].reshape(28,28)).reshape(784)
    x_train_180[i] = np.rot90(x_train_180[i].reshape(28,28), 2).reshape(784)
    x_train_270[i] = np.rot90(x_train_270[i].reshape(28,28), 3).reshape(784)
    

#add the rotated training data to one  large data set
x_train_rot = np.vstack((x_train, x_train_90))
x_train_rot = np.vstack((x_train_rot, x_train_180))
x_train_rot = np.vstack((x_train_rot, x_train_270))

y_train_rot = np.concatenate((y_train, y_train[:30000]))
y_train_rot = np.concatenate((y_train_rot, y_train[:30000]))
y_train_rot = np.concatenate((y_train_rot, y_train[:30000]))


#create the models
model_orig = create_model()
model_rot = create_model()


# turn data into DF
target_rot = pd.DataFrame({'Target': y_train_rot})
target_orig = pd.DataFrame({'Target' : y_train})

df_dummy = pd.get_dummies(target_rot['Target'])
target_rot = pd.concat([target_rot, df_dummy], axis=1)

df_dummy = pd.get_dummies(target_orig['Target'])
target_orig = pd.concat([target_orig, df_dummy], axis=1)

targets = [0,1,2,3,4,5,6,7,8,9]
features = list(range(784))

x_train_rot_df = pd.DataFrame(x_train_rot)
x_train_orig_df = pd.DataFrame(x_train)

# fit 2 seperate models, one with rotated data 
# and one with the original data set

# accuracies for rotated model = 85%
# accuracies for original = 95%
 
#model_rot.fit(x_train_rot_df[features], target_rot[targets], epochs=100)
model_rot.fit(x_train_orig_df[features], target_orig[targets], epochs=100)
prediction = model_rot.predict(x_test,)

# I believe the differences in accuracies is because the rotated model has 
# many different configurations now that have the same label. For example a rotated 5
# kind of looks like a 2, which throws off the algorithm. 




# create the training data
def create_training_data(n):
    x_list = []
    x_sqrt_list = []
    neg_sqrt_list = []
    
    for i in range(n):
        x = random.randint(0,100)
        x_sqrt = np.sqrt(x)
        
        x_list.append(x)
        x_sqrt_list.append(x_sqrt)
        neg_sqrt_list.append(-1*x_sqrt)
    
    df = pd.DataFrame({'x' : x_list, 'Sqrt(x)' : x_sqrt_list, '-Sqrt(x)' : neg_sqrt_list})
    
    return df



def calc_mean_squared_error(true, predict):
    
    mse = np.square(np.subtract(true,predict)).mean()
        
    return mse
        
    
# number of data points
n = 100000

df = create_training_data(n)

model = keras.models.Sequential()

model.add(keras.layers.Input(shape=1))
model.add(keras.layers.Dense(20, activation='sigmoid'))
model.add(keras.layers.Dense(2, activation=None))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

targets = ['Sqrt(x)', '-Sqrt(x)']
model.fit(df['x'], df[targets], epochs = 3)

test = [-20, 200]
prediction = model.predict(test)
#model doesnt do a great job, test data is much different that training data

nums = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
pred = model.predict(nums)

result = []
for i in range(len(nums)):
    mse = calc_mean_squared_error(np.sqrt(nums[i]), pred[i][0])
    result.append(mse)
    
print('MSE for the following inputs: \n')
print(nums)
print('\n')
print(result)