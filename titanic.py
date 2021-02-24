# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:49:04 2021

@author: kesuiker
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


train_data_loc = "train.csv"
test = 'test.csv'

df = pd.read_csv(train_data_loc)
df_test = pd.read_csv(test)





feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare', 'female', 'male', 'Age','CC','Q','S','A','B','C','D','E','F','G','m']
target = ['Survived']

idx = df[df['Embarked'] == 'C'].index
df.loc[idx,'Embarked'] = 'CC'

idx = df[df['Cabin'] == 'T'].index
df.loc[idx, 'Cabin'] = 'A'

         
print("Using data: " + str(feature_names))

df_sex = pd.get_dummies(df['Sex'])
df_new = pd.concat([df, df_sex], axis=1)

df_new['Embarked'] = df_new['Embarked'].fillna('S')

df_embark = pd.get_dummies(df['Embarked'])
df_newer = pd.concat([df_new, df_embark], axis=1)

df_newer['Cabin'] = df_newer['Cabin'].fillna('m')

decks = 'ABCDEFG'

for i in range(7):
    idx = df_newer[df_newer['Cabin'].str.startswith(decks[i])].index
    df_newer.loc[idx, 'Cabin'] = decks[i]
df_garbage = df_newer
df_cabin = pd.get_dummies(df_newer['Cabin'])
df_newest = pd.concat([df_newer, df_cabin], axis=1)

age_by_pclass_sex = df_newest.groupby(['Sex', 'Pclass']).median()['Age']
df_newest['Age'] = df_newest.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

lr = LogisticRegression(max_iter=2000)

lr.fit(df_newest[feature_names], df[target].values.ravel())

df_newest.loc[:,'prediction'] = lr.predict(df_newest[feature_names])

print(confusion_matrix(df_newest[target], df_newest['prediction']))
print(classification_report(df_newest[target], df_newest['prediction']))


# Test data pre processing

idx = df_test[df_test['Embarked'] == 'C'].index
df_test.loc[idx,'Embarked'] = 'CC'

idx = df_test[df_test['Cabin'] == 'T'].index
df_test.loc[idx, 'Cabin'] = 'A'

df_dummy = pd.get_dummies(df_test['Embarked'])
df_test = pd.concat([df_test, df_dummy], axis=1)

df_dummy = pd.get_dummies(df_test['Sex'])
df_test = pd.concat([df_test, df_dummy], axis=1)

df_test['Cabin'] = df_test['Cabin'].fillna('m')

decks = 'ABCDEFG'

for i in range(7):
    idx = df_test[df_test['Cabin'].str.startswith(decks[i])].index
    df_test.loc[idx, 'Cabin'] = decks[i]
    
df_dummy = pd.get_dummies(df_test['Cabin'])
df_test = pd.concat([df_test, df_dummy], axis=1)

age_by_pclass_sex = df_test.groupby(['Sex', 'Pclass']).median()['Age']
df_test['Age'] = df_test.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)
# Prediction

df_test.loc[:,'Prediction'] = lr.predict(df_test[feature_names])

df_submit = df_test.filter(['PassengerId','Prediction'], axis=1)
df_submit = df_submit.rename(columns={'Prediction' : 'Survived'})

df_submit.to_csv('submission.csv', index = False)