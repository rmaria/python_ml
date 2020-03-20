#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:43:35 2020

@author: merlin
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 	
# Default value of display.max_rows is 10 i.e. at max 10 rows will be printed.
# Set it None to display all rows in the dataframe
pd.set_option('display.max_rows', None)

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:', s)

#df = pd.read_csv(s, header=None, encoding='utf-8')
df = pd.read_csv(s, header=None, encoding='utf-8')
classes = df[4].unique()
print("-------------")
print("classes: ",classes)
print(df.shape)
print(df.head())
print("-------------")

def split(df1, percentage=0.7):
    msk = np.random.rand(len(df1)) < percentage
    df1_train = df1[msk]
    df1_test = df1[~msk]
    
    return df1_train, df1_test

# Split the dataframe in the sub-constituents
setosa_df = df[df[4] == "Iris-setosa"]
versicolor_df = df[df[4] == "Iris-versicolor"]
#virginica_df = df[df[4] == "Iris-virginica"]


def merge_dataframes(df1, df2):
    frames = [df1, df2]
    result = pd.concat(frames)
    
    return result

new_df = merge_dataframes(setosa_df, versicolor_df)
train_df, test_df = split(new_df, percentage=0.7)



# training data is not dataframe, but numpy array
x_train = train_df.iloc[0:len(train_df), [0,1,2, 3]].values
x_test = test_df.iloc[0:len(test_df), [0,1,2, 3]].values


# labels
y_train = train_df.iloc[:,4].values
y_train = np.where(y_train== 'Iris-setosa', -1, 1)  # this puts -1 for the condition, 1 for other
y_test = test_df.iloc[:,4].values
y_test = np.where(y_test == 'Iris-setosa', -1, 1)


nb_train = x_train.shape[1]

class perceptron(object):
    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=1 + nb_train)
    def fit(self, x, y):
        #rgen = np.random.RandomState(self.random_state)
        #self.w_ = rgen.normal(loc=0.0, scale=0.1, size=1 + x.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
    def predict(self, x):
        return np.where(self.net_input(x) >=0, 1, -1)

ppn = perceptron(eta=0.1, n_iter=10)
print("weights before fit")
print(ppn.w_)

# fit
hist = ppn.fit(x_train, y_train)
print("weights after fit")
print(ppn.w_)

# predict
pred = ppn.predict(x_test)

# accuracy
res = np.where(y_test==pred, 1, 0)
acc = np.sum(res) / len(pred) * 100
print("acc = {} % ".format(acc))
