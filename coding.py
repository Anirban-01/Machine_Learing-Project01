# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:12:16 2018

@author: Anirban
"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('googleplaystore.csv')
#Replace Nan values with 0 and get new_dataset
new_dataset = dataset.fillna(0)
new_dataset

#dividing new dataset into x and y
X = new_dataset.iloc[:,[-2]].values
y = new_dataset.iloc[:,[-3]].values

#Implementing training and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#Implement Linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_predict = regressor.predict(X_test)
y_score = regressor.score(X_train, y_train)

#implementing graph
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.xlabel('Reviews')
plt.ylabel('Rating')
plt.show()



