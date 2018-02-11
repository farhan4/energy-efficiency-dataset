# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:25:25 2018

@author: farhan baig
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##########################################################################
# Data Pre Processing
dataset = pd.read_excel('energy.xlsx')
X = dataset.iloc[:,: - 2].values
y1 = dataset.iloc[:, 8].values
y2 = dataset.iloc[:, 9].values 
                 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y1 = StandardScaler()
sc_y2 = StandardScaler()
X = sc_X.fit_transform(X)
y1 = sc_y1.fit_transform(y1)
y2 = sc_y2.fit_transform(y2)

from sklearn.cross_validation import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.25, random_state = 0)

from sklearn.cross_validation import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.25, random_state = 0)

y1_test = sc_y1.inverse_transform(y1_test)
y2_test = sc_y2.inverse_transform(y2_test)

# Fitting SVR to the dataset
from sklearn.svm import SVR
reg1 = SVR(kernel = 'rbf',C = 1000)
reg2 = SVR(kernel = 'rbf',C = 100)

reg1.fit(X1_train,y1_train)
reg2.fit(X2_train,y2_train)

# Predicting a new result
y1_pred = reg1.predict(X1_test)
y1_pred = sc_y1.inverse_transform(y1_pred)
y2_pred = reg2.predict(X2_test)
y2_pred = sc_y2.inverse_transform(y2_pred)


from sklearn.metrics import mean_squared_error
mean_squared_error(y1_test, y1_pred)
mean_squared_error(y2_test, y2_pred)
                 