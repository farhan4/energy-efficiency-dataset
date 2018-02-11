# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 23:58:17 2018

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
                 

from sklearn.cross_validation import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.25, random_state = 0)

from sklearn.cross_validation import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.25, random_state = 0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
reg1 = RandomForestRegressor(n_estimators = 50, random_state = 0)
reg2 = RandomForestRegressor(n_estimators = 200, random_state = 0)
reg1.fit(X1_train, y1_train)
reg2.fit(X2_train, y2_train)


y1_pred = reg1.predict(X1_test)
y2_pred = reg2.predict(X2_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y1_test, y1_pred)
mean_squared_error(y2_test, y2_pred)

