# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:34:16 2018

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


#########################################################################
# Fitting MultipleLinear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize = True)
regressor.fit(X1_train, y1_train)
regressor.fit(X2_train, y2_train)

y1_pred = regressor.predict(X1_test)
y2_pred = regressor.predict(X2_test)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((768,1)).astype(int),values = X,axis = 1)

X_opt1 = X[:, [0,1,2,3,4,5,7,8]]
regressor_OLS1 = sm.OLS(endog = y1,exog = X_opt1).fit()
regressor_OLS1.summary()

X_opt2 = X[:, [0,1,2,3,4,5,7]]
regressor_OLS2 = sm.OLS(endog = y2,exog = X_opt2).fit()
regressor_OLS2.summary()

from sklearn.metrics import mean_squared_error
mean_squared_error(y1_test, y1_pred)
mean_squared_error(y2_test, y2_pred)

#########################################################################



