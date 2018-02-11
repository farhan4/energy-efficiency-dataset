# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:06:21 2018

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
                 


#########################################################################
# Fitting Polynomial Regression to the Training set

from sklearn.preprocessing import PolynomialFeatures
poly_reg1 = PolynomialFeatures(degree = 3)
X_poly1 = poly_reg1.fit_transform(X)
poly_reg2 = PolynomialFeatures(degree = 2)
X_poly2 = poly_reg2.fit_transform(X)

from sklearn.cross_validation import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X_poly1, y1, test_size = 0.25, random_state = 0)

from sklearn.cross_validation import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X_poly2, y2, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LinearRegression

poly_reg1.fit(X1_train,y1_train)
poly_reg2.fit(X2_train,y2_train)

lin_reg1 = LinearRegression()
lin_reg1.fit(X1_train,y1_train)

lin_reg2 = LinearRegression()
lin_reg2.fit(X2_train,y2_train)


# Predicting a new result with Polynomial Regression

y1_pred = lin_reg1.predict(X1_test)
y2_pred = lin_reg2.predict(X2_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y1_test, y1_pred)
mean_squared_error(y2_test, y2_pred)









