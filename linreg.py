# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 19:08:31 2018

@author: HPP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


data= pd.read_csv("C:\harsh\ML&NN_codes\ex1data.csv");
data.head()
data.info()
data.describe()
data.columns
X = data['x'];
X= X.values.reshape(-1,1);
Y = data['y'];

regr = linear_model.LinearRegression();

# Train the model using the training sets
regr.fit(X, Y);
print("Model Coefficients:")
print(regr.coef_);
print("Model Intercept:")
print(regr.intercept_);
test=[[0],[3.5],[ 7] ,[10], [12], [15],[21],[24]];
#test= test.values.reshape(-1,1)
pred_y= regr.predict(test);
print("Predicted values:")
print(pred_y);

# The mean squared error
#print("Mean squared error: %.2f" % mean_squared_error(,Y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test, pred_y))

# Plot outputs
plt.scatter(X, Y,  color='black')
plt.scatter(test, pred_y,  color='red')
plt.plot(test, pred_y, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()