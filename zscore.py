# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:36:18 2019
@author: Sonal Kumari

To call this function usee the raw string data. put 'r' in front of string data
zlimit is by default =3 & -3, which is standard value. If you want to pass any other z pass it as 2nd argument

1. Sample function call using default z score   
#  zscore(r'C:\HPP\Desktop\data100.csv',)

2. Sample function call using z score = +2 & -2   
#  zscore(r'C:\HPP\Desktop\data100.csv',)

"""


def zscore(filelocation,zlimit=3):
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt   #Data visualisation libraries 
 import seaborn as sns
 from sklearn import datasets, linear_model
 from sklearn.metrics import mean_squared_error, r2_score

#setting zscore set up
 print('File location: %s' % filelocation)


#Reading csv data from file
 data= pd.read_csv(filelocation);
#classifying data 
 X = data['xval'];
 Y = data['yval'];

# mean calculation using numpy mean function[numpy.mean]
 meanofY= np.mean(Y)
 print("Mean of Data = %s" % meanofY)
 standard_deviation_ofY= np.std(Y)
 print("standard deviation of Data = %s" % standard_deviation_ofY )

#calculating number of dataset
 numberofdatapoints=len(X);
 print('NumberofDatasets = %s'% numberofdatapoints)


#z-score cal engine
 z=[]
 for i in range(0, len(X)):
   z.append((Y[i]-meanofY)/ standard_deviation_ofY);
    
#print(z);    

#Plot setup for original data
 plt.figure();
 plt.title("Original data Plot")
 plt.xlabel("xval")
 plt.ylabel("Y value of data");
 plt.scatter(X, Y, color='black', marker='+')


#Plot setup for z-score  val(outlier boundry is setup in red(z+3 /z-3). Outlier dataset is shown in red )

 fig=plt.figure();
 ax = fig.add_subplot(111)
 plt.title("Z score Plot")
 plt.xlabel("xval")
 plt.ylabel("Z score");
 plt.plot(X,(0*X+zlimit), color='red',linestyle='dashed', linewidth=2)   #
 plt.plot(X,(0*X-zlimit), color='red', linestyle='dashed',linewidth=2)
 plt.plot( X,(0*X), color='blue', linewidth=1)
 for i in range(0,len(X)):
  if(z[i]>zlimit or z[i]<-zlimit):
    plt.scatter(X[i],z[i],  color='red',marker='x')
    ax.annotate('%s'%X[i], xy=(X[i],z[i]))
    
