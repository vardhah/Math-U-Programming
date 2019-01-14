# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:36:18 2019
@author: Sonal Kumari

To call this function usee the raw string data. put 'r' in front of string data
zlimit is by default =3 & -3, which is standard value. If you want to pass any other z pass it as 2nd argument

1. Sample function call using default z score   
#  zscore(r'C:\HPP\Desktop\data100.csv')

2. Sample function call using z score = +2 & -2   
#  zscore(r'C:\HPP\Desktop\data100.csv',2)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def zscore(filelocation,zlimit=3):
 

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
    
def ma(filelocation,days=7):

#Created on Sun Jan 13 15:36:18 2019
#@author: Sonal Kumari

#this function takes the 'file location' as input , you can also give the numbers of days taken in to 
#account for calculating the moving average. By default( if no argument is passed) it will calculate
#for 7 days
#1. Sample function call using default days (i.e. 7)  
#  ma(r'C:\HPP\Desktop\data100.csv')

#2. Sample function call using parameterised days(i.e 6)  
#  ma(r'C:\HPP\Desktop\data100.csv',6)


#setting mpving average set up
 print('\n=> File location: %s' % filelocation)

#Reading csv data from file
 data= pd.read_csv(filelocation);
#classifying data 
 X = data['xval'];
 Y = data['yval'];
 date=data['date'];
 Yplot= data
 i= len(Y)-days+1
 print('\n=> i(number of sets of yval) used for the iterator : %s'%i)
 print('-------')
 avg=[]

#moving average engine
 if(days%2 != 0 or days%2==0):
  for k in range(0,i):
   sum=0;
   print('set of yvals: ')
   for l in range(0+k,days+k):
       #print(k)
       #print(X[l])
       print(Y[l])
       #print(l)
       sum+=Y[l]
   avg.append(sum/days)   
   print('----')
   print('Average of above set of yval : %s'% avg[k])
   print('index(position in table) of avg for above -set of Yvals :  %s'%(l-1))
   print('===============++++++++++++=========================')  
   
 print('*  Length of data column(Y) :%s'% len(Y)) 
 print('*  Moving average value- output Array:')
 print(avg)  
 print('*  Length of output moving average array :%s'% len(avg))
 factor=(int(days/2)) 
 Xbasis= date[factor : (len(Y)-factor+1-(days%2))]
 #print('*  Modified X axis(Xbasis):')
 #print(Xbasis)
 #print('*  Length of modified x axis(Xbasis):%s'%len(Xbasis))
 Yplot=Y[factor:(len(Y)-factor+1-(days%2))]

#Plotting engine
 fig=plt.figure()
 
 #plotting the moving average 
 #print(len(avg))
 ax = fig.add_subplot(111)
 for n in range(0,len(avg)):
    ax.annotate(' %0.1f'%avg[n], xy=(Xbasis[factor+n],avg[n]))
 
 plt.title("Moving average Plot")
 plt.xlabel("Day Progression -->")
 plt.ylabel("Moving Average with days: %s"%days);
 plt.scatter(Xbasis, avg, color='blue', marker='o')
 plt.plot(Xbasis,avg, color='red',linewidth=2) 
 
 plt.figure()
 plt.title("Moving average Plot overlapped with data plot")
 plt.xlabel("Day Progression -->")
 plt.ylabel("Moving Average ; original data ");
 plt.plot(Xbasis,avg, color='red',linewidth=2,label="moving average") 
 plt.plot(Xbasis,Yplot, color='green',linewidth=0.5, label="Original Data") 
 plt.legend()
 
 plt.figure();
 plt.title("Original data Plot yval vs xval")
 plt.xlabel("xval")
 plt.ylabel("yval");
 plt.scatter(date, Y, color='black', marker='+')
 plt.plot(date,Y,color='green',linewidth=1)

#CSV data writing engine(it will generate the movingavg.csv file to folder where this code is stored)
 #d=np.array(Xbasis)
 #e=np.array(avg)
 #print(e)
 #print(d)
 #g={'date':d,'movingAverage':e}
 #df=pd.DataFrame(data=g)
 #print(df)
 #df.to_csv('movingavg.csv', encoding='utf-8', index=False)
 df1=pd.DataFrame(data[['date','xval','yval']])
 #print(df1)
 #s=np.append(avg,[None])
 #sd=np.append(None,s)
 #print(sd)
 #dfsd=pd.DataFrame(sd,columns=['m=%s'%days])
 #print(dfsd)
 #df = pd.DataFrame(avg,index=range(factor,len(Xbasis)+factor),columns=['m=%s'%days])
 #df[0]=None
 df1['m=%s'%days] = pd.Series(avg,index=range(factor,len(Xbasis)+factor))
 #df2=df1.append(dfsd,ignore_index=False,sort=True)
 #print(df)
 print('***   Final output data:')
 print(df1)
 df1.to_csv('movingavg.csv', encoding='utf-8', index=False)
 
 