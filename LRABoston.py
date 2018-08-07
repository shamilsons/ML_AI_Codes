# -*- coding: utf-8 -*-

#from numpy import *
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt #http://www.python-course.eu/matplotlib_multiple_figures.php
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import math

bostonDataset = datasets.load_boston()

#print bostonDataset.keys()
#print bostonDataset.data.shape
#print bostonDataset.feature_names
#print bostonDataset.DESCR

bostonDFrame=pd.DataFrame(bostonDataset.data)
bostonDFrame.columns = bostonDataset.feature_names
#Add class attribute to the boston Dataframe
bostonDFrame['PRICE'] = bostonDataset.target
#print bostonDFrame.head()

X=bostonDFrame.drop('PRICE', axis=1)

lm=LinearRegression()
lm.fit(X,bostonDFrame.PRICE)

plt.figure(1).suptitle('Prices vs Features', fontsize=14)

plt.subplot(321)
plt.scatter(bostonDFrame.INDUS, bostonDFrame.PRICE, color='g')
plt.xlabel('INDUS', fontsize=9)
#plt.ylabel('Prices')
plt.grid(True)
plt.title('Prices vs Features', fontsize=12)

plt.subplot(322)
plt.scatter(bostonDFrame.RM, bostonDFrame.PRICE, color='r')
plt.xlabel('RM', fontsize=9)
#plt.ylabel('Prices')
plt.grid(True)

plt.subplot(323)
plt.scatter(bostonDFrame.TAX, bostonDFrame.PRICE, color='b')
plt.xlabel('TAX', fontsize=9)
#plt.ylabel('Prices')
plt.grid(True)

plt.subplot(324)
plt.scatter(bostonDFrame.PTRATIO, bostonDFrame.PRICE, color='y')
plt.xlabel('PTRATIO', fontsize=9)
#plt.ylabel('Prices')
plt.grid(True)

plt.subplot(325)
plt.scatter(bostonDFrame.LSTAT, bostonDFrame.PRICE, color='r')
plt.xlabel('LSTAT', fontsize=9)
#plt.ylabel('Prices')
plt.grid(True)

#plt.subplot(411)
#plt.scatter(bostonDFrame.PRICE, bostonDFrame.RM, color='r')
#plt.xlabel('Prices')
#plt.ylabel('RM')
#plt.axis('equal')

plt.show()

'''
print 'Estimated intercept coefficient:',lm.intercept_
print 'Number of coefficients:', len(lm.coef_)
print pd.DataFrame(zip(X.columns,lm.coef_), columns=['Features','estimatedCoefficients'])
'''
'''
plt.scatter(bostonDFrame.RM, bostonDFrame.PRICE)
plt.xlabel('Average number of rooms per dwelling (PM)')
plt.ylabel('House Price')
plt.title('Relationship between PM and Price')
plt.show()
#Actual vs Predicted prices
plt.scatter(bostonDFrame.PRICE, lm.predict(X))
plt.xlabel('Prices: $Y_i$')
plt.ylabel('Predicted Prices: $\hat{Y}_i$')
plt.title('Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$')
plt.show()
'''

#print 'Correlation coefficient:',stats.pearsonr(Y_test, pred_test)
#Write code which shows the correlation between all features
#print 'Correlation coefficient:',stats.pearsonr(bostonDFrame['CRIM'], bostonDFrame['AGE'])
cnt=1
mainkey='PRICE'
for key in bostonDFrame.keys():
    if(key!=mainkey):
        corr=stats.pearsonr(bostonDFrame[mainkey], bostonDFrame[key])
        #print cnt, '.corr(r) between ', mainkey, ' and ', key, ' :', corr

        if(corr[0]<=-0.45):
            print cnt,'.corr(r) between ',mainkey,' and ',key,' :',corr
            cnt+=1
        elif (corr[0]>= 0.45):
            print cnt, '.corr(r) between ', mainkey, ' and ', key, ' :', corr
            cnt+=1
        else:
            continue
        #cnt += 1

#Mean square error calculation
mseTotal=np.mean((bostonDFrame.PRICE - lm.predict(X))**2)
print 'Mean squared error (MSE):',mseTotal
print 'Root mean squared error (RMSE):', math.sqrt(mseTotal)

X_train, X_test, Y_train, Y_test = train_test_split(X, bostonDFrame['PRICE'], test_size=0.33, random_state=5)
print 'X train dataset:',X_train.shape
print 'X test dataset:',X_test.shape
print 'Y train dataset:',Y_train.shape
print 'Y test dataset:',Y_test.shape

lm=LinearRegression()
lm.fit(X_train,Y_train)
pred_train=lm.predict(X_train)
pred_test=lm.predict(X_test)


print 'Fit a model X_train, and calculate RMSE with Y_train:', math.sqrt(np.mean((Y_train-lm.predict(X_train))**2))
print 'Fit a model X_train, and calculate RMSE with X_test, Y_test:', math.sqrt(np.mean((Y_test-lm.predict(X_test))**2))
'''
#plt.scatter(lm.predict(X_train), lm.predict(X_train)-Y_train, c='r', s=30, alpha=0.4)
plt.scatter(lm.predict(X_test), lm.predict(X_test)-Y_test, c='g', s=40, alpha=0.6)
plt.hlines(y=0, xmin=0, xmax=50)
plt.title('Residual plot using training(red) and testing(green) housing data')
plt.ylabel('Residuals')
plt.show()
'''
