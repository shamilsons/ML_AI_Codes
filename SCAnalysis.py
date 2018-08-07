# -*- coding: utf-8 -*-

# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from random import *
import matplotlib.pyplot as plt
import scipy.stats as stats
from string import *
import numpy as np
import pandas as pd

#Loading the dataset
dataset = datasets.load_iris()

#print dataset.keys()
#print dataset.data.shape
#print dataset.feature_names

#Performing normalization or standardization techniques
option=raw_input('Do you want to normalize-n or standardize-s data or none ? : ')
if(option=='n'):
    data_norm=preprocessing.normalize(dataset.data)
elif(option=='s'):
    data_norm=preprocessing.scale(dataset.data)
else:
    data_norm=dataset.data

#print data_norm

#Creating DataFrame for the dataset and setting its features names like (sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm))
irisDtFrame=pd.DataFrame(data_norm, columns=['SPLLNG', 'SPLWDT', 'PTLLNG', 'PTLWDT'])
#irisDtFrame.columns = dataset.feature_names
#Add class attribute to the boston Dataframe
irisDtFrame['CLASS'] = dataset.target
#print irisDtFrame.head()

X_data=irisDtFrame.drop('CLASS', axis=1)

print '---- Correlation of Sepal and Petal Lengths in cm ----'
print 'Sepal correlation:',stats.pearsonr(X_data.SPLLNG, X_data.SPLWDT)
print 'Petal correlation:',stats.pearsonr(X_data.PTLLNG, X_data.PTLWDT)
#print X_data.head()
'''
#Combined scatter plot for correlational analysis
plt.figure(1).suptitle('Length verses Width size in cm', fontsize=14)
plt.subplot(211)
plt.scatter(X_data.SPLLNG, X_data.SPLWDT, color='g')
plt.xlabel('Sepal Length', fontsize=9)
plt.ylabel('Sepal Width')
plt.grid(True)
#plt.title('Prices vs Features', fontsize=12)
plt.subplot(212)
plt.scatter(X_data.PTLLNG, X_data.PTLWDT, color='r')
plt.xlabel('Pental Length', fontsize=9)
plt.ylabel('Pental Width')
plt.grid(True)
plt.show()
'''
X_train, X_test, Y_train, Y_test = train_test_split(X_data, irisDtFrame['CLASS'], test_size=0.35, random_state=3)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)

neigh = RadiusNeighborsClassifier(radius=1.0)
neigh.fit(X_train,Y_train)

dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)

nb = GaussianNB()
nb.fit(X_train,Y_train)

expectedClass = Y_test

print '\n==== PREDICTION ACCURACIES OF ALGORITHMS =====\n'
predictedClassKNN = knn.predict(X_test)
print 'KNN Accuracy  :',metrics.accuracy_score(expectedClass,predictedClassKNN),'--- MSE:',metrics.mean_squared_error(expectedClass,predictedClassKNN)

predictedClassRKNN = neigh.predict(X_test)
print 'RKNN Accuracy :',metrics.accuracy_score(expectedClass,predictedClassRKNN),'--- MSE:',metrics.mean_squared_error(expectedClass,predictedClassRKNN)

predictedClassDT = dt.predict(X_test)
print 'DT Accuracy   :',metrics.accuracy_score(expectedClass,predictedClassDT), '--- MSE:',metrics.mean_squared_error(expectedClass,predictedClassDT)

predictedClassNB = nb.predict(X_test)
print 'NB Accuracy   :',metrics.accuracy_score(expectedClass,predictedClassNB),'--- MSE:',metrics.mean_squared_error(expectedClass,predictedClassNB)
