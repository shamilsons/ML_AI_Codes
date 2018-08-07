# -*- coding: utf-8 -*-

# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
#from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from random import *
import matplotlib.pyplot as plt
from string import *
import numpy as np

'''
1. Number of times pregnant
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)
9. Class variable (0 or 1)

Attribute number:    Mean:   Standard Deviation:
    1.                     3.8     3.4
    2.                   120.9    32.0
    3.                    69.1    19.4
    4.                    20.5    16.0
    5.                    79.8   115.2
    6.                    32.0     7.9
    7.                     0.5     0.3
    8.                    33.2    11.8
'''

#Read pima dataset file and extract instances
fp = open('pima_indians_diabetes.txt', 'r')
cnt=1

data=[]
target=[]
for line in fp.xreadlines():
    instance=split(line,',')
    dp = []
    for idx in range(0,8,1):
        dp.append(float(instance[idx]))
    data.append(dp)

    target.append(float(instance[8]))
    #print cnt,'.',line
    cnt+=1

'''
option=raw_input('Do you want to normalize-n or standardize-s data or none ? : ')
if(option=='n'):
    data_norm=preprocessing.normalize(data)
elif(option=='s'):
   data_norm=preprocessing.scale(data)
else:
    data_norm=data
'''
for dp,trg in zip(data, target):
    print dp,'->',trg

data_norm=data

fig = plt.figure()
ax = fig.add_subplot(111)
type1=ax.scatter([d[5] for d in data_norm], [d[7] for d in data_norm], color='red')
#type2=ax.scatter(dataset.data[:,0][50:100], dataset.data[:,1][50:100], color='blue')
#type3=ax.scatter(dataset.data[:,0][100:150], dataset.data[:,1][100:150], color='green')
ax.set_title('Pima Diabetes Dataset', fontsize=14)
ax.set_xlabel('---')
ax.set_ylabel('---')
#ax.legend([type1, type2, type3], ["Iris Setosa", "Iris Versicolor", "Iris Virginica"], loc=1)
ax.grid(True,linestyle='-',color='0.75')
#y label
#plt.ylabel('Y Label')
# x label
#plt.xlabel('X Label')
plt.show()

#below line splits data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(data_norm, target, test_size=0.25, random_state=0)
#dt=DecisionTreeClassifier()
#dt.fit(X_train,Y_train)

knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,Y_train)

#nb = GaussianNB()
#nb.fit(X_train,Y_train)

#print len(data)
#print X_train
#print len(X_test)

expectedClass = Y_test
predictedClassKNN = knn.predict(X_test)

print '\n==== PREDICTION ACCURACIES OF ALGORITHMS =====\n'
print 'KNN Accuracy :',metrics.accuracy_score(expectedClass,predictedClassKNN),'--- MSE:',metrics.mean_squared_error(expectedClass,predictedClassKNN)

#predictedClassDT = dt.predict(X_test)
#print 'DT Accuracy  :',metrics.accuracy_score(expectedClass,predictedClassDT), '--- MSE:',metrics.mean_squared_error(expectedClass,predictedClassDT)

#predictedClassNB = nb.predict(X_test)
#print 'NB Accuracy  :',metrics.accuracy_score(expectedClass,predictedClassNB),'--- MSE:',metrics.mean_squared_error(expectedClass,predictedClassNB)

print '\n\n==== PREDICTION METRICS OF ALGORITHMS =====\n'
print 'KNN class prd:',knn.predict([[8,183,64,0,0,23.3,0.672,32]]),' --- prob:',knn.predict_proba([[8,183,64,0,0,23.3,0.672,32]])
#print 'DT  class prd:',dt.predict([[8,183,64,0,0,23.3,0.672,32]]),' --- prob:',dt.predict_proba([[8,183,64,0,0,23.3,0.672,32]])
#print 'NB  class prd:',nb.predict([[8,183,64,0,0,23.3,0.672,32]]),' --- prob:',nb.predict_proba([[8,183,64,0,0,23.3,0.672,32]])

#print data[0:1][0]
#print [d[0] for d in data]