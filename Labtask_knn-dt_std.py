# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:14:22 2016
"""

from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from random import *
#feature 1: 4.1 - 7.9
#feature 2 : 2.6 - 3.9
#feature 3 : 0.8 - 6.9
#feature 4 : 0.2 - 3.1

f1 = randrange(41, 79)/10.0

dataset = datasets.load_iris()
X, Y = dataset.data, dataset.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
#print(dataset.data)

#TRAIN 
model = DecisionTreeClassifier()
model.fit(X_train , Y_train)

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train , Y_train)

#Test
#model1 = DecisionTreeClassifier()
#model1.fit(X_test , Y_test)

#knn1 = KNeighborsClassifier(n_neighbors=3) 
#knn1.fit(X_test , Y_test)

d = list(X_test)

predictedKNN = []
predictedDT = []
for i in range(0,len(d)):
    f1 = randrange(41, 79)/10.0
    f2 = randrange(26, 39)/10.0
    f3 = randrange(8, 69)/10.0
    f4 = randrange(2, 31)/10.0
    print i,"dataset"
    print ("-— OUTPUT KNN —-")
    print (knn.predict([d[i]]))
    predictedKNN.append(knn.predict([d[i]]))
    predictedDT.append(model.predict([d[i]]))
    print(knn.predict_proba([d[i]]))
    print ("-— OUTPUT DT —-")
    print(model.predict([d[i]]))
    print(model.predict_proba([d[i]]))
    
print ("-— Accuracy!! —-")
print accuracy_score(Y_test,predictedKNN)
print accuracy_score(Y_test,predictedDT)

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:14:22 2016
@author: Syzdyk Yerbol 3enc03 Issatayev Assanali 3enc03
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import random
dataset = datasets.load_iris()
dataset1 = {'data': [], 'feature_names':dataset.feature_names,'target':[],'target_names':dataset.target_names}
dataset2 = {'data': [], 'feature_names':dataset.feature_names,'target':[],'target_names':dataset.target_names}
datasetr = {'data': [], 'feature_names':dataset.feature_names,'target':[],'target_names':dataset.target_names}
for i in range(50):
    datasetr['data'].append([random.randint(30,80)/10.0,random.randint(20,36)/10.0,random.randint(10,65)/10.0,random.randint(0,20)/10.0])
    datasetr['target'].append(random.randint(0,2))
for i in range(len(dataset.data)):
    if(i%50<25):
        dataset1['data'].append(dataset.data[i])
        dataset1['target'].append(dataset.target[i])
    else:
        dataset2['data'].append(dataset.data[i])
        dataset2['target'].append(dataset.target[i])
#dataset.target[0]=100
knn = KNeighborsClassifier(n_neighbors = 3)
model = DecisionTreeClassifier()
knn.fit(dataset1['data'],dataset1['target'])
model.fit(dataset1['data'],dataset1['target'])
print(model)
print(knn)
#dataset.target[0]=100
expectedt = dataset2['target']
test = model.predict(dataset2['data'])
test2 = knn.predict(dataset2['data'])
print("--------------------")
print('\nReport by DT:')
print(metrics.classification_report(expectedt,test))
print('\nReport by KNN:')
print(metrics.classification_report(expectedt,test2))
print('\nConfusion Matrix by DT:')
print(metrics.confusion_matrix(expectedt,test))
print('\nConfusion Matrix by KNN:')
print(metrics.confusion_matrix(expectedt,test2))
print('\nMean squared Error by DT:')
print(metrics.mean_squared_error(expectedt,test))
print('\nMean squared Error by KNN:')
print(metrics.mean_squared_error(expectedt,test2))
print('\nAccuracy Score by DT:')
print(metrics.accuracy_score(expectedt,test))
print('\nAccuracy Score by KNN:')
print(metrics.accuracy_score(expectedt,test2))
