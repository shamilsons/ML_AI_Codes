# -*- coding: utf-8 -*-
#This code implements kNN ml algorithm by using several ways.


#1. Create project in Python for the analysis of programming knowledge
#2. Announce projects in ML course

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import *
from random import *
from sklearn import preprocessing
import scipy.stats as stats

#data =load_diabetes()
dataset =load_iris()
#print dataset
#print list(dataset.target_names[0])

#x=[[0],[1],[2],[3]] # dataset with one feature
#y=[0,0,1,1] # class where dataset belongs to

#x=[[3,104],[2,100],[1,81],[101,10], [99,5], [98,2]]
#y=['R','R','R','A','A','A']

#creating the instance of knn classifier
knn=KNeighborsClassifier(n_neighbors=3)
dt=DecisionTreeClassifier()
# training the classifier
#knn.fit(x,y)

#Training x -features,Y-class(label)
knn.fit(dataset.data,dataset.target)
dt.fit(dataset.data,dataset.target)

print '--- OUTPUT ---'
for i in range(0,20):
    f1 = randrange(41, 79) / 10.0
    f2 = randrange(26, 39) / 10.0
    f3 = randrange(8, 69) / 10.0
    f4 = randrange(1, 31) / 10.0
    print i,'.dp:',f1,'-',f2,'-',f3,'-',f4
    print 'Knn class prd:',knn.predict([[f1,  f2,  f3,  f4]]), ' DT class prd:',dt.predict([[f1,  f2,  f3,  f4]])
    print 'Knn prob:',knn.predict_proba([[f1,  f2,  f3,  f4]]), 'DT prob:',dt.predict_proba([[f1,  f2,  f3,  f4]])
    
print stats.pearsonr(irisDataFrame['sepal length (cm)'],irisDataFrame['sepal width (cm)'])
b=preprocessing.normalize(irisDataFrame['sepal width (cm)'])
b=preprocessing.scale(irisDataFrame['sepal width (cm)'])
