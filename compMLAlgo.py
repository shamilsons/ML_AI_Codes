# -*- coding: utf-8 -*-

# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from random import *
import matplotlib.pyplot as plt

# load the iris datasets
dataset = datasets.load_iris()
#print dataset['target_names']
classType={'0':0, '1':0, '2':0}
#print dataset.data[:,0]
#print dataset.data[:,0][0:50]

fig = plt.figure()
ax = fig.add_subplot(111)
type1=ax.scatter(dataset.data[:,0][0:50], dataset.data[:,1][0:50], color='red')
type2=ax.scatter(dataset.data[:,0][50:100], dataset.data[:,1][50:100], color='blue')
type3=ax.scatter(dataset.data[:,0][100:150], dataset.data[:,1][100:150], color='green')
ax.set_title('Petal size from Iris dataset', fontsize=14)
ax.set_xlabel('Petal length (cm)')
ax.set_ylabel('Petal width (cm)')
ax.legend([type1, type2, type3], ["Iris Setosa", "Iris Versicolor", "Iris Virginica"], loc=1)
ax.grid(True,linestyle='-',color='0.75')
#y label
#plt.ylabel('Y Label')
# x label
#plt.xlabel('X Label')
plt.show()

'''
for idx in range(0,150,1):
    if(dataset['target'][idx]==0):
        classType['0']+=1
    elif(dataset['target'][idx]==1):
        classType['1']+=1
    else:
        classType['2'] += 1
    print 'Data set:',dataset['data'][idx],' class:',dataset['target'][idx]
print classType

#knn Classifier model
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(dataset.data,dataset.target)
print(knn)

# fit a CART model to the data
dt = DecisionTreeClassifier()
dt.fit(dataset.data, dataset.target)
print(dt)

print '--- OUTPUT ---'
for i in range(1,10):
    f1 = randrange(41, 79) / 10.0
    f2 = randrange(26, 39) / 10.0
    f3 = randrange(8, 69) / 10.0
    f4 = randrange(1, 31) / 10.0
    print i,'.dp:',f1,'-',f2,'-',f3,'-',f4
    print 'KNN class prd:',knn.predict([[f1,  f2,  f3,  f4]]), ' DT class prd:',dt.predict([[f1,  f2,  f3,  f4]])
    print 'KNN prob:',knn.predict_proba([[f1,  f2,  f3,  f4]]), 'DT prob:',dt.predict_proba([[f1,  f2,  f3,  f4]])
'''
# make predictions
#expected = dataset.target
#predicted = dt.predict(dataset.data)

# summarize the fit of the model
#print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))
#print(metrics.mean_squared_error(expected, predicted))