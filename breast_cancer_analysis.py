'''
Name: Shahriar Shamiluulu
Email: shamilsons@gmail.com
Date: 19.09.2017
Description: Code to analyze and create classifier for Breast cancer data
'''

#Import all needed libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
'''
Breast cancer data description
 #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
'''

dataset=pd.read_csv('breast-cancer-wisconsin.data')
dataset.columns = ['id','f1','f2','f3','f4','f5','f6','f7','f8','f9','class']
#print dataset

dataset.replace('?', 0, inplace=True)
dataset.drop(['id'], 1, inplace=True)

x=np.array(dataset.drop(['class'],1))
y=np.array(dataset['class'])

#below line splits data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=7)

#Create an instance of classifiers
knn=KNeighborsClassifier(n_neighbors=3)
decisionTree=DecisionTreeClassifier(criterion='entropy', splitter='best')

#Train the classifier
knn.fit(X_train, Y_train)
decisionTree.fit(X_train, Y_train)

predictedClassKNN=knn.predict(X_test)
predictedClassDT=decisionTree.predict(X_test)
print '\n==== PREDICTION ACCURACIES OF ALGORITHMS =====\n'
print 'KNN Accuracy :',metrics.accuracy_score(Y_test,predictedClassKNN),'--- MSE:',metrics.mean_squared_error(Y_test,predictedClassKNN)
print 'Decision Tree Accuracy :',metrics.accuracy_score(Y_test,predictedClassDT),'--- MSE:',metrics.mean_squared_error(Y_test,predictedClassDT)
print 'Decision Tree CONF-MATRIX :',metrics.confusion_matrix(Y_test,predictedClassDT)

fpr, tpr, thresholds = metrics.roc_curve(Y_test, predictedClassDT, pos_label=2)
print metrics.auc(fpr, tpr)