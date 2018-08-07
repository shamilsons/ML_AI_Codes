'''
Name: Shahriar Shamiluulu
Email: shamilsons@gmail.com
Date: 19.09.2017
Description: Code to check the decision tree classifier
'''

#Import all needed libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

#load build in dataset
dataset =load_iris()

#below line splits data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=0)

#print type(dataset.data)
#print dataset.data,' ',dataset.target

#Create an instance of classifier
decisionTree=DecisionTreeClassifier(criterion='entropy', splitter='best')

#Train the classifier
#decisionTree.fit(dataset.data, dataset.target)
decisionTree.fit(X_train, Y_train)

predictedClass=decisionTree.predict(X_test)
print '\n==== PREDICTION ACCURACIES OF ALGORITHMS =====\n'

#1. Using predict and predict_probability function to test classifier
print 'DT class prd:',decisionTree.predict([[1.5,  4.2,  5.3,  1.8]])
print 'DT prob:',decisionTree.predict_proba([[1.5,  4.2,  5.3,  1.8]])

#2. By using metrics parameters
print 'Decision Tree Accuracy :',metrics.accuracy_score(Y_test,predictedClass),'--- MSE:',metrics.mean_squared_error(Y_test,predictedClass)

