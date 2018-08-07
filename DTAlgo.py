# -*- coding: utf-8 -*-
#This code implements Decision Tree ml algorithm by using scikit-learn.

from time import time

from sklearn.datasets import *
#data =load_diabetes()
#data =load_iris()
#print data
#print list(data.target_names[0])

# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
actualClass = dataset.target
predictedClass = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(actualClass, predictedClass))
print(metrics.confusion_matrix(actualClass, predictedClass))
print(metrics.mean_squared_error(actualClass, predictedClass))
print(metrics.accuracy_score(actualClass, predictedClass))
print(metrics.homogeneity_score(actualClass, predictedClass))


'''
from sklearn import tree
import os
from sklearn.externals.six import StringIO
import pydot
#import pydotplus as pydot
from IPython.display import Image

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

os.unlink('iris.dot')

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,  feature_names=iris.feature_names,  class_names=iris.target_names,  filled=True, rounded=True,  special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
#print graph
Image(graph.create_png())
print 'done ...'
'''