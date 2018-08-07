from sklearn import datasets
import pandas as pd
from sklearn import preprocessing
import scipy.stats as stats
from math import log
import numpy as np
import matplotlib.pyplot as plt

irisdata = datasets.load_iris()

#print irisdata
#print irisdata.data
#print irisdata.target

irisDataFrame=pd.DataFrame(irisdata.data)
#irisDataFrame.columns=irisdata.target_names
#print irisdata.feature_names
#print irisDataFrame[0]='A'
irisDataFrame.columns=irisdata.feature_names
irisDataFrame['sepal length (cm)']
print irisDataFrame
#print stats.pearsonr(preprocessing.normalize(irisDataFrame['sepal length (cm)']), preprocessing.normalize(irisDataFrame['sepal width (cm)']))
a=preprocessing.normalize(irisDataFrame['sepal length (cm)'])
b=preprocessing.normalize(irisDataFrame['sepal width (cm)'])
#print a
print stats.pearsonr(irisDataFrame['sepal length (cm)'],irisDataFrame['sepal width (cm)'])
'''
fig = plt.figure()
ax = fig.add_subplot(111)
type1=ax.scatter()
ax.set_title('Scatter plot', fontsize=14)
ax.set_xlabel('sepal length (cm)')
ax.set_ylabel('petal length (cm)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
type1=ax.scatter()
ax.set_title('Scatter plot', fontsize=14)
ax.set_xlabel('sepal length (cm)')
ax.set_ylabel('petal length (cm)')
plt.show()
'''

plt.figure(1).suptitle('IRIS DATASET', fontsize=14)

plt.subplot(321)
plt.scatter(irisDataFrame['sepal length (cm)'],irisDataFrame['petal length (cm)'], color='blue',alpha=0.5)
plt.xlabel('Width', fontsize=9)
#plt.ylabel('Prices')
plt.grid(True)
#plt.title('Prices vs Features', fontsize=12)

plt.subplot(322)
plt.scatter(a,b, color='blue',alpha=0.5)
plt.xlabel('Width', fontsize=9)
#plt.ylabel('Prices')
plt.grid(True)
#plt.title('Prices vs Features', fontsize=12)

plt.subplot(323)
plt.scatter(irisDataFrame['sepal length (cm)'],irisDataFrame['sepal width (cm)'], color='red')
plt.xlabel('Width', fontsize=9)
#plt.ylabel('Prices')
plt.grid(True)

plt.subplot(324)
plt.scatter(preprocessing.normalize(irisDataFrame['sepal length (cm)']),preprocessing.normalize(irisDataFrame['sepal width (cm)']), color='red')
plt.xlabel('Width', fontsize=9)
#plt.ylabel('Prices')
plt.grid(True)

plt.show()

#print np.log2(a)
#print stats.pearsonr(np.log2(a[0]),np.log2(b[0]))
#print stats.pearsonr(irisDataFrame['sepal length (cm)'], irisDataFrame['petal length (cm)'])
#print stats.pearsonr(irisDataFrame['sepal length (cm)'], irisDataFrame['petal width (cm)'])

#print preprocessing.normalize(irisDataFrame['sepal length (cm)'])
#print preprocessing.scale(irisDataFrame['sepal length (cm)'])
