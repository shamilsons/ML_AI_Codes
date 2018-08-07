import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

nb = GaussianNB()
nb.fit(X, Y)

print 'Class predicted   :',nb.predict([[-0.8, -1]])
print 'Class probability :',nb.predict_proba([[-0.8, -1]])
