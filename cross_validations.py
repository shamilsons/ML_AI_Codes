from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import metrics

irisDB =load_iris() #150 records
X=irisDB.data
Y=irisDB.target

'''
scores_set={}
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=3)
#Add some details like test_size=0.1-0.9
#random_state=0,101
#n_neighbors=2-41
for nn in range(2,15):
    for tsz in range(7,10):
        tsz=tsz/10.0
        for smp in range(0,15):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=tsz, random_state=smp)
            #creating the instance of knn classifier
            knn=KNeighborsClassifier(n_neighbors=nn)
            #Training x -features,Y-class(label)
            knn.fit(X_train,y_train)
            y_pred=knn.predict(X_test)
            scores_set[str(nn)+'-'+str(tsz)+'-'+str(smp)]=metrics.accuracy_score(y_test,y_pred)
            #scores_set.append(metrics.accuracy_score(y_test,y_pred))
print len(scores_set.keys())
print scores_set
plt.plot(range(0,585),scores_set.values())
#plt.scatter(range(0,975),scores_set.values())
plt.xlabel('Random state')
plt.ylabel('Accuracy score')
plt.show()
'''
'''
kf=KFold(n_splits=5, shuffle=False)
#print '{}{:^61}{}'.format('Fold-No','TRAIN-SET','TEST-SET')
foldno=1
for train, test in kf.split(X_train):
    print foldno,' TRAIN-SET:', train
    print 'TEST-SET:',test
    foldno+=1
'''


k_range=range(2,41) #number of neighbors are going to be 1 to 40
k_scores=[] #store each mean accuracy for each neigbors number value
iteration=1
#print k_range
cls=[]
for k in k_range:
    knn2 = KNeighborsClassifier(n_neighbors=k)
    #scores = cross_val_score(knn2, irisDB.data, irisDB.target, cv=12)
    scores = cross_val_score(knn2, X_train, y_train, cv=10)
    #cls.append(knn2)
    #print scores.mean()
    #print iteration,'  accuracy:',scores.mean()
    #iteration+=1
    #print 'Scores for k:',k,'---',scores
    k_scores.append(scores.mean())

print k_scores

#print k_scores
#low values of k produces low bias but high variance
#high values of k produces high bias but low variance

plt.plot(k_range,k_scores)
#plt.scatter(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.show()

'''
kfold = KFold(n_splits=2)
#[knn.fit(X_train[train], y_train[train]).score(X_test[test], y_test[test]) for train, test in kfold.split(X_train)]
clf_scores={}
clfs={}
for k in range(1,41):
    scores = []
    knn = KNeighborsClassifier(n_neighbors=k)
    for train_indices, test_indices in kfold.split(X_train):
        #print('Train: %s | test: %s' % (train_indices, test_indices))
        knn.fit(X_train[train_indices], y_train[train_indices])
        scores.append(knn.score(X_train[test_indices], y_train[test_indices]))
    clf_scores[k]=sum(scores)/len(scores)
    clfs[k]=knn
#print clf_scores
plt.plot(range(1,41),clf_scores.values())
#plt.scatter(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.show()
y_pred=clfs[13].predict(X_test)
print metrics.accuracy_score(y_test,y_pred)
#print X_train[train_indices]
#print cls[17]
#print scores_set.append(metrics.accuracy_score(y_test,cls[17].predict(X_test)))
#knn3 = KNeighborsClassifier(n_neighbors=k)
# when we plot accuracies those which are on the middle produces better trade-off between bias and variance
#print scores
#print scores.mean()
'''
