from __future__ import division
import numpy
import random
import skflow
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

X=numpy.loadtxt(open("revxtrain.csv","rb"),delimiter=",")
X_hold=numpy.loadtxt(open("revxtest.csv","rb"),delimiter=",")
y_hold=numpy.loadtxt(open("revytest.csv","rb"),delimiter=",")
y=numpy.loadtxt(open("revytrain.csv","rb"),delimiter=",")


if shuffle:
    idx = numpy.random.permutation(y.size)
    X = X[idx]
    y = y[idx]

skf = list(StratifiedKFold(y, n_folds))

clfs = [KNeighborsClassifier(11),
	LogisticRegression(C=0.3),
	MultinomialNB(),
	svm.SVC(kernel="linear",C=0.5,probability=True,max_iter=5000)]
	skflow.TensorFlowDNNClassifier(hidden_units=[1000,1000,1000],n_classes=2, steps=5000)]


#Train full training set, test on test set
for j, clf in enumerate(clfs):
    print j, clf
    clf.fit(X, y)
    y_pred=clf.predict(X_hold)
    print(classification_report(y_hold,y_pred ))
