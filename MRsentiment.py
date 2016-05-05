from __future__ import division
import numpy as np
import random
import skflow
import xgboost as xgb
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

np.random.seed(0) # seed to shuffle the train set

n_folds = 5
verbose = False
shuffle = False

X=np.loadtxt(open("revxtrainsw2.csv","rb"),delimiter=",")
#xutf=np.loadtxt(open("revxtraintf2.csv","rb"),delimiter=",")
#X=np.concatenate([X,xutf],axis=1)
X_hold=np.loadtxt(open("revxtestsw2.csv","rb"),delimiter=",")
#X_holdutf=np.loadtxt(open("revxtesttf2.csv","rb"),delimiter=",")
#X_hold=np.concatenate([X_hold,X_holdutf],axis=1)
y_hold=np.loadtxt(open("revytestsw2.csv","rb"),delimiter=",")
y=np.loadtxt(open("revytrainsw2.csv","rb"),delimiter=",")
#data_old=np.loadtxt(open("dataset_blend1.csv","rb"),delimiter=",")
#dtrain=xgb.Dmatrix(X,label=y)
#dtest=xgb.DMatrix(X_hold,label=y_hold

if shuffle:
    idx = np.random.permutation(y.size)
    X = X[idx]
    y = y[idx]

skf = list(StratifiedKFold(y, n_folds))

clfs = [KNeighborsClassifier(11),
	LogisticRegression(C=0.3),
	MultinomialNB(),
	svm.SVC(kernel="linear", C=0.5,probability=True,max_iter=5000),
	skflow.TensorFlowDNNClassifier(hidden_units=[1000,1000,1000],n_classes=2, steps=4000)]
	

print "Creating train and test sets for stacking."
    
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_hold.shape[0], len(clfs)))

#Training on n-folds
#Testing each fold on test set, then taking mean
#Build meta training and test set for stacking
for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((X_hold.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
	print "Fold", i
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:,1]
        dataset_blend_train[test, j] = y_prob
        dataset_blend_test_j[:, i] = clf.predict_proba(X_hold)[:,1]
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    y_pred=np.rint(dataset_blend_test[:,j])
    print(classification_report(y_hold,y_pred ))
    
#np.savetxt("dataset_blend_test4swtf.csv",dataset_blend_test,delimiter=",")
#np.savetxt("dataset_blend_train4swtf.csv",dataset_blend_train,delimiter=",")
#dataset_blend_test=np.concatenate([dataset_blend_test,data_old],axis=1)

print "Stacking LR:"
clf = LogisticRegression()
clf.fit(dataset_blend_train, y)
y_LR = clf.predict(dataset_blend_test)
print(classification_report(y_hold,y_LR ))

print "Stacking NN:"
clf=skflow.TensorFlowDNNClassifier(hidden_units=[50],n_classes=2, steps=4000)
clf.fit(dataset_blend_train, y)
y_NN = clf.predict(dataset_blend_test)
print(classification_report(y_hold,y_NN ))

print "Stacking BT:"
clf=GradientBoostingClassifier(n_estimators=400)
clf.fit(dataset_blend_train, y)
y_BT = clf.predict(dataset_blend_test)
print(classification_report(y_hold,y_BT ))

print "Prob. Voting:"
y_ave=np.sum(dataset_blend_test,axis=1)/dataset_blend_test.shape[1]
y_pred=np.rint(y_ave)
print(classification_report(y_hold,y_NN ))

#y_stack=np.concatenate([y_LR,y_NN,y_BT,y_pred],axis=1)
#np.savetxt("stack_pred4swtf.csv",y_stack,delimiter=",")

    

