# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:38:53 2018

@author: HOSSAM ABDELHAMID
"""
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
# This code is for classifying features_3 located in feature3_test.mat
import numpy as np
from scipy.io import loadmat

x1 = loadmat('feature3.mat')
y1 = loadmat('label.mat')
x_test = loadmat('feature3_test.mat')
y_test = loadmat('label_test.mat')
# Load numpy
samples_train=200000
samples_test=50799
X=x1['arr']
y=y1['arr']
Xtest=x_test['arr']
Ytest=y_test['arr']
y=np.reshape(y,(samples_train,))
Ytest=np.reshape(Ytest,(samples_test,))
# Set random seed
np.random.seed(0)


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
y_pred=clf.predict(X)
print(100*np.sum(y_pred==y)/samples_train)
print(np.sum(y_pred==1)/np.sum(y==1))
ytest_pred=clf.predict(Xtest)
print(100*np.sum(ytest_pred==Ytest)/samples_test)
print(np.sum(ytest_pred==1)/np.sum(Ytest==1))
