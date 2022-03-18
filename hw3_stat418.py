#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import numpy as np
import os
import tarfile
# Q1
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
# Q2
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# Q3)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model


# ### Question 1

# In[2]:


# 1.1) load the CIFAR-10 dataset
# importing via instructions: http://www.cs.toronto.edu/~kriz/cifar.html
f1 = '/Users/lisakaunitz/Desktop/STAT_418/stat418_w22/hw3/cifar-10-batches-py/data_batch_1'
def unpickle(f1):
    import pickle
    with open(f1, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
b1 = unpickle(f1)


f2 = '/Users/lisakaunitz/Desktop/STAT_418/stat418_w22/hw3/cifar-10-batches-py/data_batch_2'
def unpickle(f2):
    import pickle
    with open(f2, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
b2 = unpickle(f2)

f3 = '/Users/lisakaunitz/Desktop/STAT_418/stat418_w22/hw3/cifar-10-batches-py/data_batch_3'
def unpickle(f3):
    import pickle
    with open(f3, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
b3 = unpickle(f3)

f4 = '/Users/lisakaunitz/Desktop/STAT_418/stat418_w22/hw3/cifar-10-batches-py/data_batch_4'
def unpickle(f4):
    import pickle
    with open(f4, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
b4 = unpickle(f4)

f5 = '/Users/lisakaunitz/Desktop/STAT_418/stat418_w22/hw3/cifar-10-batches-py/data_batch_5'
def unpickle(f5):
    import pickle
    with open(f5, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
b5 = unpickle(f5)

f6 = '/Users/lisakaunitz/Desktop/STAT_418/stat418_w22/hw3/cifar-10-batches-py/test_batch'
def unpickle(f6):
    import pickle
    with open(f6, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
test_batch = unpickle(f6)


# In[3]:


# 1.2) Make 80/20 split on the dataset into test and train data
df_train = np.concatenate([b1[b'data'],
                     b2[b'data'], 
                     b3[b'data'], 
                     b4[b'data'], 
                     b5[b'data']])
y_train = np.concatenate([b1[b'labels'], b2[b'labels'], b3[b'labels'], b4[b'labels'], b5[b'labels']])
print(df_train.shape)

df_test = test_batch[b'data']
y_test = test_batch[b'labels']
print(df_test.shape)


# In[4]:


# 1.3) Scale the data so that each feature has a minimum value of 0 and a maximum value of 1.
scaler = MinMaxScaler(feature_range=(0,1))

X_train = scaler.fit_transform(df_train)
X_test = scaler.fit_transform(df_test)


# In[5]:


# 1.4.a) PCA
pca = PCA(n_components = 2)

train_pca = pca.fit_transform(X_train)
test_pca = pca.fit_transform(X_test)

explained_variance = pca.explained_variance_ratio_
explained_variance


# In[6]:


# 1.4.b) SVD
svd = TruncatedSVD(n_components = 2)
train_svd = svd.fit_transform(X_train)
test_svd = svd.fit_transform(X_test)


# In[7]:


# 1.4.c) Non-negative Matrix Factorization
from sklearn.decomposition import NMF
nmf = NMF(n_components = 2, init='random', random_state=0)
train_nmf = nmf.fit_transform(X_train)
test_nmf = nmf.fit_transform(X_test)


# ### Question 2

# In[8]:


# 2.1.a) Linear SVC
lsvc = LinearSVC(verbose=0, max_iter=10)
lsvc.fit(X_train, y_train)
pred_lsvc = lsvc.predict(X_test)
score = lsvc.score(X_train,y_train)
print("Score: ", score)

# Report various metrics for the fitted models, such as averaged precision, recall, f1 score and accuracy on the test data.
print(classification_report(y_test, pred_lsvc))
print(confusion_matrix(y_test, pred_lsvc))
print('Accuracy score:', accuracy_score(pred_lsvc, y_test))


# In[9]:


# 2.1.b) Logistic Regression classifier
glm = LogisticRegression(max_iter = 50)
glm.fit(X_train, y_train)
pred_glm = glm.predict(X_test)

# Report various metrics for the fitted models, such as averaged precision, recall, f1 score and accuracy on the test data.
print(classification_report(y_test, pred_glm))
print('Accuracy score:', accuracy_score(pred_glm, y_test))


# In[10]:


# 2.1.c) KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

# Report various metrics for the fitted models, such as averaged precision, recall, f1 score and accuracy on the test data.
print(classification_report(y_test, pred_knn))
print('Accuracy score:', accuracy_score(pred_knn, y_test))


# In[11]:


# 2.1.d) Perceptron
perc = Perceptron(tol=1e-3, random_state=0)
perc.fit(X_train, y_train)
pred_perc = perc.predict(X_test)

# Report various metrics for the fitted models, such as averaged precision, recall, f1 score and accuracy on the test data.
print(classification_report(y_test, pred_perc))
print('Accuracy score:', accuracy_score(pred_perc, y_test))


# ### Question 3

# In[12]:


# Prep
pca = PCA()
svd = TruncatedSVD()
glm = LogisticRegression()
knn = KNeighborsClassifier()
perc = Perceptron()
lasso = linear_model.Lasso()
ridge = linear_model.Ridge()


# In[13]:


# PCA

pipe = Pipeline(steps = [('scaler', scaler), ('pca', pca), ('svd', svd)])
param_grid = {
    'pca__n_components': [1,2,3],
    'svd__n_components': [5,10,15],
}

grid1 = GridSearchCV(pipe, param_grid, cv = 5)
grid1.fit(X_train, y_train)

print('Best parameter (CV score=%0.3f):' % grd.best_score_)
print(grid1.best_params_)


# In[14]:


# logistic regression

pipe = Pipeline(steps=[('glm', glm),('svd', svd)])
param_grid = {
    'glm__C:' np.logspace(-4,4,4),
    'svd__n_components': [5,10,15]
}

grid2 = GridSearchCV(pipe, param_grid, cv=5)
grid2.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % grd.best_score_)
print(grid2.best_params_)


# In[15]:


# KNN

pipe = Pipeline(steps=[('knn', knn), ('pca', pca)])
param_grid = {
    'knn__n_neighbors:' [5,10,15],
    'pca__n_components:' [1,2,3]
}

grid3 = GridSearchCV(pipe, param_grid, cv=5)
grid3.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % grd.best_score_)
print(grid3.best_params_)


# In[16]:


# Perceptron

pipe = Pipeline(steps=[('perc', perc), ('pca', pca)])
param_grid = {
    'perceptron__max_iter:' [5,10,50] #save time
}

grid4 = GridSearchCV(pipe, param_grid, cv=5)
grid4.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % grd.best_score_)
print(grid4.best_params_)


# In[17]:


# 3.2) Many of the accuracy scores ranged around 0.26 to 0.35, this was low because of the components taken to run. I think if I had more time on my machine to run this and more space we could optimize the modles.
# Overall: This was something I have never done before this class, and it is a tool I look forward to using for tuning paramaters and ulitmately optimizing models in my future work.


