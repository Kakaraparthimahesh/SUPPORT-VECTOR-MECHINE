# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:36:38 2022

@author: MAHESH
"""

# Required libriries

import pandas as pd 
import numpy as np 

# data processing 

fire_data = pd.read_csv("forestfires.csv") 
fire_data
fire_data.shape
list(fire_data)
fire_data.describe()
fire_data.head()
fire_data.info()

# Lable encoding 

from sklearn import preprocessing
lable_encoder=preprocessing.LabelEncoder()
fire_data["month"]=lable_encoder.fit_transform(fire_data["month"])
fire_data["day"]=lable_encoder.fit_transform(fire_data["day"])
fire_data["size_category"]=lable_encoder.fit_transform(fire_data["size_category"])
fire_data.head()

array = fire_data.values

fire_data
fire_data.isnull().sum()

# Splitting X and Y

X = array[:,0:29]
Y = array[:,30]

# Splitting Train and Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Model fitting 
# Linear

from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)

# poly 

from sklearn.svm import SVC
clf = SVC(kernel='poly')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)

# Loading SVC 
# Training a classifier - kernel='rbf'
# SVC()
# clf = SVC(kernel='linear')

# rbf

clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)

# Sigmoid

clf = SVC(kernel='sigmoid')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)

#---------------------------------------------------------------------------------
#Grid Search CV

from sklearn.model_selection import GridSearchCV

param_grid = [{'kernel':['poly'],'gamma':[0.5,0.1,0.01],'C':[10,0.1,0.001,0.0001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)

clf = SVC(C= 15, gamma = 100)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)

# Exploratort data analysis 

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=fire_data,x="month",y="size_category")
plt.show()

sns.boxplot(data=fire_data,x="day",y="size_category")
plt.show()

sns.boxplot(data=fire_data,x="FFMC",y="size_category")
plt.show()

sns.pairplot(data=fire_data,hue="size_category",height=2)
plt.show()

sns.boxplot(data=fire_data,x="month",y="size_category")
plt.show()

sns.boxplot(data=fire_data,x="day",y="size_category")
plt.show()

sns.boxplot(data=fire_data,x="FFMC",y="size_category")
plt.show()

