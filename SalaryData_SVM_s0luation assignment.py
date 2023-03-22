# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 02:49:37 2022

@author: MAHESH
"""

# DATA PROCESSING 

import pandas as pd
import numpy as np

SalaryData_Test_data = pd.read_csv("SalaryData_Test(1).csv")
SalaryData_Train_data = pd.read_csv("SalaryData_Train(1).csv")

SalaryData_data = pd.concat([SalaryData_Test_data,SalaryData_Train_data])
SalaryData_data

SalaryData_data.shape
SalaryData_data.head()
list(SalaryData_data)
SalaryData_data.info()
SalaryData_data.isnull().sum()

# Lable encoding 

from sklearn import preprocessing
lable_encoder=preprocessing.LabelEncoder()
SalaryData_data["workclass"]=lable_encoder.fit_transform(SalaryData_data["workclass"])
SalaryData_data["education"]=lable_encoder.fit_transform(SalaryData_data["education"])
SalaryData_data["maritalstatus"]=lable_encoder.fit_transform(SalaryData_data["maritalstatus"])
SalaryData_data["occupation"]=lable_encoder.fit_transform(SalaryData_data["occupation"])
SalaryData_data["relationship"]=lable_encoder.fit_transform(SalaryData_data["relationship"])
SalaryData_data["race"]=lable_encoder.fit_transform(SalaryData_data["race"])
SalaryData_data["sex"]=lable_encoder.fit_transform(SalaryData_data["sex"])
SalaryData_data["native"]=lable_encoder.fit_transform(SalaryData_data["native"])
SalaryData_data["Salary"]=lable_encoder.fit_transform(SalaryData_data["Salary"])

SalaryData_data
array = SalaryData_data.values

SalaryData_data.isnull().sum()


# Splitting X and Y

X = array[:,0:12]
Y = array[:,13]

# Model fitting 
# Linear

from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X, Y)
y_pred = clf.predict(X)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y, y_pred)

acc = accuracy_score(Y, y_pred) * 100
print("Accuracy =", acc)

# poly 

from sklearn.svm import SVC
clf = SVC(kernel='poly')
clf.fit(X, Y)
y_pred = clf.predict(X)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y, y_pred)

acc = accuracy_score(Y, y_pred) * 100
print("Accuracy =", acc)

# Loading SVC 
# Training a classifier - kernel='rbf'
# SVC()
# clf = SVC(kernel='linear')

# rbf

from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(X, Y)
y_pred = clf.predict(X)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y, y_pred)

acc = accuracy_score(Y, y_pred) * 100
print("Accuracy =", acc)

# Sigmoid

from sklearn.svm import SVC
clf = SVC(kernel='sigmoid')
clf.fit(X, Y)
y_pred = clf.predict(X)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y, y_pred)

acc = accuracy_score(Y, y_pred) * 100
print("Accuracy =", acc)

#---------------------------------------------------------------------------------

#Grid Search CV

from sklearn.model_selection import GridSearchCV

param_grid = [{'kernel':['poly'],'gamma':[0.5,0.1,0.01],'C':[10,0.1,0.001,0.0001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X,Y)

clf = SVC(C= 15, gamma = 100)
clf.fit(X,Y)
y_pred = clf.predict(X)

acc = accuracy_score(Y, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(Y,y_pred)

# Exploratort data analysis 

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(data=SalaryData_data,x="age",y="Salary")
plt.show()

sns.boxplot(data=SalaryData_data,x="workclass",y="Salary")
plt.show()

sns.boxplot(data=SalaryData_data,x="hoursperweek",y="Salary")
plt.show()

sns.pairplot(data=SalaryData_data,hue="Salary",height=2)
plt.show()

sns.boxplot(data=SalaryData_data,x="age",y="Salary")
plt.show()

sns.boxplot(data=SalaryData_data,x="workclass",y="Salary")
plt.show()

sns.boxplot(data=SalaryData_data,x="hoursperweek",y="Salary")
plt.show()


