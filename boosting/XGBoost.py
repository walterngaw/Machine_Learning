#XGBoost

## Install XGBoost following the instructions on: http://xgboost.readthedocs.io/en/latest/build.html#
# Step 1: Follow steps under Building on OSX: http://xgboost.readthedocs.io/en/latest/build.html#building-on-osx
# Step 2: Follow steps under Python Package Installation: http://xgboost.readthedocs.io/en/latest/build.html#python-package-installation

## What is XGBoost?
# A gradient boosting method with trees.

# Pros
#1) High performance
#2) Fast execution speed
#3) Keep interpretation of dataset - Feature scaling not needed as it is a gradient boosting method 


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# Importing the dataset
os.chdir('/Users/Walter/Desktop/Programming/UDM - Machine Learning/Part 10 - Model Selection & Boosting/Section 49 - XGBoost')
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier() # Definition: XGBClassifier(learning_rate, n_estimators = number of trees, objective = "binary:logistic", gamma = 0, ...)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print "Predictions:\n", y_pred

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print "Confusion Matrix:\n", cm

#Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10) #Definition: cross_val_score(estimator, X = data to fit, y = dependent variable, cv = number of folds, n_jobs = -1 to run on all CPUs)
print "Accuracies:\n", accuracies
average_accuracy = accuracies.mean()
print "Average Accuracy:\n", average_accuracy
accuracies_sd = accuracies.std()
print "Standard Deviation of Accuracy:\n", accuracies_sd
