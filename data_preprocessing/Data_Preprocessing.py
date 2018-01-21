#Data Preprocessing


## Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


## Importing the dataset
os.chdir('/Users/Walter/Desktop/Programming/UDM - Machine Learning/Part 1 - Data Preprocessing')
dataset = pd.read_csv('Data.csv')
print "Raw dataset:\n", dataset.head() 
x = dataset.iloc[:,:-1].values #Taking all rows and all columns, except last column.
print "Independent variables:\n", x 
y = dataset.iloc[:,3].values # Taking all rows and last column.
print "Dependent variable:\n", y


## Taking care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0) #Indicate the settings for Inputer function. strategy = mean / median / most_frequent.
imputer = imputer.fit(x[:,1:3]) #Fit imputer into matrix x, on columns where there is missing data.Taking index 1 and 2. 
x[:,1:3] = imputer.transform(x[:,1:3]) # Replace missing data by mean of the columns
print "Independent variables after replacing missing data:\n", x 


#Encoding categorical data (Independent Variables)
le = LabelEncoder() 
x[:,0] = le.fit_transform(x[:,0]) #Fit label encoder into first column
print "Independent variables post encoding categorical column 1:\n", x 
#Dummy Encoding (Independent Variables)
#Done so that there is no numerical relationship between encoded categorical data. E.g. France = 2, Spain = 1. Hence, France = Spain + 1.
ohe = OneHotEncoder(categorical_features = [0]) #Specify the index of the columns that you want to encode. Categorical_features = all / array of indices 
x = ohe.fit_transform(x).toarray()
print "Independent variables post onehotencoding categorical column 1:\n", x 
#Encoding categorical data (Dependent Variables)
le = LabelEncoder() 
y = le.fit_transform(y) #Fit label encoder into first column
print "Dependent variable post encoding:\n", y


#Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0) #Definition: train_test_split(*arrays,**options). 20% as test set. random_state = seed, remove it for production codes.
print "x_train:\n", x_train
print "x_test:\n", x_test
print "y_train:\n", y_train
print "y_test:\n", y_test


#Feature Scaling
#Transform variable values so that no one variables will dominate others due to its magnitude.
#Not needed if machine learning libraries already include scaling.
#No scaling needed for dummy variables as you want to keep the intepretation of the dummy variables (i.e. which country each row belongs to).
#No scaling needed for dependent variable if y is categorical (1/0), for classification. 
#Scaling needed for dependent variable if y is numerical, for regression.
#Type 1: Standardisation: x = (x-mean(x))/(sd(x))
#Type 2: Normalisation: x = (x-min(x))/(max(x)-min(x))
ss = StandardScaler()
x_train = ss.fit_transform(x_train) #For training set: fit + transform
x_test = ss.transform(x_test) #For test set: transform. As it ss is already fitted on the test set.
print "x_train:\n", x_train
print "x_test:\n", x_test









