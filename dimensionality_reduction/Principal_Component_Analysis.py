#Principal_Component_Analysis


## Types of Dimensionality Reduction Techniques:
#1) Feature Selection: Backward Elimination: Forward Selection, Bidirectional Elimination, Score Comparison.
#2) Feature Extraction: Principal Component Analysis, Linear Discriminant Analysis, Kernel PCA, Quadratic Discriminant Analysis.


## What is PCA?
#From the m independent variables of your dataset, PCA extracts p <= m new independent variables that explain the most variance of the dataset, regardless of the dependent variable.
#The fact that the dependent variable is not considered makes PCA an unsupervised model.


## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

## Importing the dataset
os.chdir('/Users/Walter/Desktop/Programming/UDM - Machine Learning/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)')
dataset = pd.read_csv('Wine.csv')
print "Raw dataset:\n", dataset.head() 
x = dataset.iloc[:, 0:13].values #Index of first column to 12th column
print "Independent variables:\n", x 
y = dataset.iloc[:, 13].values #Index of last variable = 13
print "Dependent variable:\n", y


## Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0) #Definition: train_test_split(*arrays,**options). 20% as test set. random_state = seed, remove it for production codes.
print "x_train:\n", x_train
print "x_test:\n", x_test
print "y_train:\n", y_train
print "y_test:\n", y_test


## Feature Scaling
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
print "x_train_after_scaling:\n", x_train
print "x_test_after_scaling:\n", x_test


# ## Applying PCA - Part 1: Finding number of components to use
# pca = PCA(n_components = None) #None: Explain variance vector will be created
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
# explained_variance = pca.explained_variance_ratio_
# print "explained_variance:\n", explained_variance #13 components extracted as there were 13 independent variables. But these components are not the original independent variables.
# #Decided that top 2 components explain > 50% of the variance.


## Applying PCA - Part 2: Using 2 components (comment out Part 1 & re-run )
pca = PCA(n_components = 2) #None: Explain variance vector will be created
x_train = pca.fit_transform(x_train)
print "x_train_with_only_top_2_principal_components:\n", x_train
x_test = pca.transform(x_test)
print "x_test_with_only_top_2_principal_components:\n", x_test
explained_variance = pca.explained_variance_ratio_
print "explained_variance:\n", explained_variance 


## Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


## Predicting the Test set results
y_pred = classifier.predict(x_test)
print "predicted customer segment using the two principal components:\n", y_pred


## Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print "Confusion Matrix:\n", cm #3 by 3 matrix because it is a  prediction of customer segment 1,2,3. Accuracy is high because we extracted two principal components that explained >50% of the variance, and used the two components to predict.


## Visualising the Training set results
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


## Visualising the Test set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
