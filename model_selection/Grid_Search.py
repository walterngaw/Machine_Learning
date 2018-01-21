# Grid_Search

## What is Grid_Search?
# To find out the best parameters for your model
# - SVM (Linear problem) or Kernel SVM (Non-Linear problem)?
# - Which gamma values to use?

## Which model to choose for my problem?
#Regression = With dependent variable + continuous outcome
#Classficiation = With dependent variable + categorical outcome
#Clustering = No dependent variable

# Note: Grid_Search is done after K-Fold_Cross_Validation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Importing the dataset
os.chdir('/Users/Walter/Desktop/Programming/UDM - Machine Learning/Part 10 - Model Selection & Boosting/Section 48 - Model Selection')
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

#Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1) #Definition: cross_val_score(estimator, X = data to fit, y = dependent variable, cv = number of folds, n_jobs = -1 to run on all CPUs)
print "Accuracies:\n", accuracies
average_accuracy = accuracies.mean()
print "Average Accuracy:\n", average_accuracy
accuracies_sd = accuracies.std()
print "Standard Deviation of Accuracy:\n", accuracies_sd

#Applying Grid Search to find the best model and the best parameters
parameters = [
 			  {'C':[1,10,100,1000], 'kernel':['linear']},
			  {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.5,0.1,0.01,0.001,0.0001]}
			 ] #Look at SVC's parameters (the model that you used)
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1 ) #Definition: grid_search(estimator, param_grid, scoring metrics, cross_validation = Use mean of 10 accuracies, njobs = -1 to run on all CPUs)
grid_search = grid_search.fit(X_train, y_train) #Grid objects will be fitted to trianing set
best_accuracy = grid_search.best_score_
print "Best average accuracy based on grid_search parameters:\n", best_accuracy
best_parameters = grid_search.best_params_
print "Best parameters based on grid_search parameters:\n", best_parameters
#Based on the best parameters, retest the grid_search parameters to improve your model accuracy.


# # Visualising the Training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Kernel SVM (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()

# # Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Kernel SVM (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
