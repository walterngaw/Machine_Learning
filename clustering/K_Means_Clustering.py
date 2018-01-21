#K_Means_Clustering


#Step 1: Choose the number K of clusters
#Step 2: Select at random K points, the centroids (not necessarily from your dataset)
#Step 3: Assign each data point to the closest centroid -> That forms K clusters.
#Step 4: Compute and place the centroids at the new centroid of each cluster based on the central of mass (new centroid is based on the average x and y values for each clusters)
#Step 5: Reassign each data point to the new closest centroid. If any reassignment took place, go to Step 4, otherwise your model is ready.


#Random Initialization Trap
#A bad random initialisation can potentially dictate the outcome of the model
#Use K-Means++ Algorithm to avoid bad random initialisation


#Choosing the number of clusters
#WCSS (Within-cluster sums of squares) will keep decreasing untill 0 as the number of clusters increase towards the number of points
#WCSS = Sum(distiance(Point, Centroid)^2) for cluster 1 + ... + Sum(distiance(Point, Centroid)^2) for cluster n.
#Elbow Method: Optimal clusters = the point where WCSS drops drastically 


## Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.cluster import KMeans


## Importing the dataset
os.chdir('/Users/Walter/Desktop/Programming/UDM - Machine Learning/Part 4 - Clustering/Section 24 - K-Means Clustering')
dataset = pd.read_csv('Mall_Customers.csv')
print "Raw dataset:\n", dataset.head() 
x = dataset.iloc[:,[3,4]].values
print "Independent variables:\n", x 


##Using Elbow Method to find the optimal number of clusters
wcss = []
for i in range(1,11):
	kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) #Fit K-Means into data x. Use k-means++ to avoid falling into random initializatoin trap. max_iter = maximum iterations to find the clusters. n_init = number of times the algo will run with different initial centroids. Remove random_state for production.
	kmeans.fit(x) #Fit kmeans to matrix x
	wcss.append(kmeans.inertia_) #Inertia is another name for WCSS
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#k = 5 is optimal


##Applying k-means to the mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x) #Fitting k-means onto matrix x, where y_kmeans is the cluster number.


##Visualising the clusters (Only for 2d clustering i.e. 2 columns of interest)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids') #Centroids
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
#Definition: plt.scatter(x_coordinate, y_coordinate, size, cluster color, cluster label )
