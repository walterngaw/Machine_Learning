#Hierarchical_Clustering (Agglomerative)


#Types: Agglomerative (bottom-up) (most common) & Divisive (top-down)
#Step 1: Make each data point a single-point cluster -> That forms N clusters
#Step 2: Take the two closest data points and make them one cluster -> That forms N-1 clusters
#Step 3: Take the two closest clusters and make them one cluster -> That forms N-2 clusters. 
	#closest clusters is calculated based on euclidiean distance of:
		#1) two closest points of the two clusters
		#2) two furthest points of the two clusters
		#3) average distance of the two clusters
		#4) distance between centroids of the two clusters
#Step 4: Repeat Step 3 until there is only one cluster


#Dendrograms: Memory of how the clusters were formed
#Extend all horizontal line on the dendrogram
#Find the longest vertical line on the dendrogram that does not cross any horizontal line
#Set dissimilarity threshold on that highest vertical distance
#Draw a horizontal line on the dendrogram based on the threshold and the number of intersect will be the number of clusters


## Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering as ac


## Importing the dataset
os.chdir('/Users/Walter/Desktop/Programming/UDM - Machine Learning/Part 4 - Clustering/Section 25 - Hierarchical Clustering')
dataset = pd.read_csv('Mall_Customers.csv')
print "Raw dataset:\n", dataset.head()
x = dataset.iloc[:,[3,4]].values #Taking columns 4 & 5
print "Independent variables:\n", x 


## Using dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward')) #ward method minimizes variance within each cluster
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
# Largest distance where we can make vertically without crossing any horizontal line: optimal clusters = 5


## Fitting Hierarchical clustering to the mall dataset
hc = ac(n_clusters = 5, affinity = 'euclidean', linkage = 'ward') #affinity  = distance to make the linkage, ward method minimizes variance within each cluster. Use the same linkage as the one used to build the dendrogram.
y_hc = hc.fit_predict(x) #Fitting AgglomerativeClustering to data x to create vector y.
print "Clusters:\n", y_hc #y_hc only shows the clusters. Join this with matrix x to analyse the behaviour of each clusters


##Visualising the clusters (Only for 2d clustering i.e. 2 columns of interest)
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
