#Self Organizing Map


## Package Used: Minimalistic Self Organizing Map
# https://testpypi.python.org/pypi/MiniSom/1.0


##Training the SOM: Step by Step
#Step 1: We start with a dataset composed of n_features independent variables.
#Step 2: We create a grid composed of nodes, each one having a weight vector of n_features elements.
#Step 3: Randomly initialize the balues of the weight vectors to small numbers close to 0 (but not 0).
#Step 4: Select one random observation point from the dataset.
#Step 5: Compute the Euclidean distances from this point to the different neurons in the network.
#Step 6: Select the neuron that has the minimum distance to the point. This neuron is called the winning node.
#Step 7: Update the weights of the winning node to move it closer to the point.
#Step 8: Using a Gaussian neighbourhood function of mean the winning node, also update the weights of the winning node neighbours to move them closer to the point. The neighbourhood radius is the sigma in the Gaussian function.
#Step 9: Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning) or after a batch of observations (Batch Learning), until the network convergest to a point where the neighbourhood stops decereasing.


#Action Plan:
#1) Unsupervised Learning (SOM): Identify fraudulent users using SOM
#2) Supervised Learning (ANN): Create is_fraud dependent variable using fraudulent users in Part 1. Use ANN to learn and predict probabilities of fraud for each user.
#3) Output probability and customer ID 


#Question: can I cluster users based on SOM values?

## Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

## Importing the dataset
os.chdir('/Users/Walter/Desktop/Programming/UDM - Deep Learning/Volume 2 - Unsupervised Deep Learning/Part 4 - Self Organizing Maps (SOM)/Section 16 - Building a SOM')
dataset = pd.read_csv('Credit_Card_Applications.csv')
print "Raw dataset:\n", dataset.head() #Data is either categorical or continuous
X = dataset.iloc[:,:-1].values #Taking all rows and all columns, except last column. SOM is trained based on all columns except last column.
print "Independent variables:\n", X
y = dataset.iloc[:,-1].values # Taking all rows and last column.
print "Account Approval:\n", y

## Feature Scaling
sc = MinMaxScaler(feature_range = (0,1)) # Scaling:Easier for deep learning model to train if there are many dimensions
X = sc.fit_transform(X)
print "Normalized X:\n", X

## Training the SOM
som = MiniSom( x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) # Definition: MiniSom(X,y,input_len, sigma=1, learning_rate=0.5,decay_function=None,random_seed=None). 10 by 10 grid chosen as number of observation is small. Input len = number of features, including customer id in order to identify customers later. Sigma = radius of the neighbourhoods of the grid. The higher the learning rate, the faster the convergence. Decay_function can be used to improve the convergence.
# Use larger array for larger user base.
som.random_weights_init(X) #Initializing the weights randomly. Put in the data that needs to be trained.
som.train_random(X, num_iteration = 100) #Apply Step 4 to Step 9, for 100 iterations.

## Visualizing the results
bone()
pcolor(som.distance_map().T) #som.distance_map will return all the Mean Inter-Neuron Distances (MID) in one matrix
colorbar()
markers = ['o','s']
colors = ['r','g']
for i, x in enumerate(X):
	w = som.winner(x) #Winning node of the customer x
	plot(w[0] + 0.5,
		 w[1] + 0.5,
		 markers[y[i]],
		 markeredgecolor = colors[y[i]],
		 markerfacecolor = 'None',
		 markersize = 10,
		 markeredgewidth = 2) # To put the marker at the centre of the square. If customer did not get approval, y[i] will be 0, markers will be 'o' and 'r'.
show()
plt.show()
#The grid shows all the winning nodes. The higher the MID, the further away the neuron is from its neighbors. The higher the MID, the more likely it is an outlier.

## Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0) #Give list of customers associated to the following outlier winning node, axis = 0 to concatenate on the vertical axis.
frauds = sc.inverse_transform(frauds) #Inverse Transform Method
print "Whole list of cheaters:\n", frauds




# ## Supervised Deep Learning: Artificial Neural Network

# ## Creating the matrix of features
# customers = dataset.iloc[:,1:].values #all columns except first column
# print "Matrix of features:\n", customers #all information that customers need to fill in for their credit card

# ## Creating the dependent variable
# is_fraud = np.zeros(len(dataset)) #Initialize a vector of zeros (all assumed to be non-cheaters)
# for i in range(len(dataset)):
# 	if dataset.iloc[i,0] in frauds:  
# 		is_fraud[i] = 1 
# #if customer id in list of fraud
# #replace zeros by 1 for users that are fradulent based on SOM

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# customers = sc.fit_transform(customers)

# # Importing the Keras libraries and packages
# #pip install keras
# #pip install tensorflow
# from keras.models import Sequential
# from keras.layers import Dense

# # Initialising the ANN
# classifier = Sequential()

# # Adding the input layer and the first hidden layer
# classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15)) # units = number of neurons. input_dim = number of features

# # Adding the second hidden layer
# classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# # Adding the output layer
# classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# # Compiling the ANN
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# # Fitting the ANN to the Training set
# classifier.fit(customers, is_fraud, batch_size = 10, epochs = 100) 

# # Predicting the probabilities of frauds
# y_pred = classifier.predict(customers)
# y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred),axis = 1) 
# #iloc[:,0:1] takes column 0 to 1, excluding column 1. Will be in 2D array.
# #y_pred is a 2D array.
# #Horizontal concatenation: axis = 1
# #2D array: first column = customer_id, second column = p(fraud)
# y_pred = y_pred[y_pred[:,1].argsort()] #sort by column 1
# print "probabilities of fraud with customer ID:\n", y_pred
