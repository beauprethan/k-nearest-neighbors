# K-nearest neighbors classifier on UCI iris dataset

import pandas as pd
irisDF = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=0)
irisDF.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
irisDF = irisDF.sample(frac = 1, random_state = 123).reset_index(drop = True)
irisDF

print(irisDF.groupby(['class']).describe())

print(irisDF.describe())

sepal_plot = irisDF.plot.scatter(x='sepal_length', y='sepal_width')

petal_plot = irisDF.plot.scatter(x = 'petal_length', y='petal_width')

# Reshapes the values in each column to something more workable and fits
# a linear regression
from sklearn.linear_model import LinearRegression
x = irisDF['sepal_length'].values.reshape(-1, 1)
y = irisDF['sepal_width'].values.reshape(-1, 1)
linReg = LinearRegression().fit(x, y)
print("sepal length and sepal width R^2 score: " + str(linReg.score(x, y)))

x = irisDF['petal_length'].values.reshape(-1, 1)
y = irisDF['petal_width'].values.reshape(-1, 1)
linReg = LinearRegression().fit(x, y)
print("petal length and petal width R^2 score: " + str(linReg.score(x, y)))

import numpy as np
import operator

# Classifies the given row based on its k nearest neighbors in the dataframe 
def classifier(irisDF, row, k):
  predicted = 0
  neighbors = nearestNeighbors(irisDF, row, k)
  result = [iRow['class'] for iRow in neighbors]
  predicted = max(set(result), key = result.count)
  return predicted

# Returns the k nearest neighbors
def nearestNeighbors(irisDF, row, k):
  distances = list()
  # Duplicates the dataframe and removes classes
  trainingDF = irisDF.copy()
  trainingDF.drop(columns = 'class', inplace = True)
  for i in trainingDF.index:
    distance = euclideanDistance(row, trainingDF.loc[i])
    distances.append((irisDF.loc[i], distance))
  distances.sort(key = operator.itemgetter(1))
  neighbors = list()
  for i in range(k):
    neighbors.append(distances[i][0])
  return neighbors

# Calculates the euclidean distance between two points
def euclideanDistance(r1, r2):
  return np.sqrt(np.sum(np.square(r1 - r2)))

# Test classifications
k = 1
row = [3.3,1.2,4.3,2.0]
predicted = classifier(irisDF, row, k)
print('k = 1\ntest row: %s\npredicted: %s\n' % (row, predicted))

k = 3
row = [6.3,3.0,4.2,1.0]
predicted = classifier(irisDF, row, k)
print('k = 3\ntest row: %s\npredicted: %s\n' % (row, predicted))

k = 5
row = [3.3,9.0,1.0,1.7]
predicted = classifier(irisDF, row, k)
print('k = 5\ntest row: %s\npredicted: %s\n' % (row, predicted))

k = 7
row = [5.5,3.9,8.6,9.6]
predicted = classifier(irisDF, row, k)
print('k = 7\ntest row: %s\npredicted: %s\n' % (row, predicted))