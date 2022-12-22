#Linear regression over the data
import matplotlib
matplotlib.use('GTK3Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
from numpy.linalg import eig

data = pd.read_csv('Housing.csv')
X = data["price"]
Y = data["lotsize"]

# X = X.values.reshape(len(X), 1)
# Y = Y.values.reshape(len(Y), 1)

#split data into training/testing dataset
X_train = X[:-250]
X_test = X[-250:]

# Split the targets into training/testing sets
Y_train = Y[:-250]
Y_test = Y[-250:]

# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

plt.show()

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Plot outputs
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)

#To make an individual prediction using the linear regression model:
print(str(round(regr.predict(5000))))


#calculation of eigenvalues and eigenvectors
a = np.array([[2, 2, 4],
              [1, 3, 5],
              [2, 3, 4]])
w,v=eig(a)
print('E-value:', w)
print('E-vector', v)