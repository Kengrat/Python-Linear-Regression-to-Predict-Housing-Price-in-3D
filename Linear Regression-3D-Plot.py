import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting

# sklearn package for machine learning in python:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# read data from .csv in folder
df = pd.read_csv("houseprice_data.csv") 

X = df.iloc[:, [9, 10]].values # input variable grade and sqft_above
y = df.iloc[:, 0].values # target variable price

# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)

# The coefficients
print('Coefficients: ', regr.coef_)
# The coefficients
print('Intercept: ', regr.intercept_)
# The mean squared error
print('Mean squared error: %.4f' % mean_squared_error(y_test, regr.predict(X_test)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(y_test, regr.predict(X_test)))

# visualise initial data set in 3D
fig1 = plt.figure(figsize=(8,6))

ax1 = fig1.add_subplot(111, projection = '3d')
ax1.scatter(X[:,0], X[:,1], y, color = 'blue')

ax1.azim = 120
ax1.dist = 11
ax1.elev = 10

ax1.set_xlim(0,80)
ax1.set_ylim(0,30)
ax1.set_zlim(0,40)

ax1.set_title('LR Initial Data set. Price Vs Grade Vs Sqft_above')
ax1.set_xlabel('grade')
ax1.set_ylabel('price ($)')
ax1.set_zlabel('sqft_above')
fig1.tight_layout(pad=-2.0)

# visualise training set results in 3D:
fig2 = plt.figure(figsize=(8,6))
ax2 = fig2.add_subplot(111, projection = '3d')

# plot the training data
ax2.scatter(X_train[:,0], X_train[:,1], y_train, color = 'blue')

# plot the plane which represents the model
X1, X2 = np.meshgrid(range(40), range(9410))
Z = regr.coef_[0]*X1+regr.coef_[1]*X2+regr.intercept_
ax2.plot_surface(X1, X2, Z, alpha=0.5, color='red')

ax2.azim = 120
ax2.dist = 11
ax2.elev = 10

ax2.set_title('LR Train set. Price Vs Grade Vs Sqft_above')
ax2.set_xlabel('Grade')
ax2.set_ylabel('price ($)')
ax2.set_zlabel('sqft_above')

fig2.tight_layout(pad=-2.0)

# visualise test set results in 3D with plane:
fig3 = plt.figure(figsize=(8,6))
ax3 = fig3.add_subplot(111, projection = '3d')

# plot the data
ax3.scatter(X_test[:,0], X_test[:,1], y_test, color = 'blue')

# plot the plane which represents the model
X1, X2 = np.meshgrid(range(40), range(9410))
Z = regr.coef_[0]*X1+regr.coef_[1]*X2+regr.intercept_
ax3.plot_surface(X1, X2, Z, alpha=0.5, color='red')

ax3.azim = 120
ax3.dist = 11
ax3.elev = 10

ax3.set_title('LR Test set. Price Vs Grade Vs Sqft_above ')
ax3.set_xlabel('grade')
ax3.set_ylabel('price ($)')
ax3.set_zlabel('sqft_above')

fig3.tight_layout(pad=-2.0)