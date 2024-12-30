# %% [markdown]
# ### Polynomial Regression

# %%
# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values
X, y

# %%
# Training the Linear Regression model on the whole dataset
# Point to remember: We need to train our model on the whole dataset as Position needs to be taken into account
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# %%
# Training the Polynomial Regression model on the whole dataset
# Polynomial Linear regression model is created by adding polynomial terms to the linear regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# %%
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# %%
# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# %%
# Visualising the polynomial regression results with smoother curves
# Ensure X is a 1D array before using np.arange
# X_grid = np.arange(min(X), max(X), 0.1)
X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# %%
# Predicting a new result with Linear Regression Model for a level of 6.5
print(lin_reg.predict([[6.5]]))

# %%
# Predicting a new result with Polynomial Regression Model for a level of 6.5
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
# round off the value to 2 decimal places
np.set_printoptions(precision=2)
# Predicting a new result with Polynomial Regression Model for a level of 6.5
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))



