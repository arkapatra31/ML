# %% [markdown]
# ### Support Vector Regression

# %%
# import the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X,y

# %%
# Feature Scaling always accepts 2D array so we need to convert the 1D array to 2D array
y = y.reshape(len(y),1)
y

# %%
# Feature Scaling
# Dataset not to be split as the model needs to understand the relationship between all the levels
"""We need to apply feature scaling to both the dependent and independent variables
because the dependent variable value is too high as compared to the independent variable"""

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

print(np.concatenate((X,y),1))

# %%
# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(X,y)

# %%
# Predicting the SVR model with a new value
"""
Reshaping for prediction:
If you want to predict on a new data point with only one feature, 
you would need to reshape it into a 2D array with a single row and one column, 
like this: new_data = np.array([[new_feature_value]]).reshape(-1, 1). 
Link : https://medium.com/@jwbtmf/reshaping-the-dataset-for-neural-network-15ee7bcea25e#:~:text=The%20shape%20of%20input%20data,be%20processed%20by%20the%20CNN.
"""
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

# %%
# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Postion Salary')
plt.ylabel('Salary')
plt.show()

# %%
# Visualising the SVR results (for higher resolution and smoother curve)
# X and y are already both scaled
X_grid = np.arange(min(sc_X.inverse_transform(X)[:,0]), max(sc_X.inverse_transform(X)[:,0]), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1)) # X_grid is unscaled

#Plot the graph
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


