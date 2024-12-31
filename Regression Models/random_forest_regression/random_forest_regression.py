# %% [markdown]
# ### Random Forest Regression

# %%
# import the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X,y

# %%
# Train the Random Forest Regression Model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# %%
regressor.predict([[6.5]])

# %%
# Visualising the Random Forest Regression results X vs y, X vs y_pred and X_grid vs y_pred in a single plot
X_grid = np.arange(min(X[:,0]), max(X[:,0]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
fig.suptitle('Random Forest Regression')

# Plot X vs y
ax1.scatter(X, y, color='red')
ax1.plot(X, y, color='blue')
ax1.set_title('X vs y')
ax1.set_xlabel('Position Level')
ax1.set_ylabel('Salary')

# Plot X vs y_pred
ax2.scatter(X, y, color='red')
ax2.plot(X, regressor.predict(X), color='blue')
ax2.set_title('X vs y_pred')
ax2.set_xlabel('Position Level')
ax2.set_ylabel('Predicted Salary')

# Plot X_grid vs y_pred
ax3.scatter(X, y, color='red')
ax3.plot(X_grid, regressor.predict(X_grid), color='blue')
ax3.set_title('X_grid vs y_pred')
ax3.set_xlabel('Position Level')
ax3.set_ylabel('Salary')

plt.tight_layout()
plt.show()


