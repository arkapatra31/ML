# %% [markdown]
# ### Decision Tree Regression

# %%
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X,y

# %%
# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# %%
# Predicting a new result
regressor.predict([[6.5]])

# %%
# Visualizing the Decision Tree Regression results
plt.scatter(X, y, color = 'red')
#plt.plot(X, y, color = 'blue')
plt.plot(X, regressor.predict(X), color = 'green')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Decision Tree Regression")
plt.show()

# %%
# Visualizing the Decision Tree Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X[:,0]), max(X[:,0]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Decision Tree Regression")
plt.show()

# %%
# Visualizing all the plots together
# Sample data
y_pred = regressor.predict(X)
y_grid_pred = regressor.predict(X_grid)

# Create subplots
fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 5))
fig.suptitle('Decision Tree Regression', fontsize=16)

# First subplot
ax.plot(X, y, color='red')
ax.set_title('Actual Salary')
ax.set_xlabel('Postion Level')
ax.set_ylabel('Salary')
ax.grid(True)

# Second subplot
ax1.plot(X, y_pred, color='blue')
ax1.set_title('Predicted Salary')
ax1.set_xlabel('Postion Level')
ax1.set_ylabel('Salary')
ax1.grid(True)

# Third subplot
ax2.plot(X_grid, y_grid_pred, color='green')
ax2.set_title('Precise Predicted Salary')
ax2.set_xlabel('Postion Level')
ax2.set_ylabel('Salary')
ax2.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


