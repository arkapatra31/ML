# %% [markdown]
# ### Simple Linear Regression
# <p> Method used to plot the line is Ordinary Least Squares

# %%
# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# %%
# Split the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

print(f"{X_train=}")
print(f"{X_test=}")
print(f"{y_train=}")
print(f"{y_test=}")

# %% [markdown]
# ### Training the Simple Linear Regression model on training set

# %%
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %% [markdown]
# ### Predicting the test set results

# %%
y_pred = regressor.predict(X_test)
y_pred

# %% [markdown]
# ### Visualizing the Training Result set

# %%
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel("YOE")
plt.ylabel("Salary")
plt.show()

# %% [markdown]
# ### Visualizing the Test Result Set

# %%
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Test set)")
plt.xlabel("YOE")
plt.ylabel("Salary")
plt.show()

# %%
# Use the model to predict the salary of a person with 12 years of experience
print(f"Salary of a person with 12 years of experience: {regressor.predict([[12]])}")


