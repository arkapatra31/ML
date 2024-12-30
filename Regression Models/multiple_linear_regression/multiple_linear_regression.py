# %% [markdown]
# ### Multiple Linear Regression

# %% [markdown]
# <p>Assumptions of Simple Linear Regression</p>
# <li>Non linear relationship between X and Y</li>
# <li>Non Homoscedasticity or unequal variance</li>
# <li>Uneven Multivariate Normality - Normality of Error Distribution</li>
# <li>Independence - No correlation</li>
# <li>Lack of Multicolinearity - Predictors are not co-related with one another</li>
# <li>The Outliner check - This is not an assumption but extra</li>
# 
# <p>5 methods of building a model</p>
# <li>All-in</li>
# <li>Backward Elimination</li>
# <li>Forward Selection</li>
# <li>Bidirectional Selection</li>
# <li>Score Comparision</li>

# %%
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Import the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X,y

# %%
# Encoding the categorical data
# Encode the independent variable
# Also just return the values of encoded categorical data along with the actual data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# %%
# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
# Training the multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %%
# Predicting the test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# %%
plt.scatter(y_test, y_pred, color='red')
plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], color='blue', linewidth=2, linestyle='dotted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# %%
# Predict the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = California
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# %%
# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)

# %%
# Getting the final linear regression equation with the values of the intercept
print(regressor.intercept_)

# %%
# R-Squared value
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# %%
# Mean Squared Error
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

# %%
# Mean Absolute Error
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)


