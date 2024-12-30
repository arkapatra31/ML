# %% [markdown]
# ### Data Preprocessing Tools
# <h4>Contents</h6>
# <li>Importing the libraries</li>
# <li>Importing the dataset</li>
# <li>Taking care of missing data</li>
# <li>Encoding categorical data</li>
# <li>Encoding the Independent Variable</li>
# <li>Encoding the Dependent Variable</li>
# <li>Splitting the dataset into the Training set and Test set</li>
# <li>Feature Scaling</li>

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
## Fetch the Dataset
dataset = pd.read_csv("Data.csv")

# %% [markdown]
# ### Fetch the features and the labels
# <p>Features are basically inputs or the independent variables</p>
# <p>Label is the output or basically the dependent variable</p>

# %%
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X,y

# %%
# Manage Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# %%
# Apply the imputer to the dataset
imputer.fit(X[:, 1:3])

# Replace the missing data with the mean of the column
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

# %% [markdown]
# ### Encode Categorical Data
# <p>One Hot Encoding converts the column variables into a binary format</p>

# %% [markdown]
# #### Encoding the Independent variable (Features)

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(f"{X=}\n")

# %% [markdown]
# ## Encode Dependent Variable (Labels / Output)

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y

# %% [markdown]
# ### Splitting the dataset into Training set and Test set

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"{X_train=}\n")
print(f"{X_test=}\n")
print(f"{y_train=}\n")
print(f"{y_test=}\n")

# %% [markdown]
# ## Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# fit_transform on training set to compute mean and standard deviation of the features and applied to training set
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
# transform on test set to apply the same stats to test set without new computation and data leakage
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# %%
print(f"{X_train=}\n")
print(f"{X_test=}\n")


