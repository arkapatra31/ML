# %% [markdown]
# ### K-Nearest Neighbour Classification

# %% [markdown]
# #### Importing the libraries

# %%
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# #### Importing the dataset

# %%
# Import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X,y

# %% [markdown]
# #### Splitting the data into train and test set

# %%
# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train, X_test, y_train, y_test

# %% [markdown]
# #### Feature Scaling

# %%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train, X_test

# %% [markdown]
# #### Training the KKN Classification Model with test data set

# %%
# Train the KNN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2) # p = 2 is for euclidean distance
classifier.fit(X_train, y_train)

# %% [markdown]
# #### Predicting a new result

# %%
classifier.predict(sc.transform([[30,87000]]))

# %% [markdown]
# #### Predicting the Test set results

# %%
y_pred = classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)), 1))

# %% [markdown]
# #### Making the Confusion Matrix

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# %% [markdown]
# #### Visualising the Training set results

# %%
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min()-10, stop=X_set[:, 0].max()+10, step=1),
    np.arange(start=X[:, 1].min()-1000, stop=X_set[:, 1].max()+1000, step=1)
)

plt.contourf(X1, X2, 
             classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75,
             cmap = ListedColormap(('red', 'aqua')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'green'))(i), label=j)

plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# %% [markdown]
# #### Visualising the Test set results

# %%
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min()-10, stop=X_set[:, 0].max()+10, step=1),
    np.arange(start=X[:, 1].min()-1000, stop=X_set[:, 1].max()+1000, step=1)
)

plt.contourf(X1, X2, 
             classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75,
             cmap = ListedColormap(('red', 'aqua')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'green'))(i), label=j)

plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


