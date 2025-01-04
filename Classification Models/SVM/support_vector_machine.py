# %% [markdown]
# ### Support Vector Machine Classifier

# %% [markdown]
# ![SVM.gif](attachment:SVM.gif)

# %% [markdown]
# #### Importing the libraries

# %%
# import the libraries
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
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print(f"{X_train=}")
print(f"{X_test=}")
print(f"{y_train=}")
print(f"{y_test=}")

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
# #### Training the SVM model on the training data set

# %%
# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# %% [markdown]
# #### Predicting a new data

# %%
# Predicting a new result
classifier.predict(sc.transform([[30,87000]]))

# %% [markdown]
# #### Predicting the test data set result

# %%
y_pred = classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)),1))

# %% [markdown]
# #### Making the confusion matrix and accuracy prediction

# %%
# Making the Confusion Matrix and the Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# %% [markdown]
# #### Visualising the training set results

# %%
# Visulaising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step= 0.25),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step= 0.25)
)

# plot the filled contour map
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'aqua')))

# Prepare the scatter plots
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap('blue', 'green')(i), label = j)
    
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# %% [markdown]
# #### Visualising the test set results

# %%
# Visualising the test set results
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step= 0.25),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step= 0.25)
)

# Plot the filled contour
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'aqua')))


plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot the scattered points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'green'))(i), label = j)

plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


