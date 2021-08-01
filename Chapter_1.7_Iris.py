import pandas as pd
import numpy as np
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

# the object that is returned is a Bunch object, which is very similar to a dictionary
print("Keys of iris_dataset:\n", iris_dataset.keys())
# the DESCR key is a short description of the dataset
print(iris_dataset['DESCR'][:193] + "\n...")
# the value of the key target_names is an array of strings, containing the species of flower that we want to predict
print("Target names:", iris_dataset['target_names'])
# the value of feature names is a list of strings, giving the description of each feature
print("Feature names:\n", iris_dataset['feature_names'])
# the data itself is contained in the target and data fiels. data contains the numeric measurements of sepal length,
# sepal width, petal lenght, and petal width in a Numpy array
print("Type of data:", type(iris_dataset['data']))
# the rows in the array data correspond to flowers, while the columns represent the four measurements that were taken
# for each flower
print("shape of data:", iris_dataset['data'].shape)
# here are the feature values of the first five samples
print("First five rows of data:\n", iris_dataset['data'][:5])
# the target array contains the species of each of the flowers that were measured also as a NumPy array
print("Type of target:", type(iris_dataset['target']))
# target is a one-dimensional array, with one entry per flower
print("Shape of target:", iris_dataset['target'].shape)
# the sèecies are encoded as integers from 0 to 2
print("Target:\n", iris_dataset['target'])

# before making the split, the train_test_split function shuffles the dataset using a pseudorandom number generator
X_train, X_test, y_train, y_test = train_test_split( iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train_shape:", X_train.shape)
print("y_train_shape:", y_train.shape)

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)
# use the scikit-learn k-neighbors classification algorithm implemented in the KNeighborsClassifier classifier class
# instantiate the class into an object (knn)
knn = KNeighborsClassifier(n_neighbors=1)

# to build the model on the training set we call the fit method of the knn object, which takes as arguments the NumPy
# array x_train containing the training data and y_train containing the corresponding training labels
knn.fit(X_train, y_train)

# we can now make predictions using the model on new data
X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:", iris_dataset['target_names'][prediction])

# use the test set to evaluate the model's accuracy
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
# we can aso use the score method of the knn object, which will compute the test set accuracy for us
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
