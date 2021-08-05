import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# we use the mglearn forge dataset
X, y = mglearn.datasets.make_forge()
# we split the data in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# we instantiate the classifier class
clf = KNeighborsClassifier(n_neighbors=3)
# we fit the model using the training set
clf.fit(X_train, y_train)
# we make prediction on the test data calling the predict method
print("Test set predictions:", clf.predict(X_test))
# we call the score method to evaluate how well our model generalizes
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# the following code produces the visualization of the decision boundaries for one, three, and nine neighbors
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
