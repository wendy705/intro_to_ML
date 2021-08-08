import mglearn.datasets
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:2.f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:2.f}".format(ridge.score(X_test, y_test)))

# higher values of the alpha parameter restrict the coefficients in the model
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:2.f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:2.f}".format(ridge10.score(X_test, y_test)))

# lower values of the alpha parameter allow the coefficients to be less restricted -
# the less the coefficients are restricted, the more our model will become similar to linear regression
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:2.f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:2.f}".format(ridge01.score(X_test, y_test)))

# plot the different coefficients to compare
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-25, 25)
plt.legend()

