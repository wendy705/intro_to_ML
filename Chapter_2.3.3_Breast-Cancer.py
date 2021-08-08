from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)

print("Training set score: {:2.f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:2.f}".format(logreg.score(X_test, y_test)))

# increasing C allows us to have a more flexible model, since higher values of C mean lower coefficients' regularization
# an higher C means a more complex model
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score: {:2.f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:2.f}".format(logreg100.score(X_test, y_test)))

# decreasing C increases model regularization, which means we will have a less complex model
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score: {:2.f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:2.f}".format(logreg001.score(X_test, y_test)))