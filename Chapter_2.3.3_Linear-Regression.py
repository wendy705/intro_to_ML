import mglearn.datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

# the slope parameters (w) also called weights or coefficients, are stored in the coef_ attribute, while
# the offset or intercept (b) is stored in the intercept_ atribute
print("lr.coef_:", lr.coef_)
print("lr.intercept_:", lr.intercept_)
