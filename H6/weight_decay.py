# %%
import numpy as np


# %%
def read_data(file):
    data = np.loadtxt(file)

    X, y = np.hsplit(data, [2])
    return X, y.flatten()


X_train, y_train = read_data("in.dta")
X_test, y_test = read_data("out.dta")

# %%
def transform(X):
    def f(x):
        x1, x2 = x
        return [1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)]

    return np.apply_along_axis(f, 1, X)


X_train = transform(X_train)
X_test = transform(X_test)

# %%
def train(X, y, k):
    λ = 10 ** int(k)
    weights = np.linalg.solve(X.T @ X + λ * np.identity(X.shape[1]), X.T @ y)

    def predict(X):
        return np.sign(X @ weights)

    return predict


def calc_errors(k):
    predict = train(X_train, y_train, k)

    train_error = np.mean(predict(X_train) != y_train)
    test_error = np.mean(predict(X_test) != y_test)

    print(f"{k:5.1f}, in: {train_error:.3f}, out: {test_error:.3f}")

    return test_error


vcalc = np.vectorize(calc_errors)
errors = vcalc(np.arange(-10, 10, 0.1).reshape((-1, 1)))

print()
print("min(out):", np.min(errors))
