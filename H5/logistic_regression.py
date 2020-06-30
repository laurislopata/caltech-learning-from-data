# %%
import random
import numpy as np


def random_point():
    return random.uniform(-1, 1), random.uniform(-1, 1)

def random_line():
    x0, y0 = random_point()
    x1, y1 = random_point()

    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0

    return slope, intercept


def generate_dataset(n, slope, intercept, d=2):
    X = 2 * np.random.rand(n, d + 1) - 1
    X[:, 0] = 1
    y = np.sign(slope * X[:, 1] + intercept - X[:, 2])

    return X, y


def logistic_regression(X, y, learning_rate=0.01):
    N = X.shape[0]

    w = np.zeros(X.shape[1])
    w_previous = np.ones(X.shape[1])

    epoch_count = 0
    epochs = []
    change = True

    def error(X=X, y=y):
        errors = []
        for n in range(X.shape[0]):
            errors.append(np.log(1 + np.exp(-y[n] * w.dot(X[n, :]))))

        return np.mean(errors)

    while change:
        epoch_count += 1

        for n in np.random.permutation(N):
            n = np.random.randint(0, N)
            gradE = -y[n]*X[n, :] / (1 + np.exp(y[n] * w.dot(X[n, :])))
            w = w - learning_rate * gradE

        change = np.linalg.norm(w - w_previous) > 0.01
        # print(np.linalg.norm(w - w_previous), error())
        w_previous = w

        # print(epoch_count, error())

    def predict(X):
        return np.sign(X @ w)

    return epoch_count, error, predict


def main():
    errors_in = []
    errors_out = []
    epochs = []
    for i in range(100):
        slope, intercept = random_line()

        X, y = generate_dataset(100, slope, intercept)
        epoch_count, error, predict = logistic_regression(X, y)
        errors_in.append(error())
        epochs.append(epoch_count)

        # X_test, y_test = generate_dataset(2000, slope, intercept)
        # errors_out.append(error(X_test, y_test))

        print(
            i, f"out: {np.mean(errors_out)}, in: {np.mean(errors_in)}, epoch: {np.mean(epochs)}")

    # print(np.mean(errors))


if __name__ == "__main__":
    main()


# %%
