import numpy as np
from random import choice, uniform


def regression(X, y):
    inverse = np.linalg.pinv(X)
    # weights = np.dot(inverse, y)
    weights = np.linalg.solve(X.T@X, X.T@y)

    def predict(point):
        return int(np.sign(np.dot(weights, np.array(point))))

    return predict, weights


def random_point():
    return uniform(-1, 1), uniform(-1, 1)


def target(point):
    _, x1, x2 = point
    return np.sign(x1 ** 2 + x2 ** 2 - 0.6)


def generate_data(n=1000):
    X = []
    y = []
    for i in range(n):
        point = random_point()
        x = [1, point[0], point[1]]
        classification = target(x)
        X.append(x)
        y.append(classification * (1 if uniform(0, 1) < 0.9 else -1))

    return np.array(X), np.array(y)

def transform(X):
    X_transformed = []
    for x in X:
        _, x1, x2 = x
        X_transformed.append([1, x1, x2, x1*x2, x1*x1, x2*x2])

    return np.array(X_transformed)

def main():
    mean_mean = 0
    new_points = []
    for i in range(1000):
        X, y = generate_data()
        X_transformed = transform(X)
        predict, weights = regression(X_transformed, y)

        mean = 0
        new_X, new_y = generate_data()
        new_X_transformed = transform(new_X)
        for x, x_transformed, y in zip(new_X, new_X_transformed, new_y):
            if predict(x_transformed) != y:
                mean += 1
        mean_mean += mean/len(X)
        print(i)

    print(mean_mean / 1000)


if __name__ == "__main__":
    main()
