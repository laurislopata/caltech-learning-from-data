from matplotlib import pyplot as plt
import numpy as np
from random import choice, uniform

def regression(X, y):
    inverse = np.linalg.pinv(X)
    weights = np.dot(inverse, y)
    # weights = np.linalg.solve(X.T@X, X.T@y)
    print(weights)

    def predict(point):
        return int(np.sign(np.dot(weights, np.array(point))))

    return weights


def random_point():
    return uniform(-1, 1), uniform(-1, 1)


def generate_data(n=10):
    x0, y0 = random_point()
    x1, y1 = random_point()

    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0

    def classify(p):
        _, x, y = p
        return 1 if slope * x + intercept > y else -1

    X = []
    y = []
    for i in range(n):
        p = random_point()
        x = [1, p[0], p[1]]
        X.append(x)
        y.append(classify(x))
    return np.array(X), np.array(y), classify


def perceptron(X, y, weights):
    def predict(point):
        return int(np.sign(np.dot(weights, point)))

    iter_count = 0

    while True:
        misclassified = []
        for point, classification in zip(X, y):
            prediction = predict(point)
            # print(weights[0] + weights[1]*x + weights[2]*y, prediction, classification)
            if prediction != classification:
                misclassified.append((point, classification))
        if misclassified:
            x, classification = choice(misclassified)
            weights = weights + x*classification
            iter_count += 1
        else:
            break

    return iter_count, predict

def main():
    mean = 0
    for i in range(1000):
        X, y, classify = generate_data()
        weights = regression(X, y)
        iter_count, predict = perceptron(X, y, weights)
        # for j in range(1000):
        #     p = random_point()
        #     x = [1, p[0], p[1]]
        #     if classify(x) != predict(x):
        #         incorrect_count += 1            
        # mean += incorrect_count / 1000
        print(i, mean/(i + 1))
        mean+=iter_count

    print(mean/1000)


if __name__ == "__main__":
    main()
