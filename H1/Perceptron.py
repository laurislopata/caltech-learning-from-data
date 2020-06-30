# %%
import numpy as np
from random import choice, uniform
from matplotlib import pyplot as plt

# %%

def perceptron(learning_data):
    weights = np.array([0, 0, 0])

    def predict(point):
        x, y = point
        return int(np.sign(weights[0] + weights[1]*x + weights[2]*y))

    iter_count = 0

    while True:
        misclassified = []
        for point in learning_data:
            (x, y), classification = point
            prediction = predict((x, y))
            # print(weights[0] + weights[1]*x + weights[2]*y, prediction, classification)
            if prediction != classification:
                misclassified.append((np.array([1, x, y]), classification))
        if misclassified:
            x, classification = choice(misclassified)
            weights = weights + x*classification
            iter_count += 1
        else:
            break

    return iter_count, predict


def random_point():
    return uniform(-1, 1), uniform(-1, 1)


def generate_data(n=100):
    x0, y0 = random_point()
    x1, y1 = random_point()

    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0

    def classify(p):
        x, y = p
        return 1 if slope * x + intercept > y else -1

    data = []
    for i in range(n):
        p = random_point()
        data.append((p, classify(p)))
    return data, classify


def main():
    # plt.scatter([x for ((x, _), _) in data], [
    #             y for ((_, y), _) in data], color=["red" if c else "blue" for _, c in data])
    # plt.show()

    iter_count = 0
    incorrect_count = 0
    for i in range(1000):
        data, classify = generate_data()
        iterations, predict = perceptron(data)
        iter_count += iterations

        point = random_point()
        if classify(point) != predict(point):
            incorrect_count += 1

        print(i, iter_count/(i + 1), incorrect_count/(i + 1))

    print(iter_count/1000)
    print(incorrect_count/1000)


if __name__ == "__main__":
    main()
