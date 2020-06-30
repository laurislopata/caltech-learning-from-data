# %%
import numpy as np
import matplotlib as plt


def E(x):
    return (x[0]*np.exp(x[1]) - 2*x[1]*np.exp(-x[0]))**2


def gradient(x):
    u, v = x
    return np.array([
        2*(u*np.exp(v) - 2*v*np.exp(-u))*(np.exp(v) + 2*v*np.exp(-u)),
        2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v) - 2*np.exp(-u))
    ])


def gradient_decent(x, learning_rate):
    error = E(x)
    count = 0
    
    while count < 16:
        d = gradient(x)
        x[0] = x[0] - learning_rate * d[0]

        d = gradient(x)
        x[1] = x[1] - learning_rate * d[1]

        error = E(x)
        count += 1

    return x, count, error


def main():
    print(gradient_decent(np.array([1.0, 1.0]), 0.1))


if __name__ == "__main__":
    main()


# %%
