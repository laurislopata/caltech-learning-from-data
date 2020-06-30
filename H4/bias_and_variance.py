# %%
from math import pi,sin
from random import uniform

import matplotlib.pyplot as plt
import numpy as np


def random_point():
    x = uniform(-1, 1)
    y = sin(pi*x)
    return x, y


def main():
    pass

if __name__ == "__main__":
    main()
