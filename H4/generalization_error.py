# %%
import numpy as np
import matplotlib.pyplot as plt

dVC = 10
conf = 0.05

def mH(N):
    return N ** dVC


def vc_inequality(N):
    return np.sqrt((8/N) * np.log((4*mH(2*N)/0.05)))


def rademacher_penalty_bound(N):
    print(N)
    return np.sqrt((2*np.log(2*N*mH(N)))/N) + np.sqrt(2/N*np.log(1/conf)) + 1/N


# %%
def main():
    N = np.arange(1, 10_000)
    plt.plot(N, vc_inequality(N))
    plt.plot(N, rademacher_penalty_bound(N))
    plt.legend([
    "Original VC bound",
    "Rademacher Penalty Bound",
    ])

    plt.show()



if __name__ == "__main__":
    main()
