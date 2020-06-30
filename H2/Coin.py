# %%
from random import choice, randint

# %%


def simulation():
    v_1 = 0
    v_rand = 0
    v_min = None
    index = randint(0, 1000)

    for i in range(1000):
        flips = []

        for j in range(10):
            outcome = choice(['T', 'H'])
            flips.append(outcome)

        heads = len([x for x in flips if x == 'H']) / len(flips)

        if i == 0:
            v_1 = heads
        if i == index:
            v_rand = heads
        if v_min == None or heads < v_min:
            v_min = heads

    return v_1, v_rand, v_min


def main():
    v_1_average = 0
    v_rand_average = 0
    v_min_average = 0
    for i in range(10000):
        v_1, v_rand, v_min = simulation()

        v_1_average += v_1
        v_rand_average += v_rand
        v_min_average += v_min
        print(i)
    print(v_1_average / 10000, v_rand_average / 10000, v_min_average / 10000)


if __name__ == "__main__":
    main()

# %%
