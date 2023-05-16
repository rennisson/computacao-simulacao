import time
import numpy as np
import numpy.random
from scipy.stats import qmc, beta, uniform

SEED = 13687175
A = 0.386617636
B = 0.47950399848


def main():
    numpy.random.seed(13687175)
    x = np.array([1, 1, 2])
    y = np.array([3, 2, 1])
    n = 10
    points = numpy.random.gamma(0.5, size=(n, 3))  # generate n vectors of theta
    posterior_function = posterior_density_function(points, x, y)


def posterior_density_function(points, x, y):
    f = points**(x*y-1)  # matrix (n, 3), where each triple is theta^(x * y - 1)
    f_values = np.ndarray(shape=(10, 1))

    for i in range(10):
        f_values[i] = f[i][0] * f[i][1] * f[i][2]  # product among the 3 therms from each triple

    print(f'f_values: {f_values}')
    return f_values


if __name__ == '__main__':
    main()

