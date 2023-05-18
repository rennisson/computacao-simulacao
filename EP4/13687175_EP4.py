import time
import numpy as np
import numpy.random
from scipy import stats

SEED = 13687175
A = 0.386617636
B = 0.47950399848


def main():
    numpy.random.seed(13687175)
    pvals = np.random.random(3)
    print(f'pvals: {pvals}')
    x = np.random.multinomial(10, pvals, 1)
    y = np.random.multinomial(10, pvals, 1)
    print(f'x: {x}, y: {y}')
    theta = stats.dirichlet.rvs(x[0], size=1)
    print(f'theta: {theta}')
    posterior_function = posterior_density_function(theta, x, y)
    print(f'posterior function: {posterior_function}')
    sup_f = max(posterior_function[0])
    print(sup_f)


def posterior_density_function(theta, x, y):
    return theta**(x*y-1)  # matrix (n, 3), where each triple is theta^(x * y - 1)


if __name__ == '__main__':
    main()

