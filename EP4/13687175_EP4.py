import numpy as np
from scipy import stats

SEED = 13687175
N = 10000  # number of points
K = 1000  # number of cuts
V = 10  # cut we are interested in


def main():
    np.random.seed(SEED)
    pvals = np.random.random(3)  # pvals to use in multinomial generator
    x = np.random.multinomial(10, pvals)
    y = np.random.multinomial(10, pvals)
    alpha = x + y
    theta = stats.dirichlet.rvs(alpha, size=N)  # generates triples of theta
    theta_density = stats.dirichlet.pdf(theta.T, alpha)  # gets dirichlet density from 'theta'
    theta_density.sort()

    cutoff_set = np.split(theta_density, K)  # generates 'K' cuts in 'theta_density'
    soma = 0
    for i in range(K):
        if max(cutoff_set[i] >= V):
            break
        soma += cutoff_set[i].size

    result = soma / N
    print(f'Result: {result}')


def posterior_density_function(theta, x, y):
    return theta**(x*y-1)  # matrix (n, 3), where each triple is theta^(x * y - 1)


if __name__ == '__main__':
    main()

