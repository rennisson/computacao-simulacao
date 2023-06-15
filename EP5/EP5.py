import time
import numpy as np
import scipy.stats as stats


def main():
    # Definições importantes
    np.random.seed(13685534)
    x = np.random.random(2)
    print(f'x: {x}')
    theta_1 = stats.norm.pdf(x, loc=0, scale=1)
    g = stats.dirichlet.rvs(theta_1, size=1)
    g_1 = stats.dirichlet.rvs(x, size=1)
    print(f'g: {g}')
    print(f'g_1: {g_1}')
    alpha = g / g_1
    print(f'Alpha: {alpha}')


if __name__ == '__main__':
    main()
