import math
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

N = 1000 # Sample size
A = 0.386617636
B = 0.47950399848

def main():
    np.random.seed(13687175)

    print('{0:^10} | {1:^25}'.format('Sample',
                                     f'Estimativa'))
    print('-'*35)
    for i in range(1, 20):
        crude = Crude(1000*i)
        print('{0:^10} | {1:^25}'.format(f'{crude.n}',
                                        f'{crude.estimativa()}'))

    print('\n \n')

    print('{0:^10} | {1:^25}'.format('Sample',
                                     f'Estimativa'))
    print('-' * 35)
    for i in range(1, 20):
        hom = HitOrMiss(1000 * i)
        print('{0:^10} | {1:^25}'.format(f'{hom.n}',
                                         f'{hom.estimativa()}'))

    print('\n \n')

    print('{0:^10} | {1:^25}'.format('Sample',
                                     f'Estimativa'))
    print('-' * 35)
    for i in range(1, 20):
        imp_sample = ImportanceSampling(1000 * i)
        print('{0:^10} | {1:^25}'.format(f'{imp_sample.n}',
                                         f'{imp_sample.estimativa()}'))


class Crude:
    def __init__(self, n):
        self.n = n
        points = self.uniform(n)
        self.soma = self.experimento(points)

    # Choose randomly points in [0,1) using Uniform distribution
    def uniform(self, n):
        return np.random.uniform(0, 1, n)

    def experimento(self, points):
        n = len(points)
        soma = 0
        for i in range(n):
            soma += math.exp(-A*points[i]) * math.cos(B*points[i])

        return soma / n

    def estimativa(self):
        return self.soma


# VERIFICAR SE ESTÁ CERTO
class HitOrMiss:
    def __init__(self, n):
        self.n = n
        self.xPoints = np.random.uniform(0, 1, n)
        self.yPoints = np.random.uniform(0, 1, n)

        self.soma = self.experimento()

    def indicadora(self, x, y):
        fComparison = []
        for i in range(len(self.xPoints)):
            fComparison = math.exp(-A*self.xPoints[i]) * math.cos(B*self.xPoints[i])

        function = self.yPoints <= fComparison
        return function

    def experimento(self):
        function = self.indicadora(self.xPoints, self.yPoints)
        soma = 0
        for i in range(len(function)):
            soma += function[i]

        return soma / self.n

    def estimativa(self):
        return self.soma


class ImportanceSampling:
    def __init__(self, n):
        self.n = n
        # g(x) = Beta Distribuition(alpha, beta)
        self.alpha = 0.35
        self.beta = 1

        points = self.dist_beta()
        self.soma = self.experimento(points)

    def dist_beta(self):
        return np.random.beta(self.alpha, self.beta, self.n)

    def experimento(self, points):
        n = len(points)
        soma = 0
        for i in range(n):
            soma += (np.exp(-A * points[i]) * np.cos(B * points[i])) / beta.pdf(points[i], self.alpha, self.beta)

        return soma / self.n

    def estimativa(self):
        return self.soma

    def grafico(self):
        x = np.linspace(0, 1, 10000)
        # g(x) = Beta Distribuition(alpha, beta)
        function_g = beta.pdf(x, self.alpha, self.beta)
        # f(x) = exp(-Ax)cos(Bx)
        y2 = np.exp(-A * x) * np.cos(B * x)
        plt.title("PDF of Beta", fontsize=14)
        plt.xlabel("X", fontsize=8)
        plt.ylabel("Probability Density", fontsize=8)
        plt.plot(x, function_g, linewidth=3, color='firebrick')
        plt.plot(x, y2, linewidth=3, color='green')
        plt.ylim([0, 2])
        plt.xlim([0, 1.01])
        plt.show()


class ControlVariate:
    def __init__(self, n):
        self.n = n
        # phi(x) = 0.4(x-1)^2+0.6 (polynomial function)
        self.integration_phi = 0.73333  # integral de phi no intervalo (0,1)
        points = self.uniform(n)
        self.soma = self.experimento(points)

    def uniform(self, n):
        return np.random.uniform(0, 1, n)

    def experimento(self, points):
        n = len(points)
        soma = 0
        for i in range(n):
            soma += (np.exp(-A * points[i]) * np.cos(B * points[i])) \
                    - (0.4*(points[i] - 1)**2+0.6) \
                    + self.integration_phi

        return soma / self.n


if __name__ == '__main__':
    main()

