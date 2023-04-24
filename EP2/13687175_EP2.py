import math
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

SEED = 13687175
N = 1000 # Sample size
A = 0.386617636
B = 0.47950399848

def main():
    np.random.seed(SEED)

    print('CRUDE')
    print('{0:^10} | {1:^25}'.format('Sample',
                                     f'Estimativa'))
    print('-'*35)
    # for i in range(1, 3):
    #     crude = Crude(13465688)
    #     print('{0:^10} | {1:^25}'.format(f'{crude.n}',
    #                                     f'{crude.estimativa()}'))

    print('\n \n')
    print('Hit or Miss')
    print('{0:^10} | {1:^25}'.format('Sample',
                                     f'Estimativa'))
    print('-' * 35)
    for i in range(1, 10):
        hom = HitOrMiss(10128002)
        print('{0:^10} | {1:^25}'.format(f'{hom.n}',
                                         f'{hom.estimativa()}'))

    print('\n \n')
    print('IMPORTANCE SAMPLING')
    print('{0:^10} | {1:^25}'.format('Sample',
                                     f'Estimativa'))
    print('-' * 35)
    for i in range(1, 3):
        imp_sample = ImportanceSampling(1000 * i)
        print('{0:^10} | {1:^25}'.format(f'{imp_sample.n}',
                                         f'{imp_sample.estimativa()}'))

    print('\n \n')
    print('CONTROL VARIATE')
    print('{0:^10} | {1:^25}'.format('Sample',
                                     f'Estimativa'))
    print('-' * 35)
    for i in range(1, 3):
        control = ControlVariate(3059462)
        print('{0:^10} | {1:^25}'.format(f'{control.n}',
                                         f'{control.estimativa()}'))


class Crude:
    def __init__(self, n):
        self.n = n
        # Choose randomly points in [0,1) using Uniform distribution
        points = np.random.uniform(0, 1, n)
        self.soma = self.experimento(points)

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

    def experimento(self):
        # Lista que vai armazenar os valores da função f(x)
        f_values = np.zeros(self.n)  # inicializada com zeros
        for i in range(len(self.n)):
            f_values[i] = math.exp(-A * self.xPoints[i]) * math.cos(B * self.xPoints[i])

        # retorna um array com TRUE se y < f_values. FALSE, caso contrario.
        h = (self.yPoints <= f_values)

        soma = h.sum()

        return soma / self.n

    def estimativa(self):
        return self.soma


class ImportanceSampling:
    def __init__(self, n):
        self.n = n
        # g(x) = Beta Distribuition(alpha, beta)
        self.alpha = 0.35
        self.beta = 1
        points = np.random.beta(self.alpha, self.beta, self.n)

        self.soma = self.experimento(points)

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
        # phi(x) = -0.39727x + 1
        self.integration_phi = 0.801365  # integral de phi no intervalo (0,1)
        points = np.random.uniform(0, 1, n)
        self.soma = self.experimento(points)

    def experimento(self, points):
        n = len(points)
        soma = 0
        for i in range(n):
            soma += (np.exp(-A * points[i]) * np.cos(B * points[i])) \
                    - (-0.39727*points[i] + 1) \
                    + self.integration_phi

        return soma / self.n

    def estimativa(self):
        return self.soma


if __name__ == '__main__':
    main()

