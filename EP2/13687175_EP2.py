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

    # print('CRUDE')
    # print('{0:^10} | {1:^25}'.format('Sample',
    #                                  f'Estimativa'))
    # print('-'*35)
    # for i in range(1, 10):
    #     n = 13465688
    #     crude = Crude(n)
    #     print('{0:^10} | {1:^25}'.format(f'{n}',
    #                                     f'{crude.estimativa()}'))
    #
    # print('\n \n')
    # print('Hit or Miss')
    # print('{0:^10} | {1:^25}'.format('Sample',
    #                                  f'Estimativa'))
    # print('-' * 35)
    # for i in range(1, 10):
    #     n = 10128002
    #     hom = HitOrMiss(n)
    #     print('{0:^10} | {1:^25}'.format(f'{n}',
    #                                      f'{hom.estimativa()}'))
    #
    # print('\n \n')
    # print('IMPORTANCE SAMPLING')
    # print('{0:^10} | {1:^25}'.format('Sample',
    #                                  f'Estimativa'))
    # print('-' * 35)
    # n = 65281716
    # for i in range(1, 10):
    #     imp_sample = ImportanceSampling(n)
    #     print('{0:^10} | {1:^25}'.format(f'{n}',
    #                                      f'{imp_sample.estimativa()}'))

    print('\n \n')
    print('CONTROL VARIATE')
    print('{0:^10} | {1:^25}'.format('Sample',
                                     f'Estimativa'))
    print('-' * 35)
    for i in range(1, 10):
        n = 3059462
        control = ControlVariate(n)
        print('{0:^10} | {1:^25}'.format(f'{n}',
                                         f'{control.estimativa()}'))


class Crude:
    def __init__(self, n):
        # Choose randomly points in [0,1) using Uniform distribution
        points = np.random.uniform(0, 1, n)
        f_values = np.exp(-A * points) * np.cos(B * points)

        self.gamma_hat = f_values.sum() / n

    def estimativa(self):
        return self.gamma_hat


# VERIFICAR SE ESTÁ CERTO
class HitOrMiss:
    def __init__(self, n):
        self.xPoints = np.random.uniform(0, 1, n)
        self.yPoints = np.random.uniform(0, 1, n)
        # Lista que vai armazenar os valores da função f(x)
        f_values = np.exp(-A * self.xPoints) * np.cos(B * self.xPoints)

        # retorna um array com TRUE se y < f_values. FALSE, caso contrario.
        h = (self.yPoints <= f_values)

        # h.sum() retorna a soma de todos os valores do array
        self.gamma_hat = h.sum() / n

    def estimativa(self):
        return self.gamma_hat


class ImportanceSampling:
    def __init__(self, n):
        # g(x) = Beta Distribuition(alpha, beta)
        self.alpha = 0.35
        self.beta = 1
        points = np.random.beta(self.alpha, self.beta, n)  # geradora de pontos da função beta
        f = np.exp(-A * points) * np.cos(B * points)  # guarda todos os valores da funcao f(x)
        beta_values = beta.pdf(points, self.alpha, self.beta)  # guarda todos os valores da funcao beta(alpha, beta)

        # (f / beta_values).sum() retorna um array com a soma de todas as divisoes feitas entre 'f' e 'beta_values'
        self.gamma_hat = (f / beta_values).sum() / n  # estimativa da importance sample

    def estimativa(self):
        return self.gamma_hat

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


# CONSERTAR OS VALORES DA SOMA
class ControlVariate:
    def __init__(self, n):
        # phi(x) = -0.39727x + 1
        self.integration_phi = 0.801365  # integral de phi no intervalo (0,1)
        points = np.random.uniform(0, 1, n)
        f_values = (np.exp(-A * points) * np.cos(B * points))
        varphi_values = np.array((-0.39727 * points) + 1)
        result = f_values.sum() - varphi_values.sum()

        # soma = 0
        # for i in range(n):
        #     soma += (np.exp(-A * points[i]) * np.cos(B * points[i])) \
        #             - (-0.39727 * points[i] + 1) \
        #             + self.integration_phi
        self.gamma_hat = (result.sum() + self.integration_phi) / n

    def estimativa(self):
        return self.gamma_hat


if __name__ == '__main__':
    main()

