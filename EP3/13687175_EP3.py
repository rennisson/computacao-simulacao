import time
import numpy as np
from scipy.stats import beta
from scipy.stats import qmc

SEED = 13687175
A = 0.386617636
B = 0.47950399848

def main():
    np.random.seed(SEED)

    print('\n \n')
    print('CRUDE')
    print('{0:^10} | {1:^25} | {2:^25}'.format('Sample',
                                               f'Estimativa',
                                               f'Tempo'))
    print('-'*35)
    for i in range(1, 10):
        #n = 13465688
        crude = Crude(10000*i)
        print('{0:^10} | {1:^25} | {2:^25}'.format(f'{1000*i}',
                                         f'{crude.estimativa()}',
                                         f'{crude.time}'))

    print('\n \n')
    print('Hit or Miss')
    print('{0:^10} | {1:^25} | {2:^25}'.format('Sample',
                                               f'Estimativa',
                                               f'Tempo'))
    print('-' * 35)
    for i in range(1, 10):
        #n = 10128002
        hom = HitOrMiss(1000*i)
        print('{0:^10} | {1:^25} | {2:^25}'.format(f'{1000*i}',
                                                   f'{hom.estimativa()}',
                                                   f'{hom.time}'))
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
    #
    # print('\n \n')
    # print('CONTROL VARIATE')
    # print('{0:^10} | {1:^25}'.format('Sample',
    #                                  f'Estimativa'))
    # print('-' * 35)
    # for i in range(1, 10):
    #     n = 3059462
    #     control = ControlVariate(n)
    #     print('{0:^10} | {1:^25}'.format(f'{n}',
    #                                      f'{control.estimativa()}'))


class Crude:
    def __init__(self, n, d=1):
        # Choose randomly points in [0,1) using Uniform distribution
        start_time = time.time()
        points = halton(n, d=1)
        f_values = np.exp(-A * points) * np.cos(B * points)

        self.gamma_hat = f_values.sum() / n
        self.time = time.time() - start_time

    def estimativa(self):
        return self.gamma_hat


# VERIFICAR SE ESTÁ CERTO
class HitOrMiss:
    def __init__(self, n):
        start_time = time.time()
        points = halton(n, d=2)
        # Lista que vai armazenar os valores da função f(x)
        f_values = np.exp(-A * points[0]) * np.cos(B * points[1])
        # retorna um array com TRUE se y < f_values. FALSE, caso contrario.
        h = (np.array(points[1]) <= f_values)

        # h.sum() retorna a soma de todos os valores do array
        self.gamma_hat = h.sum() / n
        self.time = time.time() - start_time

    def estimativa(self):
        return self.gamma_hat


class ImportanceSampling:
    def __init__(self, n):
        # g(x) = Beta Distribuition(alpha, beta)
        self.alpha = 0.35
        self.beta = 1
        points = np.random.beta(self.alpha, self.beta, n)  # geradora de pontos da função beta
        f_values = np.exp(-A * points) * np.cos(B * points)  # guarda todos os valores da funcao f(x)
        beta_values = beta.pdf(points, self.alpha, self.beta)  # guarda todos os valores da funcao beta(alpha, beta)

        # (f / beta_values).sum() retorna um array com a soma de todas as divisoes feitas entre 'f' e 'beta_values'
        self.gamma_hat = (f_values / beta_values).sum() / n  # estimativa da importance sample

    def estimativa(self):
        return self.gamma_hat


# CONSERTAR OS VALORES DA SOMA
class ControlVariate:
    def __init__(self, n):
        # phi(x) = -0.39727x + 1
        self.integration_phi = 0.801365  # integral de phi no intervalo (0,1)
        points = np.random.uniform(0, 1, n)
        f_values = (np.exp(-A * points) * np.cos(B * points))  # valores de f(x)
        phi_values = np.array(-0.39727 * points + 1)  # valores da função que aproxima f(x)

        self.gamma_hat = (f_values - phi_values + self.integration_phi).sum() / n

    def estimativa(self):
        return self.gamma_hat


def halton(n, d=1):
    sampler = qmc.Halton(d, scramble=True)
    return np.array(sampler.random(n, workers=-1).T)


if __name__ == '__main__':
    main()

