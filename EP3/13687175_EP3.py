import time
import numpy as np
from scipy.stats import qmc, beta, uniform

SEED = 13687175
A = 0.386617636
B = 0.47950399848

def main():
    print('\n \n')
    print('CRUDE')
    print('{0:^10} | {1:^25} | {2:^25}'.format('Sample',
                                               f'Estimativa',
                                               f'Tempo'))
    print('-'*35)
    for i in range(1, 10):
        #n = 13465688
        crude = Crude(100*i)
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
        n = 1000*i
        hom = HitOrMiss(n)
        print('{0:^10} | {1:^25} | {2:^25}'.format(f'{n}',
                                                   f'{hom.estimativa()}',
                                                   f'{hom.time}'))

    print('\n \n')
    print('IMPORTANCE SAMPLING')
    print('{0:^10} | {1:^25} | {2:^25}'.format(f'Sample',
                                               f'Estimativa',
                                               f'Tempo'))
    print('-' * 35)
    for i in range(1, 10):
        n = 1000*i
        imp_sample = ImportanceSampling(n)
        print('{0:^10} | {1:^25} | {2:^25}'.format(f'{n}',
                                                   f'{imp_sample.estimativa()}',
                                                   f'{imp_sample.time}'))

    print('\n \n')
    print('CONTROL VARIATE')
    print('{0:^10} | {1:^25}'.format('Sample',
                                     f'Estimativa'))
    print('-' * 35)
    for i in range(1, 10):
        #n = 3059462
        n = 10*i
        control = ControlVariate(n)
        print('{0:^10} | {1:^25} | {2:^25}'.format(f'{n}',
                                                   f'{control.estimativa()}',
                                                   f'{control.time}'))


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
        points = halton(n, d=2).T
        points = np.reshape(points, (2, n))

        # Lista que vai armazenar os valores da função f(x)
        f_values = (np.exp(-A * points[0]) * np.cos(B * points[0]))
        # retorna um array com TRUE se y < f_values. FALSE, caso contrario.
        h = (np.array(points[1]) <= f_values)

        # h.sum() retorna a soma de todos os valores do array
        self.gamma_hat = h.sum() / n
        self.time = time.time() - start_time

    def estimativa(self):
        return self.gamma_hat


class ImportanceSampling:
    def __init__(self, n):
        start_time = time.time()
        # g(x) = Beta Distribuition(alpha, beta)
        self.alpha = 0.35
        self.beta = 1
        points = halton_beta(n, self.alpha, self.beta, d=1).T  # geradora de pontos
        f_values = (np.exp(-A * points) * np.cos(B * points))  # guarda todos os valores da funcao f(x)
        beta_values = beta.pdf(points, self.alpha, self.beta)  # guarda todos os valores da funcao beta(alpha, beta)

        # (f / beta_values).sum() retorna um array com a soma de todas as divisoes feitas entre 'f' e 'beta_values'
        self.gamma_hat = (f_values / beta_values).sum() / n  # estimativa da importance sample
        self.time = time.time() - start_time

    def estimativa(self):
        return self.gamma_hat


# CONSERTAR OS VALORES DA SOMA
class ControlVariate:
    def __init__(self, n):
        start_time = time.time()
        # phi(x) = -0.39727x + 1
        self.integration_phi = 0.801365  # integral de phi no intervalo (0,1)
        points = halton(n, d=1).T
        f_values = (np.exp(-A * points) * np.cos(B * points))  # valores de f(x)
        phi_values = np.array(-0.39727 * points + 1)  # valores da função que aproxima f(x)

        self.gamma_hat = (f_values - phi_values + self.integration_phi).sum() / n
        self.time = time.time() - start_time

    def estimativa(self):
        return self.gamma_hat


def sobol(n, d=1):
    sampler = qmc.Sobol(d, scramble=True, seed=SEED)
    return sampler.random(n)


def halton(n, d=1, seed=SEED):
    sampler = qmc.Halton(d, scramble=True, seed=seed)
    return sampler.random(n)


def halton_beta(n, a, b, d=1):
    sampler = qmc.Halton(d, scramble=True, seed=SEED)
    x = sampler.random(n)
    return beta.ppf(x, a, b)


def halton_uniform(n, d=1):
    sampler = qmc.Halton(d, scramble=True, seed=SEED)
    x = sampler.random(n)
    return uniform.ppf(x)


if __name__ == '__main__':
    main()

