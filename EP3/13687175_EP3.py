import time
import numpy as np
from scipy.stats import qmc, beta, uniform

SEED = 13687175
A = 0.386617636
B = 0.47950399848


def main():
    # print('\n \n')
    # n, t = 1600, 1000
    # print(f'\tCRUDE (n={n}, t={t})')
    # crude = Crude(n, t)
    # print(f'\tMÉDIA: {crude.media_estimativa():0.10f}')
    # print(f'\tVARIÂNCIA: {crude.var_estimativa():0.10f}')
    # print(f'\tPRECISÃO: {crude.precisao():0.10f}')
    # print(f'\tTEMPO DE EXECUÇÃO: {crude.media_tempo():0.10f}')

    # print('\n \n')
    # n, t = 1000, 1000
    # print(f'\tHIT OR MISS (n={n}, t={t})')
    # hit_miss = HitOrMiss(n, t)
    # print(f'\tESTIMATIVA: {hit_miss.media_estimativa():0.10f}')
    # print(f'\tVARIÂNCIA: {hit_miss.var_estimativa():0.10f}')
    # print(f'\tTEMPO: {hit_miss.media_tempo():0.10f}')
    # print(f'\tPRECISÃO: {hit_miss.precisao():0.10f}')

    # print('\n \n')
    # n, t = 8000, 1000
    # print(f'\tIMPORTANCE SAMPLING (n={n}, t={t})')
    # imp_samp = ImportanceSampling(n, t)
    # print(f'\tESTIMATIVA: {imp_samp.media_estimativa():0.10f}')
    # print(f'\tVARIÂNCIA: {imp_samp.var_estimativa():0.10f}')
    # print(f'\tTEMPO: {imp_samp.media_tempo():0.10f}')
    # print(f'\tPRECISÃO: {imp_samp.precisao():0.10f}')

    print('\n \n')
    n, t = 5, 1000
    print(f'\tCONTROL VARIATE (n={n}, t={t})')
    control = ControlVariate(n, t)
    print(f'\tESTIMATIVA: {control.media_estimativa():0.10f}')
    print(f'\tVARIÂNCIA: {control.var_estimativa():0.10f}')
    print(f'\tTEMPO: {control.media_tempo():0.10f}')
    print(f'\tPRECISÃO: {control.precisao():0.10f}')


class Crude:
    def __init__(self, n, t, d=1):
        self.ESTIMATIVAS = np.array([])
        self.n = n
        self.t = t

        start_time = time.time()
        cont = 0
        for i in range(t):
            estimativa = self.experimento(seed=SEED*i)
            self.ESTIMATIVAS = np.append(self.ESTIMATIVAS, estimativa)
            if estimativa >= 0.9995*0.800931:
                cont += 1

        self.ic = cont / self.t
        self.tempo = time.time() - start_time

    def experimento(self, seed=SEED):
        # Choose randomly points in [0,1) using Uniform distribution
        points = halton(self.n, d=1, seed=seed)
        f_values = np.exp(-A * points) * np.cos(B * points)
        estimativa = f_values.sum() / self.n

        return estimativa

    def media_estimativa(self):
        return self.ESTIMATIVAS.mean()

    def var_estimativa(self):
        return self.ESTIMATIVAS.var()

    def media_tempo(self):
        return self.tempo / self.t

    def precisao(self):
        return self.ic


# VERIFICAR SE ESTÁ CERTO
class HitOrMiss:
    def __init__(self, n, t):
        self.ESTIMATIVAS = np.array([])
        self.n = n
        self.t = t

        start_time = time.time()
        cont = 0
        for i in range(t):
            estimativa = self.experimento(seed=SEED * i)
            self.ESTIMATIVAS = np.append(self.ESTIMATIVAS, estimativa)
            if estimativa >= 0.9995 * 0.800931:
                cont += 1

        self.ic = cont / self.t
        self.tempo = time.time() - start_time

    def experimento(self, seed=SEED):
        points = halton(self.n, d=2, seed=seed).T
        points = np.reshape(points, (2, self.n))

        # Lista que vai armazenar os valores da função f(x)
        f_values = (np.exp(-A * points[0]) * np.cos(B * points[0]))
        # retorna um array com TRUE se y < f_values. FALSE, caso contrario.
        h = (np.array(points[1]) <= f_values)

        # h.sum() retorna a soma de todos os valores do array
        estimativa = h.sum() / self.n
        return estimativa

    def media_estimativa(self):
        return self.ESTIMATIVAS.mean()

    def var_estimativa(self):
        return self.ESTIMATIVAS.var()

    def media_tempo(self):
        return self.tempo / self.t

    def precisao(self):
        return self.ic


class ImportanceSampling:
    def __init__(self, n, t):
        self.ESTIMATIVAS = np.array([])
        self.n, self.t = n, t
        # g(x) = Beta Distribuition(alpha, beta)
        self.alpha, self.beta = 0.35, 1

        start_time = time.time()
        cont = 0
        for i in range(t):
            estimativa = self.experimento(seed=SEED * i)
            self.ESTIMATIVAS = np.append(self.ESTIMATIVAS, estimativa)
            if estimativa >= 0.9995 * 0.800931: cont += 1

        self.ic = cont / self.t
        self.tempo = time.time() - start_time

    def experimento(self, seed=SEED):
        points = halton_beta(self.n, self.alpha, self.beta, d=1, seed=seed).T  # geradora de pontos
        f_values = (np.exp(-A * points) * np.cos(B * points))  # guarda todos os valores da funcao f(x)
        beta_values = beta.pdf(points, self.alpha, self.beta)  # guarda todos os valores da funcao beta(alpha, beta)

        # (f / beta_values).sum() retorna um array com a soma de todas as divisoes feitas entre 'f' e 'beta_values'
        estimativa = (f_values / beta_values).sum() / self.n  # estimativa da importance sample
        return estimativa

    def media_estimativa(self):
        return self.ESTIMATIVAS.mean()

    def var_estimativa(self):
        return self.ESTIMATIVAS.var()

    def media_tempo(self):
        return self.tempo / self.t

    def precisao(self):
        return self.ic


# CONSERTAR OS VALORES DA SOMA
class ControlVariate:
    def __init__(self, n, t):
        self.ESTIMATIVAS = np.array([])
        self.n, self.t = n, t
        # phi(x) = -0.39727x + 1
        self.integration_phi = 0.801365  # integral de phi no intervalo (0,1)

        start_time = time.time()
        cont = 0
        for i in range(t):
            estimativa = self.experimento(seed=SEED * i)
            self.ESTIMATIVAS = np.append(self.ESTIMATIVAS, estimativa)
            if estimativa >= 0.9995 * 0.800931:
                cont += 1

        self.ic = cont / self.t
        self.tempo = time.time() - start_time

    def experimento(self, seed=SEED):
        points = halton(self.n, d=1, seed=seed).T
        f_values = (np.exp(-A * points) * np.cos(B * points))  # valores de f(x)
        phi_values = np.array(-0.39727 * points + 1)  # valores da função que aproxima f(x)

        estimativa = (f_values - phi_values + self.integration_phi).sum() / self.n
        return estimativa

    def media_estimativa(self):
        return self.ESTIMATIVAS.mean()

    def var_estimativa(self):
        return self.ESTIMATIVAS.var()

    def media_tempo(self):
        return self.tempo / self.t

    def precisao(self):
        return self.ic


def halton(n, d=1, seed=SEED):
    sampler = qmc.Halton(d, scramble=True, seed=seed)
    return sampler.random(n)


def halton_beta(n, a, b, d=1, seed=SEED):
    sampler = qmc.Halton(d, scramble=True, seed=seed)
    x = sampler.random(n)
    return beta.ppf(x, a, b)


if __name__ == '__main__':
    main()

