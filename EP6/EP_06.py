import time
import numpy as np
import scipy.stats as stats
import math

'''
Nome : Gustavo Silva de Almeida Nunes, Rennisson Davi Alves
NUSP : 13685534, 13687175

Propriedades do código: 
A função "teste" serve para teste do N para o intervalo de confiança. Fazem 100 testes com a
função de estimativa de U retorna quantas estão na margem de erro definida, 
com os valores de referência gerado usando N_ref. 

Testes foram feitos com a seed definida. 
'''

def main():
    # Definições importantes 
    np.random.seed(13685534)
    N = 10000 #int(input("Digite o valor para N: "))
    k = 1000 #int(input("Digite o valor para K: "))

    # Informação a Priori
    prior = np.array([1, 1, 1])
    prior_teste = np.array([0, 0, 0])
    n = 20

    # Observações
    x_1 = int(input("Valor de x1: "))
    x_3 = int(input("Valor de x3: "))
    x_2 = n - x_1 - x_3
    obs = np.array([x_1, x_2, x_3])

    # Parâmetro
    alpha = obs + prior
    alpha2 = obs + prior

    # Achar valor máximo da função surpresa
    s_argmax = 0
    start_time = time.time()

    for c in range(0, 10000):
        theta_1 = c/10000
        theta_3 = (1 - math.sqrt(theta_1))**2
        theta_2 = 1 - theta_1 - theta_3
        teta = (theta_1, theta_2, theta_3)
        f = stats.dirichlet.pdf(teta, alpha)
        if f > s_argmax:
            s_argmax = f

    val_f = gera_valores(alpha, k, N)
    val_f2 = gera_valores(alpha2, k, N)
    massa = calcula_U(val_f, k, N, s_argmax, 0)
    massa_teste = calcula_U(val_f2, k, N, s_argmax, 0)

    exec_time = time.time() - start_time
    # Print valor de ev
    print(f"s*: {s_argmax}")
    print(f"W(s*): {massa}")
    print(f'Precisão: {massa / massa_teste}')
    print(f'Tempo: {exec_time:0.4f}\n')


# Gera valores já repartidos de acordo com a Dirichlet
def gera_valores(alpha, k, N):
    # Gerar valores das "tríades" teta, f(teta) e o maximo (v_k)
    teta = stats.dirichlet.rvs(alpha, size=N)
    f = stats.dirichlet.pdf(teta.T, alpha)
    f.sort()
    f = np.split(f, k)
    return f


# Calcula o valor de U(v)
def calcula_U(f, k, N, v, mode):
    start_time = time.time()
    soma = 0
    for c in range(0, int(k)):
        soma += f[c].size
        if max(f[c]) >= v:
            break
    if mode == 1:
        exec_time = time.time() - start_time
        print(f'U(v): {soma / N:0.4f}')
        print(f'Tempo: {exec_time:0.4f}\n')
    else:
        return float(soma) / float(N)


if __name__ == '__main__':
    main()