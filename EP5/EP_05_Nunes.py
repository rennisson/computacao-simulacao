import random 
import time
import numpy as np
import scipy.stats as stats

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
    N = int(input("Digite o valor para N: "))
    k = int(input("Digite o valor para K: "))

    # Gerar x, y, alpha
    x = np.random.randint(1, 10, 3)
    y = np.random.randint(1, 10, 3)
    alpha = x + y

    # Gerar os valores usados para os cálculos de acordo com a Dirichlet
    val_f = gera_valores(alpha, k, N)
    ref_f = gera_valores(alpha, 1000000, 10000000)

    print(f"Valor máximo global de f: {np.amax(val_f)}\n")

    while True:
        v = float(input("Valor de v (-1 para terminar): "))
        if v == -1:
            break
        else:
            precisao, massa, tempo = calcula_U_teste(val_f, k, N, v, ref_f)
            print(f'U(v): {massa:0.4f}')
            print(f'Precisão: {precisao}%')
            print(f'Tempo: {tempo:0.4f}\n')

# Gera valores já repartidos de acordo com a Dirichlet
def gera_valores(theta, k, N): 
    f = np.zeros((N, 3))
    theta_1 = stats.multivariate_normal.rvs((0.3, 0.3, 0.4))            # Primeiro Theta arbitrário

    while True:                                                        
        if(theta_1[0] < 0 or theta_1[1] < 0 or theta_1[2] < 0): 
            theta_1 = stats.multivariate_normal.rvs((0.3, 0.3, 0.4))
        else: 
            theta_1 = theta_1 / theta_1.sum()
            break 

    for c in range(0, 200):                                             # Aquecimento da cadeira
        theta_2 = stats.multivariate_normal.rvs(theta_1, cov=0.06)
        if(theta_2[0] < 0 or theta_2[1] < 0 or theta_2[2] < 0):         # Rejeitar se tiver fora do domínio
            alpha = 0
        else: 
            theta_2 = theta_2 / theta_2.sum()
            alpha = stats.dirichlet.pdf(theta_2, theta) / stats.dirichlet.pdf(theta_1, theta)
        if(alpha >= 1):                                                 # Aceitar o rejeitar de acordo com o critério do algorítmo
            f[c] = theta_2
        else:
            rand = random.uniform(0, 1)
            if(rand < alpha):
                f[c] = theta_2
            else: 
                f[c] = theta_1
    theta_1 = f[199]

    for c in range(0, N):                                               # Parte principal de geração de f
        theta_2 = stats.multivariate_normal.rvs(theta_1, cov=0.06)
        if(theta_2[0] < 0 or theta_2[1] < 0 or theta_2[2] < 0):         # Rejeitar se tiver fora do domínio
            alpha = 0
        else: 
            theta_2 = theta_2 / theta_2.sum()
            alpha = stats.dirichlet.pdf(theta_2, theta) / stats.dirichlet.pdf(theta_1, theta)
        if(alpha >= 1):                                                 # Aceitar o rejeitar de acordo com o critério do algorítmo
            f[c] = theta_2
        else:
            rand = random.uniform(0, 1)
            if(rand < alpha):
                f[c] = theta_2
            else: 
                f[c] = theta_1
        theta_1 = f[c]
    g = stats.dirichlet.pdf(f.T, theta)                                     
    g.sort()
    g = np.split(g, k)
    return g

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

# Teste para estimação da precisão da função calcula_U
def calcula_U_teste(f, k, N, v, ref_f):
    ref = calcula_U(ref_f, 1000000, 10000000, v, 0)
    arr = np.zeros(100)
    tempo = np.zeros(100)
    for c in range(0, 100):
        start_time = time.time()
        arr[c] = calcula_U(f, k, N, v, 0)
        tempo[c] = time.time() - start_time
    t_1 = arr <= ref * 1.0005
    t_2 = arr >= ref * 0.9995
    t = t_1 & t_2
    return t.sum(), arr.mean(), tempo.mean()


if __name__ == '__main__':
    main()
