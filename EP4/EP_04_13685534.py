import time
import numpy as np
import scipy.stats as stats


def main():
    # Definições importantes 
    np.random.seed(13685534)
    N = int(input("Digite o valor para N: "))
    k = int(input("Digite o valor para K: "))

    x = np.random.randint(1, 10, 3)
    y = np.random.randint(1, 10, 3)
    alpha = x + y
    val_f = gera_valores(alpha, k, N)

    print(f"Valor máximo global de f: {np.amax(val_f)}\n")

    while True:
        v = float(input("Valor de v (-1 para terminar): "))
        if v == -1:
            break
        else:
            print(calcula_U_teste(val_f, k, N, v))


def gera_valores(alpha, k, N):
    # Gerar valores das "tríades" teta, f(teta) e o maximo (v_k)
    teta = stats.dirichlet.rvs(alpha, size=N)
    f = stats.dirichlet.pdf(teta.T, alpha)
    f.sort()
    f = np.split(f, k)
    return f


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
        return soma / N


def calcula_U_teste(f, k, N, v):
    ref = calcula_U(f, k, 1000000, v, 0)
    arr = np.zeros(100)
    for c in range(0, 100):
        arr[c] = calcula_U(f, k, N, v, 0)
    t_1 = arr <= ref * 1.0005
    t_2 = arr >= ref * 0.9995
    t = t_1 & t_2
    return t.sum()


if __name__ == '__main__':
    main()
