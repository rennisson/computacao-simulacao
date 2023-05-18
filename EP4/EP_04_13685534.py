import sys
import numpy as np
import scipy.stats as stats

# Definições importantes 
N = 10000000
x = np.array([1, 2, 3])
y = np.array([5, 2, 2])
alpha = alpha2 = x+y
k = 100000
v = 10

# Gerar valores das "tríades" teta, f(teta) e o maximo (v_k)
teta = stats.dirichlet.rvs(alpha, size=N)
f = stats.dirichlet.pdf(teta.T, alpha)
f.sort()

f = np.split(f, k)
soma = 0
for c in range(0, k):
    soma += f[c].size
    if(max(f[c]) >= v):
        break
    
print(soma/ N)


