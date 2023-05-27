#this is the example of OU process, for more detail, see Section 4.2 
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.special import gamma, beta, expit, logit, logsumexp, erf
from scipy.special import eval_jacobi, gammaln,  polygamma, exp1, erfinv
from scipy.stats import expon, multivariate_normal
from scipy import special
from scipy import stats
from matplotlib import rc

random.seed(2023)

alpha = 0.65
theta = 1 / (-gamma(-alpha))
q = 1.0
r = 1

n = int(1e3)
xarr = np.arange(0, 4.05, 0.05)
yarr = np.arange(0, 4.05, 0.05)
T = np.zeros(n)
U = np.zeros(n)

u = np.zeros((len(xarr), len(yarr)))
gmma = np.array([[1, 1], [0, 1]])  # gmma is the volatility of the SDE


def myrand_lambda():  # Q has levy density 1{s>1}s^(-5)
    return np.power(np.random.rand(), -1 / 4)


for k in range(1, n + 1):
    if k % 10 == 0:
        print(f"Iteration for {k}")
    t = 5  # t is the boundary
    if U[k - 1] < t:
        auxT, auxU, auxV = rand_crossing_subordinator(alpha, theta, q, t - U[k - 1], 0, 1, 1 / 2, 1 / 4, myrand_lambda)
        #the function rand_crossing_subordinator is provided in source.py
        T[k - 1] += auxT
        U[k - 1] += auxU + auxV

    for i in range(len(xarr)):
        for j in range(len(yarr)):
            dd = multivariate_normal([0, 0], gmma * np.transpose(gmma) * ((1 - np.exp(-2 * T[k - 1])) / 2))
            x = np.exp(-np.array([[1, 0], [0, 1]]) * T[k - 1]).dot(np.array([[xarr[i]], [yarr[j]]])) + dd.rvs(1)
            u[i, j] += x[0, 0] + x[1, 0] ** 2

u /= n

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(xarr, yarr)
ax.plot_surface(X, Y, u)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$u$")
ax.set_title(r"Solution $u(t,x)$ of the FPDE")
plt.show()