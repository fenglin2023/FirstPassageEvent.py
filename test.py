# this is the test code of the FirstPassageEvent algorithm
# we consider the empirical cdf of the first passage time of a suborinator,
# whose truncated tempered stable component has levy density 1{0<x<1}e^(-x)x^(-alpha-1)dx
# and compound Poisson compounent has levy density 1{0<x<1}(1-e^(-x))x^(-alpha-1)dx+1{x>1}x^(-alpha-1)dx.
# we get the ecdf via algorithm, and via its law of $(a0/S(1))^alpha$ directly

random.seed(2023)
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

alpha = 0.6
vartheta = 1.
q = 1.0
a0 = 3.0
r = 1.0
Lambda1 = 2.20171  # \int_0^1 (1-e^(-qx))x^(-alpha-1)dx
Lambda2 = 1.66667  # \int_1^infty x^(-alpha-1)dx

def my_rand():
    if np.random.rand() < Lambda1 / (Lambda1 + Lambda2):
        X = np.random.rand()**(1 / (1 - alpha))
        while np.random.rand() > (1 - np.exp(-q * X)) / (q * X):
            X = np.random.rand()**(1 / (1 - alpha))
    else:
        X = np.random.rand()**(-1 / alpha)
    return X

n = 10000

FPE = [rand_crossing_subordinator(alpha,vartheta,q,a0,0,r,1/2,Lambda1+Lambda2,my_rand) for _ in range(n)]
XT = [t[0] for t in FPE]# vector of samples of the first passage time
FT = [(a0/rand_stable(alpha,vartheta*gamma(1-alpha)/alpha,1.))**alpha for _ in range(n)]
x = np.linspace(0,5,100)
ecdf1 = ECDF(XT)
y1 = ecdf1(x)
ecdf2 = ECDF(FT)
y2 = ecdf2(x)
plt.plot(x,y1,label='from algorithm')
plt.plot(x,y2,label='from distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of empirical cdfs')
plt.legend()
plt.show()
