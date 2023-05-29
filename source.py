#source
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

######################################
# Auxiliary parameters and functions #
######################################

# Number of nodes in Gaussian quadrature
n_nodes = 2 ** 10

gl_x, gl_w = np.polynomial.legendre.leggauss(n_nodes)

def sigma(alpha, x):
    return np.where(x == 0., (1. - alpha) * alpha ** (alpha / (1. - alpha)), np.sin((1. - alpha) * np.pi * x) * np.sin(alpha * np.pi * x) ** (alpha / (1. - alpha)) * np.sin(np.pi * x) ** (-1. / (1. - alpha)))

def Dsigma(alpha, x):
    return np.where(x == 0., 0., np.pi * sigma(alpha, x) * (alpha ** 2 * 1/math.tan(alpha * np.pi * x) + (1. - alpha) ** 2 * 1/math.tan((1. - alpha) * np.pi * x) - 1/math.tan(np.pi * x)) / (1. - alpha))

def D2sigma(alpha, x):
    return np.pi ** 2 * sigma(alpha, x) * ((alpha ** 2 * 1/math.tan(alpha * np.pi * x) + (1. - alpha) ** 2 * 1/math.tan((1. - alpha) * np.pi * x) - 1/math.tan(np.pi * x)) ** 2 / (1. - alpha) ** 2 + (- alpha ** 3 * (1. / np.sin(alpha * np.pi * x)) ** 2 - (1. - alpha) ** 3 * (1. / np.sin((1. - alpha) * np.pi * x)) ** 2 + (1. / np.sin(np.pi * x)) ** 2) / (1. - alpha))

def DlogDsigma(alpha, x):
    aux = alpha ** 2 * 1/math.tan(alpha * np.pi * x) + (1. - alpha) ** 2 * 1/math.tan((1. - alpha) * np.pi * x) - 1/math.tan(np.pi * x)
    return np.pi * aux / (1 - alpha) + (- alpha ** 3 * (1. / np.sin(alpha * np.pi * x)) ** 2 - (1. - alpha) ** 3 * (1. / np.sin((1. - alpha) * np.pi * x)) ** 2 + (1. / np.sin(np.pi * x)) ** 2) / aux

def Dlogsigma(alpha, x):
    return np.where(x == 0., 0., np.pi * (alpha ** 2 * 1/math.tan(alpha * np.pi * x) + (1 - alpha) ** 2 * 1/math.tan((1 - alpha) * np.pi * x) - 1/math.tan(np.pi * x)) / (1 - alpha))

def D2logsigma(alpha, x):
    return np.pi ** 2 * (1. / np.sin(np.pi * x)) ** 2 - alpha ** 3 * (1. / np.sin(alpha * np.pi * x)) ** 2 - (1 - alpha) ** 3 * (1. / np.sin((1 - alpha) * np.pi * x)) ** 2 / (1 - alpha)

def D2sigma0(alpha, x):
    return math.pi**2 * sigma0(alpha, x) * (alpha**2 * 1/math.tan(alpha * math.pi * x) + (1 - alpha)**2 * 1/math.tan((1 - alpha) * math.pi * x) - 1/math.tan(math.pi * x))**2 + math.pi**2 * sigma0(alpha, x) * (1 / math.sin(math.pi * x)**2 - alpha**3 / math.sin(alpha * math.pi * x)**2 - (1 - alpha)**3 / math.sin((1 - alpha) * math.pi * x)**2)

def sigma0(alpha, x):
    if x ==0:
        return ((1. - alpha) * alpha ** (alpha / (1. - alpha)))**(1/(1-alpha))
    else:
        return math.sin((1 - alpha) * math.pi * x)**(1 - alpha) * math.sin(alpha * math.pi * x)**alpha / math.sin(math.pi * x)

def Dsigma0(alpha, x):
    return math.pi * sigma0(alpha, x) * (alpha**2 * 1/math.tan(alpha * math.pi * x) + (1 - alpha)**2 * 1/math.tan((1 - alpha) * math.pi * x) - 1/math.tan(math.pi * x))

def Dlogsigma0(alpha, x):
    if x == 0:
        return 0
    else:
        return math.pi * (alpha**2 * 1/math.tan(alpha * math.pi * x) + (1 - alpha)**2 * 1/math.tan((1 - alpha) * math.pi * x) - 1/math.tan(math.pi * x))

def D2sigma0(alpha, x):
    return math.pi**2 * sigma0(alpha, x) * (alpha**2 * 1/math.tan(alpha*math.pi*x) + (1-alpha)**2 * 1/math.tan((1-alpha)*math.pi*x) - 1/math.tan(math.pi*x))**2 + math.pi**2 * sigma0(alpha, x) * (1 / math.sin(math.pi*x)**2 - alpha**3 / math.sin(alpha*math.pi*x)**2 - (1-alpha)**3 / math.sin((1-alpha)*math.pi*x)**2)

def Psiaux(a, x):
    ca = 1 - a
    ax = a * x
    cax = ca * x
    gax = gamma(1 + ax)
    gcax = gamma(1 + cax)
    ecax = math.exp(cax)
    eax = math.exp(ax - 1)
    axx = ax ** ax
    caxx = cax ** cax
    sqax = math.sqrt(2 * math.pi * a * cax)

    C1 = float('inf') if x == 0 else (
        gax * eax * (a / ca + ax) ** (1 + cax) / ax ** (x + 1)
    )
    C2 = gcax * ecax / caxx
    C3 = gax * eax * (1 + 1 / cax) ** (1 + cax) / (axx * sqax)
    C4 = gcax * ecax / (caxx * sqax)

    Cmin = min(C1, C2, C3, C4)

    if C1 == Cmin:
        return (1, math.exp(x) * gax * x ** (a / ca) / (ca * C1))
    elif C2 == Cmin:
        return (2, math.exp(x) * gcax / C2)
    elif C3 == Cmin:
        return (
            3,
            erf(sqax * math.sqrt(math.pi) / 2) * math.exp(x) * gax * x ** (a / ca) / (C3 * sqax)
        )
    else:  # C4 == Cmin
        return (
            4,
            erf(sqax * math.sqrt(math.pi) / 2) * math.exp(x) * gcax / (C4 * sqax)
        )
    
    
def varphi(alpha, x):
    aux = sigma(alpha, (1 + gl_x) / 2)
    return (alpha / (2 - 2 * alpha)) * x ** (-1 / (1 - alpha)) * np.sum(gl_w * aux * np.exp(-aux * x ** (-alpha / (1 - alpha))))


def g(alpha, θ, t, x):
    sc = (θ * t) ** (-1 / alpha)
    return varphi(alpha, sc * x) * sc


def G(alpha, θ, t, x):
    sc = (θ * t) ** (-1 / alpha)
    aux = sigma(alpha, (1 + gl_x) / 2)
    return (1 / 2) * np.sum(gl_w * np.exp(-aux * (sc * x) ** (-alpha / (1 - alpha))))


def beta_inc(a, b, x):
    return beta(a, b) * (x ** a) * ((1 - x) ** b)


def Devroye_logconcave(f, a, b, c, L):
    if c != 0:
        l1 = np.log(b)
        l2 = np.log(c)
        dl = l1 - l2
        s = 1 / (1 + b + c / dl)
        X = 0
        while True:
            U = np.random.rand()
            if U < s:
                X = np.random.rand() * a
            elif U < (1 + b) * s:
                X = a * (1 + np.random.rand())
            else:
                X = a * (2 + np.random.exponential() / dl)
            if X < L and np.random.rand() < f(X) / (1. if X < a else (b if X < 2 * a else np.exp(((2 * a - X) * l1 + (X - a) * l2) / a))):
                return X
    else:
        l1 = np.log(b)
        s = 1 / (1 + b)
        X = 0
        while True:
            U = np.random.rand()
            if U < s:
                X = np.random.rand() * a
            else:
                X = a * (1 + np.random.rand())
            if np.random.rand() < f(X) / (1. if X < a else b):
                return X
            
def invsigma(alpha, x):
    s0 = sigma(alpha, 0.)
    if x <= s0:
        return 0.

    n = 6
    N = 60
    eps = 1e-15

    s = np.log(x) * (1 - alpha)

    def varsigma(x):
        return alpha * np.log(np.sin(alpha * np.pi * x)) + (1 - alpha) * np.log(np.sin((1 - alpha) * np.pi * x)) - np.log(np.sin(np.pi * x)) - s

    def Dvarsigma(x):
        return np.pi * (alpha ** 2 * 1/np.tan(alpha * np.pi * x) + (1 - alpha) ** 2 * 1/np.tan((1 - alpha) * np.pi * x) - 1/np.tan(np.pi * x))

    # Binary search
    M = 3 / (np.pi * (1 - alpha ** 3 - (1 - alpha) ** 3))
    y = 1
    ym = 0
    sigmap, sigmam = 1., varsigma(ym)

    i = 0
    while y - ym > eps and (y >= 1 or M * (y - ym) > ym * y ** 2 * (1 - y)) and i < N:
        i += 1
        if y < 1:
            mid1 = (y + ym) / 2
            mid2 = (ym * sigmap - y * sigmam) / (sigmap - sigmam)
            sigmamid1 = varsigma(mid1)
            sigmamid2 = varsigma(mid2)
            if sigmamid1 < 0:
                if mid1 > mid2 or not np.isfinite(mid2):
                    ym, sigmam = mid1, sigmamid1
                elif sigmamid2 < 0:
                    ym, sigmam = mid2, sigmamid2
                else:
                    ym, sigmam = mid1, sigmamid1
                    y, sigmap = mid2, sigmamid2
            else:
                if mid1 < mid2 or not np.isfinite(mid2):
                    y, sigmap = mid1, sigmamid1
                elif sigmamid2 > 0:
                    y, sigmap = mid2, sigmamid2
                else:
                    ym, sigmam = mid2, sigmamid2
                    y, sigmap = mid1, sigmamid1
        else:
            mid = (y + ym) / 2
            sigmamid = varsigma(mid)
            if sigmamid < 0:
                ym, sigmam = mid, sigmamid
            else:
                y, sigmap = mid, sigmamid

    if y - ym > eps:
        # Newton-Raphson
        for _ in range(n):
            z = y - varsigma(y) / Dvarsigma(y)  # Newton
            if z == y:
                return z
            elif z > 1 or z < 0:
                z = y - varsigma(y)
                if z > 1:
                    z = (y + 1) / 2
                elif z < 0:
                    z = y / 2
            y = z

    return y

def inv_Sigma(alpha, z, y):
    n = 6
    N = 100
    
    def varsigma(x):
        return x * sigma(alpha, x) ** alpha - y
    
    x0 = z
    sigma0 = varsigma(z)
    x1 = 0.
    sigma1 = -y

    i = 0
    M = 2 * (sigma(alpha, 1/2) / sigma(alpha, 0)) ** alpha
    
    while True:
        i += 1
        m1 = (x0 + x1) / 2
        sigmam1 = varsigma(m1)
        m2 = (x0 * sigma1 - x1 * sigma0) / (sigma1 - sigma0)
        sigmam2 = varsigma(m2)

        if sigmam1 > 0:
            if sigmam2 > 0:
                if m1 < m2:
                    x0, sigma0 = m1, sigmam1
                else:
                    x0, sigma0 = m2, sigmam2
            else:
                x0, sigma0 = m1, sigmam1
                x1, sigma1 = m2, sigmam2
        else:
            if sigmam2 > 0:
                x0, sigma0 = m2, sigmam2
                x1, sigma1 = m1, sigmam1
            else:
                if m1 < m2:
                    x1, sigma1 = m2, sigmam2
                else:
                    x1, sigma1 = m1, sigmam1

        aux1 = Dlogsigma(alpha, x0)
        aux2 = D2logsigma(alpha, x0)
        aux3 = Dlogsigma(alpha, x1)
        
        if i > 10 and (i >= N or (x1 - x0) * M * (1 + x0 * alpha * (aux2 + alpha * aux1 ** 2) / 2) < 1 + max(x1, 0) * alpha * aux3):
            break

    x0 = (x1 + x0) / 2
    
    # Newton-Raphson
    for i in range(n):
        y0 = x0 - (x0 - y / sigma(alpha, x0) ** alpha) / (1 + x0 * alpha * Dlogsigma(alpha, x0))
        if y0 == x0:
            return x0
        x0 = y0
    
    return x0

def rand_neg_pow(r, a, b, z):
    if r != 2:
        a2, b2 = a / (1 - r), b / (2 - r)
        C0 = a2 * 2**(r - 1) + b2 * 2**(r - 2)
        C = a2 * (1 - z)**(1 - r) + b2 * (1 - z) - C0

        # Newton-Raphson
        x0 = z
        y = random.random() * C + C0
        n = 100
        for i in range(1, n+1):
            aux = 1 - x0
            y0 = x0 + (a2 * aux**(1 - r) + b2 * aux**(2 - r) - y) / (a * aux**(-r) + b * aux**(1 - r))
            if y0 == x0:
                return x0
            x0 = y0
        return x0
    else:
        C0 = 2 * a + b * np.log(2)
        C = a / (1 - z) - b * np.log(1 - z) - C0

        # Newton-Raphson
        x0 = z
        y = random.random() * C + C0
        n = 100
        for i in range(1, n+1):
            aux = 1 - x0
            y0 = x0 - (a / aux - b * np.log(aux) - y) / (a / aux**2 + b / aux)
            if y0 == x0:
                return x0
            x0 = y0
        return x0
    
def int_tilt_exp_sigma(alpha, s, x):
    z = invsigma(alpha, alpha/s)
    return int_tilt_exp_sigma(alpha, s, x, z)

def int_tilt_exp_sigma(alpha, s, x, z):
    if 0 < z < x:
        # Break the integral in two: [0,z] and [z,x]
        sigmax1 = sigma(alpha, (gl_x + 1) * (z/2))
        sigmax2 = sigma(alpha, (x+z)/2 + gl_x * (x-z)/2)
        return np.sum(gl_w * (sigmax1 ** alpha * np.exp(-sigmax1 * s))) * z/2 + np.sum(gl_w * (sigmax2 ** alpha * np.exp(-sigmax2 * s))) * (x-z)/2
    else:
        sigmax1 = sigma(alpha, (gl_x + 1) * (x/2))
        return np.sum(gl_w * (sigmax1 ** alpha * np.exp(-sigmax1 * s))) * x/2

def alt_int_tilt_exp_sigma(alpha, s, x, z):
    if 0 < z < x:
        # Break the integral in two: [0,z] and [z,x]
        if x <= 1/2:
            sigmax1 = sigma(alpha, (gl_x + 1) * (z/2))
            sigmax2 = sigma(alpha, (x+z)/2 + gl_x * (x-z)/2)
            return np.sum(gl_w * (sigmax1 ** alpha * np.exp(-sigmax1 * s))) * z/2 + np.sum(gl_w * (sigmax2 ** alpha * np.exp(-sigmax2 * s))) * (x-z)/2
        elif z <= 1/2:
            sigmax1 = sigma(alpha, (gl_x + 1) * (z/2))
            I = np.sum(gl_w * (sigmax1 ** alpha * np.exp(-sigmax1 * s))) * z/2 
            if z < 1/2:
                sigmax2 = sigma(alpha, (1/2+z)/2 + gl_x * (1/2-z)/2)
                I += np.sum(gl_w * (sigmax2 ** alpha * np.exp(-sigmax2 * s))) * (1/2-z)/2
            y = gl_x * (x/2-1/4) + (x/2+1/4)
            I += -(np.exp(-sigma(alpha, x) * s) / Dsigma0(alpha,x) - np.exp(-sigma(alpha, 1/2) * s) / Dsigma0(alpha,1/2)) * (1-alpha) / s
            return  I - np.sum(gl_w * np.exp(-sigma(alpha, y) * s) * D2sigma0(alpha, y) / Dsigma0(alpha, y) ** 2) * (x/2-1/4) * (1-alpha) / s 
        else: # 1/2 < z < x
            sigmax1 = sigma(alpha, (gl_x + 1) * (1/4))
            I = np.sum(gl_w * (sigmax1 ** alpha * np.exp(-sigmax1 * s))) * 1/4 
            y = gl_x * (x/2-1/4) + (x/2+1/4)
            I += -(np.exp(-sigma(alpha, x) * s) / Dsigma0(alpha,x) - np.exp(-sigma(alpha, 1/2) * s) / Dsigma0(alpha,1/2)) * (1-alpha) / s
            return  I - np.sum(gl_w * np.exp(-sigma(alpha, y) * s) * D2sigma0(alpha, y) / Dsigma0(alpha, y) ** 2) * (x/2-1/4) * (1-alpha) / s 
    else:
        if x <= 1/2:
            sigmax1 = sigma(alpha, (gl_x + 1) * (x/2))
            return np.sum(gl_w * (sigmax1 ** alpha * np.exp(-sigmax1 * s))) * x/2
        else: # 1/2 < x <= z
            sigmax1 = sigma(alpha, (gl_x + 1) * (1/4))
            I = np.sum(gl_w * (sigmax1 ** alpha * np.exp(-sigmax1 * s))) * 1/4 
            y = gl_x * (x/2-1/4) + (x/2+1/4)
            I += -(np.exp(-sigma(alpha, x) * s) / Dsigma0(alpha,x) - np.exp(-sigma(alpha, 1/2) * s) / Dsigma0(alpha,1/2)) * (1-alpha) / s
            return  I - np.sum(gl_w * np.exp(-sigma(alpha, y) * s) * D2sigma0(alpha, y) / Dsigma0(alpha, y) ** 2) * (x/2-1/4) * (1-alpha) / s

def int_exp_sigma(alpha, s, x):
    y = sigma(alpha, (gl_x + 1) * (x/2))
    return np.sum(gl_w * np.exp(- y * s)) * x/2

def int_pow_sigma(alpha, x):
    return np.sum(gl_w * sigma(alpha, (gl_x + 1) * (x/2)) ** alpha) * x/2

def rand_exp_sigma(alpha, s, z):
    aux0 = sigma(alpha, z)
    def f(x):
        return np.exp(s * (aux0 - sigma(alpha, z+x)))

    aux = np.log(4.) / s + aux0
    a1 = (1-z)/2.
    while sigma(alpha, z + a1) > aux:
        a1 /= 2.
    a2 = f(a1)
    U = 0.
    V = np.zeros(3)

    if a1 == (1-z)/2.:
        a4 = 1. / (1. + a2)
        V = np.random.rand(3)
        U = a1 * V[0] if V[1] < a4 else a1 * (1+V[0])
        while V[2] > f(U) / (1. if U < a1 else a2):
            V = np.random.rand(3)
            U = a1 * V[0] if V[1] < a4 else a1 * (1+V[0])
        return U
    else:
        a3 = f(2. * a1)
        a4 = 1. / (1. + a2 + a3 / np.log(a2/a3))
        V = np.random.rand(3)
        U = a1 * V[0] if V[1] < a4 else (a1 * (1+V[0]) if V[1] < a4 * (1. +a2) else a1 * (2. + np.log(1/V[0])/np.log(a2/a3)))
        while U >= 1 or V[2] > f(U) / (1. if U < a1 else (a2 if U < 2. *a1 else np.exp(((2. *a1-U)*np.log(a2) + (U-a1)*np.log(a3))/a1))):
            V = np.random.rand(3)
            U = a1 * V[0] if V[1] < a4 else (a1 * (1+V[0]) if V[1] < a4 * (1. +a2) else a1 * (2. + np.log(1/V[0])/np.log(a2/a3)))
        return U
    
def rand_exp_sigma(alpha, s):
    aux0 = sigma(alpha,0.)
    def f(x):
        return np.exp(s * (aux0 - sigma(alpha, x))) if 0. < x < 1. else 0.

    aux = np.log(4.) / s + aux0
    a1 = .5
    while sigma(alpha, a1) > aux:
        a1 /= 2.
    a2 = f(a1)
    a3 = f(2. * a1) if a1 == .5 else 0.
    #print("Devroye for exp(-sigma(u)s)")
    return Devroye_logconcave(f, a1, a2, a3, 1.)

def rand_tilt_exp_sigma(alpha, s):
    r = alpha / (1. - alpha)
    z = invsigma(alpha, alpha/s)

    def f0(u):
        auxf = sigma(alpha, u)
        return auxf ** alpha * np.exp(auxf * (-s))

    p = int_tilt_exp_sigma(alpha, s, 1., z)

    # Initialise just in case (avoids certain julia errors)
    a = aux = 1.
    β = min(z,.5)
    U = np.random.rand()

    if U > int_tilt_exp_sigma(alpha, s, z, z) / p:  # Sample lies on [z,1], where f is log-concave
        #print("Devroye for sigma(u)^alpha*exp(-sigma(u)s)")
        a = (1. - z)/2.
        aux = 1. / f0(z)  # (s/alpha) ^ alpha * exp(alpha)
        g = lambda x: f0(z + x) * aux
        while g(a) <= .25:
            a *= .5
        return z + Devroye_logconcave(g, a, g(a), 0. if a == (1-z) * .5 else g(2. * a), 1. - z)
    elif z <= .5 or U < int_tilt_exp_sigma(alpha, s, β, z) / p:  # Sample lies on [0,β]. Here we sample from sigma: u ↦ usigma(u)^alpha
        #print("Sample from u ↦ u*sigma(u)^alpha on [0,β]")
        C = β*sigma(alpha,β)**alpha
        U = inv_Sigma(alpha, β, np.random.rand()*C)
        while np.random.rand() > np.exp(-sigma(alpha,U)*s) / (1. + U * alpha * Dlogsigma(alpha,U)):
            U = inv_Sigma(alpha, β, np.random.rand()*C)
    elif alpha < .5:  # Sample lies on [1/2,z]
        #print("Sample from the density u ↦ (1-u)^(-r)*(a+b(1-u)) on [1/2,z]")
        a, b = np.sin(np.pi*(1. -alpha)), np.pi*alpha*(1. -alpha)*np.cos(np.pi*alpha)
        sc = 2. **r * a**(1. -alpha)
        U = rand_neg_pow(r,a,b,z)
        aux = sigma(alpha,U)
        while np.random.rand() > aux**alpha * np.exp(-s*aux) * (1. -U)**r * sc / (a + b*(1. -U)):
            U = rand_neg_pow(r,a,b,z)
            aux = sigma(alpha,U)
    elif alpha == .5:  # Sample lies on [1/2,z]
        #print("Sample from the density u ↦ (1-u)^(-1) on [1/2,z]")
        return 1. - np.exp(np.log(2. *(1. -z)) * np.random.rand())* .5
    else:  # alpha>1/2, r>1, z>1/2 and sample lies on [1/2,z]
        #print("Sample from u ↦ ρ(u)^(r-1) on [1/2,z]")
        C0 = sigma0(alpha,.5)**(r-1.)
        C = sigma0(alpha,z)**(r-1.) - C0
        c = Dlogsigma0(alpha,.5) / sigma0(alpha,.5)
        U = invsigma(alpha, (np.random.rand() * C + C0)**(1. /(2. *alpha-1.)))
        while np.random.rand() > c*sigma0(alpha,U)*np.exp(-s*sigma(alpha,U))/Dlogsigma0(alpha,U):
            U = invsigma(alpha, (np.random.rand() * C + C0)**(1. /(2. *alpha-1.)))
    return U

def crossing_functions(alpha, a0, a1, r):
    if a0 < 0.:
        return (0, 0, 0)
    elif a1 == 0.:
        b1 = lambda t: min(a0, r)
        Db1 = lambda t: 0.
        B1 = lambda t: (min(a0, r) / t) ** alpha
        return (b1, Db1, B1)
    else:
        b2 = lambda t: min(a0 - a1 * t, r)
        aux = (a0 - r) / a1
        if aux > 0.:
            Db2 = lambda t: -a1 if t > aux else 0.
            def B2(t):
                if t > r * aux ** (-1. / alpha):
                    return (t / r) ** (-alpha)
                else:
                    ra = 1. / alpha
                    x = (a0 - t * aux ** ra) / a1
                    for i in range(50):
                        y = x - (t * x ** ra + a1 * x - a0) / (a1 + ra * t * x ** (ra - 1.))
                        if y == x:
                            return x
                        else:
                            x = y
                    return x
            return (b2, Db2, B2)
        else:
            def Db3(t): return -a1
            def B3(t):
                ra = 1. / alpha
                x = (t / a0 + (a1 / a0) ** ra) ** (-alpha)
                for i in range(50):
                    y = x - (t * x ** ra + a1 * x - a0) / (a1 + ra * t * x ** (ra - 1.))
                    if y == x:
                        return x
                    else:
                        x = y
                return x
            return (b2, Db3, B3)

###########################
# Main simulation methods #
###########################

def rand_stable(alpha, θ, t):
    return (θ * t) ** (1 / alpha) * (sigma(alpha, np.random.rand()) / expon.rvs()) ** ((1 - alpha) / alpha)

def rand_tempered_stable(alpha, θ, q, t):
    mass = (θ * t) ** (1 / alpha) * q
    ξ = mass ** alpha
    r = alpha / (1 - alpha)
    (i, Cmin) = Psiaux(alpha, ξ)
    if i == 1:
        ax = alpha * ξ
        aux = ξ ** (r + 1)
        while True:
            U, V = np.random.rand(2)
            X = np.random.gamma(ax, 1)
            ρ = sigma0(alpha, U) ** (r + 1)
            X1 = X ** (-r)
            if V <= ρ * X ** (-ax) * X1 * np.exp(-ρ * aux * X1) * Cmin:
                return X / q
    elif i == 2:
        cax = (1 - alpha) * ξ
        aux = ξ ** (r + 1)
        while True:
            U, V = np.random.rand(2)
            X = np.random.gamma(1 + cax, 1)
            s = sigma0(alpha, U) ** (1 / alpha) * X ** (-1 / r)
            if V <= X ** (-cax) * np.exp(-mass * s) * Cmin:
                return (θ * t) ** (1 / alpha) * s
    elif i == 3:
        ax = alpha * ξ
        aux = ξ ** (r + 1)
        sc = np.pi * np.sqrt((1 - alpha) * ax / 2)
        while True:
            U, V = np.random.rand(2)
            U = erfinv(U * erf(sc)) / sc
            X = np.random.gamma(ax, 1)
            ρ = sigma0(alpha, U) ** (r + 1)
            X1 = X ** (-r)
            if V <= ρ * X ** (-ax) * X1 * np.exp((1 - alpha) * ax * U ** 2 / 2 - ρ * aux * X1) * Cmin:
                return X / q
    else:  # i == 4
        cax = (1 - alpha) * ξ
        aux = ξ ** (r + 1)
        sc = np.pi * np.sqrt(alpha * cax / 2)
        while True:
            U, V = np.random.rand(2)
            U = erfinv(U * erf(sc)) / sc
            X = np.random.gamma(1 + cax, 1)
            s = sigma0(alpha, U) ** (1 / alpha) * X ** (-1 / r)
            if V <= X ** (-cax) * np.exp(alpha * cax * U ** 2 / 2 - mass * s) * Cmin:
                return (θ * t) ** (1 / alpha) * s

def rand_small_stable(alpha, θ, t, s):
    if s == np.inf:
        return rand_stable(alpha, θ, t)
    s1 = (θ * t / s ** alpha) ** (1 / (1 - alpha))
    aux0 = sigma(alpha, 0)
    f = lambda x: np.exp(s1 * (aux0 - sigma(alpha, x)))
    aux = np.log(4) / s1 + aux0
    a1 = 0.5
    while sigma(alpha, a1) > aux:
        a1 *= 0.5
    a2 = f(a1)
    U = 0
    V = np.zeros(3)

    if a1 == 0.5:
        a4 = 1 / (1 + a2)
        V = np.random.rand(3)
        U = a1 * V[0] if V[1] < a4 else a1 * (1 + V[0])
        while V[2] > f(U) / (1 if U < a1 else a2):
            V = np.random.rand(3)
            U = a1 * V[0] if V[1] < a4 else a1 * (1 + V[0])
    else:
        a3 = f(2 * a1)
        a4 = 1 / (1 + a2 + a3 / np.log(a2 / a3))
        V = np.random.rand(3)
        U = (
            a1 * V[0]
            if V[1] < a4
            else a1 * (1 + V[0])
            if V[1] < a4 * (1 + a2)
            else a1 * (2 + np.log(1 / V[0]) / np.log(a2 / a3))
        )
        while U >= 1 or V[2] > f(U) / (1 if U < a1 else (a2 if U < 2 * a1 else np.exp(((2 * a1 - U) * np.log(a2) + (U - a1) * np.log(a3)) / a1))):
            V = np.random.rand(3)
            U = (
                a1 * V[0]
                if V[1] < a4
                else a1 * (1 + V[0])
                if V[1] < a4 * (1 + a2)
                else a1 * (2 + np.log(1 / V[0]) / np.log(a2 / a3))
            )

    return (θ * t) ** (1 / alpha) * (expon.rvs() / sigma(alpha, U) + s1) ** (-(1 - alpha) / alpha)

def rand_small_tempered_stable(alpha, θ, q, t, s):
    if s == np.inf:
        return rand_tempered_stable(alpha, θ, q, t)
    elif 0.5 > np.exp(q ** alpha * θ * t - q * s):
        x = rand_tempered_stable(alpha, θ, q, t)
        while x > s:
            x = rand_tempered_stable(alpha, θ, q, t)
        return x
    else:
        x = rand_small_stable(alpha, θ, t, s)
        while x > -np.log(np.random.rand()) / q:
            x = rand_small_stable(alpha, θ, t, s)
        return x
    
def rand_undershoot_stable(alpha, θ, t, s):
    r = alpha / (1 - alpha)
    sc = (θ * t) ** (1 / alpha)
    s1 = s / sc
    s2 = s1 ** (-r)
    z = invsigma(alpha, alpha / s2)
    p = (2 - 2 ** alpha) ** (-alpha) * r ** (-alpha) * s1 ** (alpha * r) * int_exp_sigma(alpha, s2, 1) / (
        gamma(1 - alpha) * int_tilt_exp_sigma(alpha, s2, 1, z)
    )
    p = 1 / (1 + 1 / p)
    c1 = (1 - 2 ** (alpha - 1)) ** (-alpha) * s1 ** (-alpha)
    c2 = (2 * r) ** alpha * s2

    E = U = 0
    bool = True
    while bool:
        if np.random.rand() <= p:
            U = rand_exp_sigma(alpha, s2)
            E = expon.rvs() / sigma(alpha, U) + s2
        else:
            U = rand_tilt_exp_sigma(alpha, s2)
            E = np.random.gamma(1 - alpha, 1) / sigma(alpha, U) + s2
        bool = (
            np.random.rand()
            > abs(s1 - E ** (-1 / r)) ** (-alpha)
            / (c1 + c2 * (E > s2) * (E - s2) ** (-alpha))
        )
    return sc * E ** (-1 / r)

def rand_crossing_stable(alpha, θ, b, Db, B):
    S = rand_stable(alpha, θ, 1)
    T = B(S)
    w0 = -Db(T)
    if w0 != 0:
        w1 = b(T) / (alpha * T)
        if np.random.rand() <= w0 / (w0 + w1):
            return (T, b(T), 0)
    U = rand_undershoot_stable(alpha, θ, T, b(T))
    return (T, U, (b(T) - U) * np.random.rand() ** (-1 / alpha))

def rand_crossing_small_stable(alpha, θ, b, Db, B, T):
    S = rand_stable(alpha, θ, 1)
    aux = T ** (-1 / alpha) * b(T)
    while S < aux:
        S = rand_stable(alpha, θ, 1)
    t = B(S)
    w0 = -Db(t)
    if w0 != 0:
        w1 = b(t) / (alpha * t)
        if np.random.rand() <= w0 / (w0 + w1):
            return (t, b(t), 0)
    U = rand_undershoot_stable(alpha, θ, t, b(t))
    return (t, U, (b(t) - U) * np.random.rand() ** (-1 / alpha))

def rand_crossing_tempered_stable(alpha, θ, q, b, Db, BB):
    Tmax = (2 * q * b(0) + 1 - 2 ** (-alpha)) / ((2 ** alpha - 1) * q ** alpha * θ)
    Tf = Uf = 0
    S = rand_tempered_stable(alpha, θ, q, Tmax)
    while S < b(Tmax + Tf) - Uf:
        Tf += Tmax
        Uf += S
        S = rand_tempered_stable(alpha, θ, q, Tmax)
    locb = lambda x: b(x + Tf) - Uf
    locDb = lambda x: Db(x + Tf)
    locB = BB(Tf, Uf)

    (T, U, V) = rand_crossing_small_stable(alpha, θ, locb, locDb, locB, Tmax)
    S = rand_stable(alpha, θ, Tmax - T)
    while q * (S + U + V) > -np.log(np.random.rand()):
        (T, U, V) = rand_crossing_small_stable(alpha, θ, locb, locDb, locB, Tmax)
        S = rand_stable(alpha, θ, Tmax - T)
    return (Tf + T, Uf + U, V)

def rand_crossing_tempered_stable(alpha, θ, q, a0, a1, r):
    if q == 0:
        return rand_crossing_stable(alpha, θ, a0, a1, r)

    R0 = (2 ** alpha - 1) / (2 * q)
    Tf = Uf = V = 0
    locb = lambda t: 0
    locDb = lambda t: 0
    locB = lambda t: 0

    while Uf + V <= min(a0 - a1 * Tf, r):
        Uf += V
        R = R0 + Uf
        Tmax = (2 * q * min(a0, r, R - Uf) + 1 - 2 ** (-alpha)) / ((2 ** alpha - 1) * q ** alpha * θ)
        S = rand_tempered_stable(alpha, θ, q, Tmax)
        while S < min(a0 - a1 * (Tmax + Tf), r, R) - Uf:
            Tf += Tmax
            Uf += S
            S = rand_tempered_stable(alpha, θ, q, Tmax)

        locb, locDb, locB = crossing_functions(alpha, a0 - a1 * Tf, a1, min(r, R) - Uf)
        (T, U, V) = rand_crossing_small_stable(alpha, θ, locb, locDb, locB, Tmax)
        S = rand_stable(alpha, θ, Tmax - T)

        while q * (S + U + V) > locb(Tmax) - np.log(np.random.rand()):
            (T, U, V) = rand_crossing_small_stable(alpha, θ, locb, locDb, locB, Tmax)
            S = rand_stable(alpha, θ, Tmax - T)
        Tf += T
        Uf += U
    return (Tf, Uf, V)

def rand_crossing_subordinator(alpha, ϑ, q, a0, a1, r, ρ, mass, randmass):
    t = u = v = 0
    c = a0
    θ = ϑ * gamma(1 - alpha) / alpha
    D = np.random.exponential(scale=1 / mass)
    while c > 0:
        t1, u1, v1 = rand_crossing_tempered_stable(alpha, θ, q, c, a1, r * ρ)
        while v1 > r:
            t1, u1, v1 = rand_crossing_tempered_stable(alpha, θ, q, c, a1, r * ρ)
        if D > t1:
            t += t1
            u += v + u1
            v = v1
            c = a0 - v - u
            D -= t1
        else:
            w = rand_small_tempered_stable(alpha, θ, q, D, min(c - a1 * D, r * ρ))
            t += D
            u += v + w
            v = randmass()
            c = a0 - v - u
            D = np.random.exponential(scale=1 / mass)
    return t, u, v
