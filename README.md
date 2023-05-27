# FirstPassageEvent_py
This package provides a fast simulation algorithm for the first passage event of a subordinator, consisiting of a truncated tempered stable subordinator and a compound Poisson process, across a target function. 


This package includes many auxiliary functions, including the following functions. (The notation used below follows that of the original article published by the same authors of this package.)

* The following function produces a sample of $S_t$ under $\mathbb{P}_0$ (i.e. a stable increment or marginal):
```python
rand_crossing_subordinator(alpha, ϑ, q, a0, a1, r, ρ, mass, randmass):
```
The subordinator has Levy density $I \lbrace 0 \leq x\leq r \rbrace e^{-qx}x^{-\alpha-1}dx+\lambda(dx)$, and mass $=\lambda(0,\infty)$, randmass is the function that generate the jump size of the compound Poisson component. The boundary is fucntion $c(t)=a_0-a_1*t$.
The output $(T,U,V)$ contains the first passage time $T$, the undershoot $U$ and the overshoot $V$.
