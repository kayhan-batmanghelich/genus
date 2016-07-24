'''
sparse canonical correlation analysis
'''
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

X = np.array([np.random.normal(size=100) for x in range(500)]).T
Y = np.array([np.random.normal(size=100) for x in range(700)]).T

X_row, X_col = X.shape
Y_row, Y_col = Y.shape
'''
np.cov(X,Y, rowvar=False) will return a partioned matrix:

XX|XY
-----
YY|YX

'''
C_xx = np.cov(X, rowvar=False)
C_yy = np.cov(Y, rowvar=False)
C_xy = np.cov(X,Y, rowvar=False)[:X_col,X_col:]

xI = np.eye(X_col) 
yI = np.eye(Y_col) 
'''
∥∥K−duvt∥∥2F + Ψ(u) + Φ(v) s.t. ∥u∥22=1, ∥v∥22=1 (4)

'''
K = np.dot(np.dot(xI, C_xy), yI)
r = np.linalg.matrix_rank(K)
u, d, v = linalg.svd(np.dot(K.T, K))
u, d, v = u[:r], d[:r], v[:r]
duvT = np.dot(np.dot(d,u), v.T)
K_minus_duvT = np.subtract(K[:r].T, duvT)
n_K = np.linalg.norm(K_minus_duvT, 'fro')

'''
optimization algorithm:


  1. select lamda_u, lamda_v

  2. select intial valeus u_^{0}, v_^{0}, set i  = 0

  3. (a) u_^{i+1} <-- K*v_i
     (b) u_^{i+1} <-- u_i+1/norm(u_i+1)
     (c) u_j_^{i+1} <-- (abs(u_j_^{i+1}) - 1/2 * lamda_u) + sign(u_j_^{i+1}), for j = 1,...P
     (d) repeat b
  
  4. do all of step 3, replace u for v

  5. i <-- i + 1

  6. repeat steps 3 and 4 until convergence:
'''

def scav(u, v, K, r, lambda_u, lambda_v):
    K = K[:r]
    iters = range(len(u))
    def soft_thresh(u, lambda_u):
        u = (abs(u) - 1/2 * lambda_u) + np.sign(u)
        return u
    def update(first,u,v,K, lambda_):
        if first == 'u':
            pass
        elif first == 'v':
            K = K.T
        u = K * v
        u = np.linalg.norm(u)
        u = soft_thresh(u, lambda_)
        u = np.linalg.norm(u)
        return u
    if len(u) != len(v):
        return "u and v are not of the same length"
    else:
        for i in iters:
            update(first='u',u, v, K, lambda_)
            update(first='v',u,v,K, lambda_)
