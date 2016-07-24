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
