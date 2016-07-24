import numpy as np
from scipy import linalg

'''
references: 

  Rencher, C.A., Methods of Multivariate Analysis,
  John Wiley & Sons, Inc, 2002.

  Lin, D., et. al., Group sparse canonical correlation analysis for genomic data integration,
  BMC Bioinformatic, 2013.
'''


def cca(X,Y,ccatype=None):
    '''
    work in progress
    vanilla canonical correlation analysis
    '''
    def lin_dep(data):
        R = np.linalg.qr(data)[1]
        ind_cols = np.where(np.abs(R.diagonal()) > 0)[0]
        cols = np.array([x for x in range(data.shape[1])])
        dep_cols = sorted(list(set(cols) - set(ind_cols)))
        return data[:, ind_cols], dep_cols
    
    def val_args(X, Y):
        X_row, X_col = X.shape
        Y_row, Y_col = Y.shape
        return [(X_row, X_col), (Y_row, Y_col)]
        
    def demean(X, Y):
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
        return X, Y
    
    def decomp(X, Y):
        q_x, r_x, p_x = linalg.qr(X, mode='economic', pivoting=True)
        q_y, r_y, p_y = linalg.qr(Y, mode='economic', pivoting=True)
        return [(q_x, r_x, p_x), (q_y, r_y, p_y)]
    
    def rank(X, Y):
        rank_x = np.linalg.matrix_rank(r_x)
        rank_y = np.linalg.matrix_rank(r_y)
        return rank_x, rank_y
        
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
