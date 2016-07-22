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
X = np.array([np.random.normal(size=100) for x in range(1000)]).T
Y = np.array([np.random.normal(size=100) for x in range(2000)]).T

X_row, X_col = X.shape
Y_row, Y_col = Y.shape

C_xx = np.cov(X, rowvar=False)**(-1/2)
C_yy = np.cov(Y, rowvar=False)**(-1/2)
C_xy = np.cov(X,Y, rowvar=False)[:X_col,X_col:]

'''
np.cov(X,Y, rowvar=False) will return a partioned matrix:

XX|XY
-----
YY|YX

'''
# a check of the above
print(np.allclose(np.cov(X, rowvar=False), 
                  np.cov(X,Y, rowvar=False)[:X_col,:X_col]))

# using the covariance matrices
K_tmp = np.dot(C_xx,  C_xy)
K = np.dot(K_tmp, C_yy) # equation 2
u, d, v = linalg.svd(K) # equation 2
alpha = np.dot(C_xx, u) # equation 3
beta = np.dot(C_yy, v) # equation 3
r = np.linalg.matrix_rank(np.dot(X.T, Y))

# Lin, et. al. suggests replacing covariance matrices with identity matrices
xI = np.eye(X_col)
yI = np.eye(Y_col)
kI_tmp = np.dot(xI, C_xy)
kI = np.dot(kI_tmp, yI)
uI, dI, vI = linalg.svd(kI)
alphaI = np.dot(xI, uI)
betaI = np.dot(yI, vI)
