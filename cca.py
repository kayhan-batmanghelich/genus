import numpy as np
from scipy import linalg

'''
references: 

  Rencher, C.A., Methods of Multivariate Analysis,
  John Wiley & Sons, Inc, 2002.

  Lin, D., et. al., Group sparse canonical correlation analysis for genomic data integration,
  BMC Bioinformatic, 2013.
'''


def cca(X,Y,type=None):
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
        Y_col = Y.shape[1]
        return [(X_row, X_col), Y_col]
        
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

C_xx = np.cov(X, rowvar=False)**(-1/2)
C_yy = np.cov(Y, rowvar=False)**(-1/2)
C_xy = np.cov(X,Y, rowvar=False)[:1000,1000:]
K_tmp = np.dot(C_xx,  C_xy)
K = np.dot(K_tmp, C_yy) # equation 2
K_trans_K = np.dot(K.T, K)
eighval, eighvec = linalg.eigh(K_trans_K)
d = np.sqrt(eighval)
