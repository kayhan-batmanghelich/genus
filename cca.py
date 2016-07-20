import numpy as np

def lin_dep(data):
    '''
    ind_cols: linearly independent columns (features)
    dep_cols: linearly dependent columns
    '''
    R = np.linalg.qr(data)[1]
    ind_cols = np.where(np.abs(R.diagonal()) > 0)[0]
    cols = np.array([x for x in range(data.shape[1])])
    dep_cols = sorted(list(set(cols) - set(ind_cols)))
    return data[:, ind_cols], dep_cols
    

X = np.array([np.random.normal(size=3) for x in range(60)]).T
Y = np.array([np.random.normal(size=3) for x in range(60)]).T


'''
rowvar: default - True
True: row are variables, columns are observations
False: rows are observations, columns are variables


This does not work as you cannot multiply:
covarx**(-1/2) * covarxy * covary**(-1/2)
'''
covarx = np.cov(X, rowvar=False)
covary = np.cov(Y, rowvar=False)
covarxy = np.cov(X,Y, rowvar=False)


'''
vanilla cca
'''

def demean(X,Y):
    X_ = X - X.mean(0)
    Y_ = Y - Y.mean(0)
    return X_, Y_

class cca(object):
    
    def val_args(self, X, Y):
        X_row, X_col = X.shape
        Y_col = Y.shape[1]
        
    def demean(self, X, Y):
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
        return X, Y
    
    def rank(self, X, Y):
        q_x, r_x, p_x = linalg.qr(X,mode='economic', pivoting=True)
        q_y, r_y, p_y = linalg.qr(Y, mode='economic', pivoting=True)
        rank_x = np.linalg.matrix_rank(r_x)
        rank_y = np.linalg.matrix_rank(r_y)
        
