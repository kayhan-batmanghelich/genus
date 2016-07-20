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


