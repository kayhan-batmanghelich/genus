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
    
