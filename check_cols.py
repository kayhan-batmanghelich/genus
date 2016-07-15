import numpy as np
import scipy.io

# find linearly dependent columns and remove them
data = scipy.io.loadmat('scz_eur_pgc_annealed.mat')
I = data['I']

def lin_dep(data):
    R = linalg.qr(data)[1]
    ind_cols = np.where(np.abs(R.diagonal()) > 0)[0]
    cols = np.array([x for x in data.shape[1]])
    dep_cols = sorted(list(set(cols) - set(ind_cols)))
    return data[:, ind_cols], dep_cols
    
I = lin_dep(I)[0]

I = I - I.mean(0) # mean center

I = (I - I.min(0)) / I.ptp(0)  # apply normalization

# create the new mat file
G = data['G']
y = data['y']
scipy.io.savemat('scz_eur_pgc_annealed_centered_normed.mat',
                  mdict={'G':G, 'I':I, 'y':y})
