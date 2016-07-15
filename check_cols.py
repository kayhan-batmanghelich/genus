import numpy as np
import scipy.io
import pandas as pd

# find linearly dependent columns and remove them
data = scipy.io.loadmat('scz_eur_pgc_annealed.mat')
I = data['I']

def lin_dep(data):
    R = linalg.qr(data)[1]
    ind_cols = np.where(np.abs(R.diagonal()) > 0)[0]
    return data[:, ind_cols]

I = lin_dep(I)

# check that you actually dropped the linearly dependent columns
def val_lin_dep(data):
    if  np.linalg.matrix_rank(data):
        print("colinearity removed")
    else:
        print("matrix is not linearly independent")

val_lin_dep(I)

I = I - I.mean(0) # mean center

I = (I - I.min(0)) / I.ptp(0)  # apply normalization

# create the new mat file
G = data['G']
y = data['y']
scipy.io.savemat('scz_eur_pgc_annealed_centered_normed.mat',
                  mdict={'G':G, 'I':I, 'y':y})
