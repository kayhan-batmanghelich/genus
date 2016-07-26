import numpy as np

def demean(x):
    return x - x.mean(0)

np.random.seed(443)
A = demean(np.array([np.random.normal(size=4) for x in range(6)]).T)
B = demean(np.array([np.random.normal(size=4) for x in range(5)]).T)
A_row, A_col = A.shape
B_row, B_col = B.shape

C_ab = np.cov(A, B, rowvar=False)[:A_col,A_col:]
C_ba = C_ab.T
C_aa = np.cov(A, rowvar=False)
C_bb = np.cov(B, rowvar=False)
'''
from: 
Sparse Canonical Correlation Analysis
with Application to Genomic Data Integration
''' 
C_aa_diag = np.diag(np.diag(C_aa))
C_bb_diag = np.diag(np.diag(C_bb))
print(C_aa_diag.shape, C_bb_diag.shape)
K_c = np.dot(C_aa_diag, C_ab).dot(C_bb_diag)
u,s,v = np.linalg.svd(K_c, full_matrices=False)

lambda_u = .9
lambda_v = .8

def update(K, vec, lambda_val):
    def norm(x):
        return x / np.linalg.norm(x, ord=1)
    vec_sparse = np.dot(K, vec)
    vec_sparse = norm(vec_sparse)
    vec_sparse = np.abs(vec_sparse) - .5*lambda_val + np.sign(vec_sparse)
    vec_sparse = norm(vec_sparse)
    return vec_sparse

update(K_c.T, update(K_c, v, lambda_u), lambda_v)

