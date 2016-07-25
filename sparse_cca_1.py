U = np.array([np.random.normal(size=100) for x in range(500)]).T
W = np.array([np.random.normal(size=100) for x in range(700)]).T
U = lin_dep(U)[0]
W = lin_dep(W)[0]
U_row, U_col = U.shape
W_row, W_col = W.shape
C_uu = np.cov(U, rowvar=False)
C_ww = np.cov(W, rowvar=False)
C_uw = np.cov(U,W, rowvar=False)[:U_col,U_col:]

def lin_dep(data):
    R = np.linalg.qr(data)[1]
    ind_cols = np.where(np.abs(R.diagonal()) > 0)[0]
    cols = np.array([x for x in range(data.shape[1])])
    dep_cols = sorted(list(set(cols) - set(ind_cols)))
    return data[:, ind_cols], dep_cols

Ux = np.linalg.cholesky(C_uu)
Uy = np.linalg.cholesky(C_ww)

K_Ux = np.linalg.inv(Ux) 

'''
(K'*Cov_uw)'(K'*Cov_uw)
'''
t1 = np.dot(np.dot(K_Ux.T, C_uw).T, np.dot(K_Ux.T, C_uw))
'''
(U_y'*U_y)
'''
t2 = np.dot(Uy.T, Uy)
p_squared, y = linalg.eig(t1,t2)
concoefs = np.sqrt(p_squared)
