import numpy as np
from numpy.linalg import norm
from scipy.linalg import sqrtm


def svd(train, k):
    util_mat = np.array(train)
    mask = np.isnan(util_mat)
    masked_arr = np.ma.masked_array(util_mat, mask)
    item_means = np.mean(masked_arr, axis=0)
    util_mat = masked_arr.filled(item_means)
    x = np.tile(item_means, (util_mat.shape[0],1))
    util_mat = util_mat - x
    U, s, V =np.linalg.svd(util_mat, full_matrices=False)
    s = np.diag(s)
    # s[:][s < 30] = 0
    # print(s)
    # k = np.count_nonzero(s)
    # print(k)
    s = s[0:k, 0:k]
    U = U[:, 0:k]
    V = V[0:k, :]
    s_root=sqrtm(s)
    Usk = np.dot(U, s_root)
    skV = np.dot(s_root, V)
    UsV = np.dot(Usk, skV)
    UsV = UsV + x
    return UsV
