import numpy as np

def compute_bound(d0, K, kappa, eps, rho):
    d0 = np.asarray(d0)
    K = np.asarray(K)
    kappa = np.asarray(kappa)
    eps = np.asarray(eps)
    rho = np.asarray(rho)
    return 1 + (np.log(d0) + np.log(K) + 2.5 * np.log(kappa) - 2 * np.log(eps)) / -np.log(rho)


def file_string(reg_num: int, K: int, collapsed: bool) -> str:
    coll_str = "collapsed" if collapsed else "vanilla"
    
    return f"reg{reg_num}_K{K}_{coll_str}"