import numpy as np
import pandas as pd
import os
import scipy.linalg as la

from coupling_bgs import funct_simulation as funct
import coupling_bgs.utils as utils

cd = os.path.join("paper", "insteval")
output_dir = os.path.join(cd, "output") 


def compute_condition_number(A):
    singular_values = la.svdvals(A)
    return singular_values[0] / singular_values[-1]

def run_insteval(data, reffs, tau, tau_e, num_runs=10, eps=0.1, collapsed=True, tmax=100):
    x = funct.Data()
    # N=1000; K=3; I=100; mu=3.; tau_k=1.; b=1.
    x.import_df(data, reffs)


    rho_pv, rho_coll, Sigma, _, _ = x.conv_rate(tau_e, tau)
    kappa = compute_condition_number(Sigma)

    times = []; distances=[]
    for _ in range(num_runs):
        _, _, t, d0 = funct.MCMC_sampler_coupled_realdataset(x, collapsed=collapsed, T=tmax, tau=tau, tau_e=tau_e, return_dist0=True)
        times.append(t); distances.append(d0)
    dist0 = np.mean(distances)
    bound_coll = utils.compute_bound(dist0, x.K, kappa, eps, rho_coll)
    bound_pv = utils.compute_bound(dist0, x.K, kappa, eps, rho_pv)
    return np.mean(times), bound_coll, bound_pv

data = pd.read_csv(os.path.join(cd, "insteval.csv"))
eps = .1

times_coll = []; times_pv = []; bounds_coll = []; bounds_pv = []

reffs = ['d', 's']
tau=[9.4149102, 3.6531701]; tau_e= 0.7208871
t_coll, bound_coll, bound_pv = run_insteval(data, reffs, tau, tau_e, collapsed=True)
t_pv, _, _ = run_insteval(data, reffs, tau, tau_e, collapsed=False)
times_coll.append(t_coll); times_pv.append(t_pv)
bounds_coll.append(bound_coll); bounds_pv.append(bound_pv)

reffs = ['s', 'd', 'lectage', 'studage']
tau = [9.3592072, 3.7295522, 128.4982743, 338.3219925]; tau_e = 0.7226101
t_coll, bound_coll, bound_pv = run_insteval(data, reffs, tau, tau_e, collapsed=True)
t_pv, _, _ = run_insteval(data, reffs, tau, tau_e, collapsed=False)
times_coll.append(t_coll); times_pv.append(t_pv)
bounds_coll.append(bound_coll); bounds_pv.append(bound_pv)


reffs = ['s', 'd', 'lectage', 'studage', 'service']
tau = [9.4079960, 3.7436722, 143.4601876, 392.8648024, 398.4144663]; tau_e = 0.7227555 
t_coll, bound_coll, bound_pv = run_insteval(data, reffs, tau, tau_e, collapsed=True)
t_pv, _, _ = run_insteval(data, reffs, tau, tau_e, collapsed=False)
times_coll.append(t_coll); times_pv.append(t_pv)
bounds_coll.append(bound_coll); bounds_pv.append(bound_pv)

# create df with columns times_coll, bounds_coll, times_pv, bounds_pv and Ks = [2,4,5]
results_df = pd.DataFrame({
    'K': [2, 4, 5],
    'times_collapsed': times_coll,
    'bounds_collapsed': bounds_coll,
    'times_vanilla': times_pv,
    'bounds_vanilla': bounds_pv
})

results_df.to_csv(os.path.join(cd, "insteval_results.csv"), index=False)