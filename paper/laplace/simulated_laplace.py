import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import coupling_bgs.funct_simulation_MH_11 as fmh
# import coupling_bgs.funct_simulation_MH as fmh2

cd = os.path.join("paper","laplace")
plot_dir = os.path.join(cd, "plots")
output_dir = os.path.join(cd, "output")

def run_one_step_mh_coupling(data, T_max=1000, L=1, S=1, fixed = False):
    K = data.K
    a = data.a
    b = data.b

    iterv = fmh.iter_value_laplace(data, T_max,a,0,  0, rand = False)
    iterv_2 = fmh.iter_value_laplace(data, T_max,a,0,  0, rand = False)
    mask = [np.array([data.ii[:,k] == i for i in range(data.iss[k])]) for k in range(K)]
        
    for l in range(L):
        iterv.update(data,l,mask,S=S, fixed=fixed)
    dist = iterv.distance(iterv_2)
    # if dist/(K*(i+1)) < 10:
    #     close = True
    # else:
    #     close = False
    for t in (range(T_max)):
        if iterv.coupled_update(iterv_2, data,t,L,mask,b,s, fixed=fixed, close = True, factorized_proposals = True, factorized_acceptance = True, a_pvb= True, a_pvf = True) == -1:
            break
    if t == T_max-1:
        print("Warning: max time reached")
    
    return t

I0 = np.array([40, 70, 90])
# I0 = np.array([10, 10, 10])
# I0 = np.array([50, 50, 50])
K = len(I0)
tau_k = np.ones(K)
L = 1

T_max = 1000
n_exp = 1
n_trials = 10
p0 = 0.1/K

# create a dataframe with columns 'I' a tuple, 'S' number of MH steps, 'var_fixed' boolean, 'time' meeting time
df = pd.DataFrame(columns=['I', 'S', 'var_fixed', 'time'])

for j in range(n_exp):
    iss = I0 * 2**j
    pi = p0 * iss.sum()**2 / iss.prod()
    N = np.random.binomial(iss.prod(), pi, 1)[0]
    data = fmh._asymptotic_regimes_lapl(N, iss, b=1, mu=0, tau_k=tau_k)

    for s in [3, 5]:
        times_fixed = [run_one_step_mh_coupling(data, T_max=T_max, L=L, S=s, fixed = True) for _ in range(n_trials)]
        times_free = [run_one_step_mh_coupling(data, T_max=T_max, L=L, S=s, fixed = False) for _ in range(n_trials)]
        df = pd.concat([df, pd.DataFrame({'I': [tuple(iss)]*n_trials, 'S': [s]*n_trials, 'var_fixed': [True]*n_trials, 'time': times_fixed})], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({'I': [tuple(iss)]*n_trials, 'S': [s]*n_trials, 'var_fixed': [False]*n_trials, 'time': times_free})], ignore_index=True)
        
        print(f'Finished exp {j+1} with I={iss}, S={s}. Avg time fixed: {np.mean(times_fixed)}, free: {np.mean(times_free)}')

df.to_csv(os.path.join(output_dir, 'results_laplace.csv'), index=False)
