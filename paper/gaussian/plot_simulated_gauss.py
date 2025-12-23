import numpy as np
import pandas as pd
import scipy.linalg as la
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from coupling_bgs import funct_simulation
from coupling_bgs.utils import compute_bound, file_string


cd = os.path.join("paper", "gaussian")
plot_dir = os.path.join(cd, "plots")
output_dir = os.path.join(cd, "output")


def compute_condition_number(A):
    singular_values = la.svdvals(A)
    return singular_values[0] / singular_values[-1]


def plot_results(file_name, K,tau_e, tau_k, delta, eps, reg_num, vanilla = False, save= False, rho=1, output_name =[], log_scale=True):
    data = pd.read_csv(file_name)
    data = data[data['I'] < 5000] # remove I=5000, o/w too long runtime

    I = data['I'].unique()
    bound = np.zeros(len(I))
    
    for en,i in enumerate(I):
        print(f'######### {i} #########\n' )
        x = funct_simulation.asymptotic_regimes(reg_num,K,I=i)

        # these automatically takes care of eliminating 0
        #rho[en], rho_coll[e], Sigma[e], B[en], B_coll[en]= x.conv_rate(tau_e,tau_k)
        rho, rho_coll, Sigma, B, B_coll= x.conv_rate(tau_e,tau_k)
        kappa = compute_condition_number(Sigma)


        if vanilla:
            mask = (data['I']==i) & (data['var']=='fixed') & (data['coll']=='vanilla')
        else:
            mask = (data['I']==i) & (data['var']=='fixed') & (data['coll']=='collapsed')
            rho = rho_coll
        
        dist0 = np.mean(data.loc[mask, 'dist'])
        eps = np.mean(data.loc[mask, 'eps']) 
        # Compute theoretical bound
        bound[en]= compute_bound(dist0, K, kappa, eps, rho)

        print(f"mixing time: {1/(1-rho)}, log(kappa) = {np.log(kappa)}, log(dist0) = {dist0}, -log(eps) = {-np.log(eps)}, bound = {bound[en]}")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.tick_params(axis='both', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xticks(range(len(I)))
    ax.set_xticklabels(np.array(I)*K+1)
    ax.set_yticks([10, 30, 100, 300, 1000, 3000])
    ax.set_yticklabels([10, 30, 100, 300, 1000, 3000])
    ax.yaxis.set_minor_formatter(NullFormatter())

    for _, row in data[['coll', 'var']].drop_duplicates().iterrows():
        coll_type = row['coll']; var_type = row['var']

        if vanilla == False and coll_type == "vanilla":
            continue # skip vanilla if not vanilla

        avg_meeting_times = []
        for i in I:
            mask = (data['coll'] == coll_type) & (data['var'] == var_type) & (data['I'] == i)
            avg_meeting_times.append(1 + data.loc[mask, 't'].mean())

        label = f"{coll_type},  {var_type}"
        ax.plot(range(len(I)), avg_meeting_times, label= label)
        ax.scatter(range(len(I)), avg_meeting_times, s=50)


    ax.plot(range(len(I)), bound,'--',label='bound, fixed var, delta='+str(delta))
    ax.scatter(range(len(I)), bound,s= 200, marker = '*')



    ax.set_ylabel("Meeting time", fontsize=20)
    ax.set_xlabel("Parameters number", fontsize= 20)
    ax.legend(fontsize=15)

    # ax.set_title("Average meeting times",fontsize = 30)
    # ax.set_ylim(0.1)
    ax.grid(True, which="major")
    fig.tight_layout()
    if save:
        fig.savefig(str(output_name)+'.pdf', bbox_inches="tight")
    # plt.show()


delta= 0.5  # WHAT IS THIS??
eps=0.1
rand=True
save= True
log_scale= True

# Vanilla
collapsed = False
for reg_num, K in [(1,2), (2,2)]:
    tau_e = 1; tau_k = np.ones(K)

    fname = os.path.join(output_dir, f"{file_string(reg_num, K, collapsed)}.csv")
    plot_name = os.path.join(plot_dir, file_string(reg_num, K, collapsed) + ("_log_scale" if log_scale else ""))

    plot_results(fname, K,tau_e, tau_k, delta, eps, reg_num, vanilla=not collapsed, save=save, output_name = plot_name)

# Collapsed
collapsed = True
for reg_num, K in [(1,2), (1,3), (1,4), (2,2), (2,4)]:
# for reg_num, K in [(2,4)]:
    tau_e = 1; tau_k = np.ones(K)

    fname = os.path.join(output_dir, f"{file_string(reg_num, K, collapsed)}.csv")
    plot_name = os.path.join(plot_dir, file_string(reg_num, K, collapsed) + ("_log_scale" if log_scale else ""))

    plot_results(fname, K,tau_e, tau_k, delta, eps, reg_num, vanilla=not collapsed, save=save, output_name = plot_name)
