import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg as la
import os

from coupling_bgs import funct_simulation
from coupling_bgs.utils import compute_bound, file_string


cd = os.path.join("paper", "gaussian")
plot_dir = os.path.join(cd, "plots")
output_dir = os.path.join(cd, "output")


def compute_condition_number(A):
    singular_values = la.svdvals(A)
    return singular_values[0] / singular_values[-1]


def plot_results(file_name, K,tau_e, tau_k, delta, eps, reg_num, vanilla = False, save= False, rho=1, output_name =[]):
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
    plt.figure(figsize=(10,5))
    # font = {'fontname':'DejaVu Sans'}
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)    # fontsize of the tick labels

    plt.xticks(range(len(I)), np.array(I)*K+1)
    for _, row in data[['coll', 'var']].drop_duplicates().iterrows():
        coll_type = row['coll']; var_type = row['var']

        if vanilla == False and coll_type == "vanilla":
            continue # skip vanilla if not vanilla

        avg_meeting_times = []
        for i in I:
            mask = (data['coll'] == coll_type) & (data['var'] == var_type) & (data['I'] == i)
            avg_meeting_times.append(1 + data.loc[mask, 't'].mean())

        label = f"{coll_type},  {var_type}"
        plt.plot(range(len(I)), avg_meeting_times, label= label)
        plt.scatter(range(len(I)), avg_meeting_times, s=50)


    plt.plot(range(len(I)), bound,'--',label='bound, fixed var, delta='+str(delta))
    plt.scatter(range(len(I)), bound,s= 200, marker = '*')


    plt.ylabel("Meeting time", fontsize=20)
    plt.xlabel("Parameters number", fontsize= 20)
    plt.legend(fontsize=15)

    plt.title("Average meeting times",fontsize = 30)
    plt.ylim(0)
    plt.grid(True, which="both")
    if save:
        plt.savefig(str(output_name)+'.pdf', bbox_inches="tight")
    # plt.show()


delta= 0.5  # WHAT IS THIS??
eps=0.1
rand=True
save= True

# Vanilla
collapsed = False
for reg_num, K in [(1,2), (2,2)]:
    tau_e = 1; tau_k = np.ones(K)

    fname = os.path.join(output_dir, f"{file_string(reg_num, K, collapsed)}.csv")
    plot_name = os.path.join(plot_dir, file_string(reg_num, K, collapsed))

    plot_results(fname, K,tau_e, tau_k, delta, eps, reg_num, vanilla=not collapsed, save=save, output_name = plot_name)

# Collapsed
collapsed = True
for reg_num, K in [(1,2), (1,3), (1,4), (2,2)]:
# for reg_num, K in [(1,2)]:
    tau_e = 1; tau_k = np.ones(K)

    fname = os.path.join(output_dir, f"{file_string(reg_num, K, collapsed)}.csv")
    plot_name = os.path.join(plot_dir, file_string(reg_num, K, collapsed))

    plot_results(fname, K,tau_e, tau_k, delta, eps, reg_num, vanilla=not collapsed, save=save, output_name = plot_name)
