import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg as la
from scipy.special import erfinv
from coupling_bgs import funct_simulation
from coupling_bgs.utils import compute_bound


cd = "paper/gaussian/"
plot_dir = cd + "plots/"
output_dir = cd + "output/"


def compute_condition_number(A):
    singular_values = la.svdvals(A)
    return singular_values[0] / singular_values[-1]


def plot_results(file_name, K,tau_e, tau_k, delta, eps, reg_num, vanilla = False, save= False, rho=1, output_name =[]):
    data = pd.read_csv(file_name)
    data = data[data['I'] < 5000] # remove I=5000, o/w too long runtime

    I = data['I'].unique()
    bound = np.zeros(len(I))
    bound_vanilla = np.zeros(len(I))
    
    for en,i in enumerate(I):
        print(f'######### {i} #########\n' )
        x = funct_simulation.asymptotic_regimes(reg_num,K,I=i)

        # these automatically takes care of eliminating 0
        #rho[en], rho_coll[e], Sigma[e], B[en], B_coll[en]= x.conv_rate(tau_e,tau_k)
        rho, rho_coll, Sigma, B, B_coll= x.conv_rate(tau_e,tau_k)
        kappa = compute_condition_number(Sigma)
        mask = (data['I']==i) & (data['var']=='fixed') & (data['coll']=='collapsed')
        dist0 = np.mean(data.loc[mask, 'dist'])
        

        print(f"mixing time coll: {1/(1-rho_coll)}, log(kappa) = {np.log(kappa)}, log(dist0) = {np.log(dist0)}, -log(eps) = {-np.log(eps)}, bound = {compute_bound(dist0, K, kappa, eps, rho_coll)}")
        # Compute theoretical bound
        bound[en]= compute_bound(dist0, K, kappa, eps, rho_coll)

        if vanilla:
            # print(f"mixing time vanilla: {1/(1-rho)}. log(kappa) = {np.log(kappa)}, log(dist0) = {np.log(dist0)}, -log(eps) = {-np.log(eps)}, bound = {compute_bound(dist0, K, kappa, eps, rho)}")
            mask = (data['I']==i) & (data['var']=='fixed') & (data['coll']=='vanilla')
            dist0 = np.mean(data.loc[mask, 'dist'])
            bound_vanilla[en] = compute_bound(dist0, K, kappa, eps, rho)
    
    print(f"Vanilla: {vanilla}")
    plt.figure(figsize=(12,6))
    # font = {'fontname':'DejaVu Sans'}
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)    # fontsize of the tick labels

    plt.xticks(range(len(I)), np.array(I)*K+1)
    if not vanilla:
        plt.plot(range(len(I)), bound,'--',label='bound, fixed var, delta='+str(delta))
        plt.scatter(range(len(I)), bound,s= 200, marker = '*')
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


    if vanilla:
        plt.plot(range(len(I)), bound_vanilla,'--',label='bound, fixed var, delta='+str(delta))
        plt.scatter(range(len(I)), bound_vanilla,s= 200, marker = '*')


    plt.ylabel("Meeting time", fontsize=20)
    plt.xlabel("Parameters number", fontsize= 20)
    plt.legend(fontsize=15)

    plt.title("Average meeting times",fontsize = 30)
    plt.ylim(0, 30)
    plt.grid(True, which="both")
    if save:
        plt.savefig(str(output_name)+'.pdf', bbox_inches="tight")
    plt.show()


delta= 0.5  # WHAT IS THIS??
eps=0.1
rand=True
save= True

# Vanilla
vanilla = True
for reg_num, K in [(1,2), (2,2)]:
    tau_e = 1; tau_k = np.ones(K)

    fname = f"{output_dir}reg{reg_num}_k{K}_vanilla.csv"
    plot_name = f"{plot_dir}reg{reg_num}_k{K}" + ("_vanilla" if vanilla else "_coll")

    plot_results(fname, K,tau_e, tau_k, delta, eps, reg_num,rand,vanilla, save, output_name = plot_name)


# Collapsed
vanilla = False
for reg_num, K in [(1,2), (1,3), (1,4), (2,2)]:
# for reg_num, K in [(1,2)]:
    tau_e = 1; tau_k = np.ones(K)

    fname = f"{output_dir}reg{reg_num}_k{K}.csv"
    plot_name = f"{plot_dir}reg{reg_num}_k{K}" + ("_vanilla" if vanilla else "_coll")

    plot_results(fname, K,tau_e, tau_k, delta, eps, reg_num, vanilla=vanilla, save=save, output_name = plot_name)

