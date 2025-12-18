import numpy as np
import os

from coupling_bgs import funct_simulation as funct
import coupling_bgs.utils as utils

cd = os.path.join("paper", "gaussian")
output_dir = os.path.join(cd, "output") 

I= np.array([50,100, 200, 500, 1000])
num_runs = 100
eps = .1
export_results = True
rand = True


# sparse, K= 2, vanilla
K= 2
reg_num = 2
collapsed = [False, False]
variance = [False, True]
tau_e = 1
tau_k = np.ones(K)
T_max = 1000 
filename=os.path.join(output_dir, f"{utils.file_string(reg_num, K, collapsed[0])}.csv")
delta = 0.1
funct.run_experiment(K,I, num_runs, tau_e, tau_k, eps, reg_num, 
                    rand = rand,export_results = export_results,
                    T_max = T_max, filename = filename, 
                    collapsed = collapsed , variance = variance)

# sparse, K= 2, collapsed
K= 2
reg_num = 2
collapsed = [True, True]
variance = [False, True]
tau_e = 1
tau_k = np.ones(K)
T_max = 1000 
filename=os.path.join(output_dir, f"{utils.file_string(reg_num, K, collapsed[0])}.csv")
delta = 0.1
funct.run_experiment(K,I, num_runs, tau_e, tau_k, eps, reg_num, 
                    rand = rand,export_results = export_results,
                    T_max = T_max, filename = filename, 
                    collapsed = collapsed , variance = variance)

# dense, K= 2, vanilla
K= 2
reg_num = 1
collapsed = [False, False]
variance = [False, True]
tau_e = 1
tau_k = np.ones(K)
T_max = 1000 
filename=os.path.join(output_dir, f"{utils.file_string(reg_num, K, collapsed[0])}.csv")
delta = 0.1
funct.run_experiment(K,I, num_runs, tau_e, tau_k, eps, reg_num, 
                    rand = rand,export_results = export_results,
                    T_max = T_max, filename = filename, 
                    collapsed = collapsed , variance = variance)

# dense, K= 2, collapsed
K= 2
reg_num = 1
collapsed = [True, True]
variance = [False, True]
tau_e = 1
tau_k = np.ones(K)
T_max = 1000 
filename=os.path.join(output_dir, f"{utils.file_string(reg_num, K, collapsed[0])}.csv")
delta = 0.1
funct.run_experiment(K,I, num_runs, tau_e, tau_k, eps, reg_num, 
                    rand = rand,export_results = export_results,
                    T_max = T_max, filename = filename, 
                    collapsed = collapsed , variance = variance)


# dense, K= 3, collapsed
K = 3
reg_num = 1
collapsed = [True, True]
variance = [False, True]
tau_e = 1
tau_k = np.ones(K)
T_max = 1000 
filename=os.path.join(output_dir, f"{utils.file_string(reg_num, K, collapsed[0])}.csv")
delta = 0.1
funct.run_experiment(K,I, num_runs, tau_e, tau_k, eps, reg_num, 
                    rand = rand,export_results = export_results,
                    T_max = T_max, filename = filename, 
                    collapsed = collapsed , variance = variance)


# dense, K= 4, collapsed
K = 4
reg_num = 1
collapsed = [True, True]
variance = [False, True]
tau_e = 1
tau_k = np.ones(K)
T_max = 1000 
filename=os.path.join(output_dir, f"{utils.file_string(reg_num, K, collapsed[0])}.csv")
delta = 0.1
funct.run_experiment(K,I, num_runs, tau_e, tau_k, eps, reg_num, 
                    rand = rand,export_results = export_results,
                    T_max = T_max, filename = filename, 
                    collapsed = collapsed , variance = variance)