import numpy as np
import time
from tqdm import tqdm
import pickle
from multiprocessing import Pool
from functools import partial

import DAG_lib as dag



    
with open("Transfusion Training data_20240311.pkl", 'rb') as pickle_file:
    data = pickle.load(pickle_file)
w,d,input_Xt,input_Y,T = dag.load_data(data)

ep_num=5000
lr=0.005

# PGD - step 1: rough estimation
# projected gradient descent no penality 0 intialization
# project everything to be non-negative
print("step 1 starting")
theta_whole, _ = dag.pgd(d, w, T, input_Xt, input_Y, ep_num=ep_num, lr=lr,decay_ep=3000)
A_fitted = theta_whole[:,1:]
mu_fitted =  theta_whole[:,0]
print("step 1 finished")
    
step2_dict = dag.cycle_find(A_fitted,w,d)
A_constraint = step2_dict["A_constraint"]
delta_lst = step2_dict["delta_lst"]

dag_step_1 = {}
dag_step_1["A_fitted"] = A_fitted
dag_step_1["mu_fitted"] = mu_fitted
with open("transfusion_training_data_finetune_lambda_DAG_20240327.pkl", 'wb') as pickle_file:
    pickle.dump(dag_step_1, pickle_file)

# step 3: DAG estimation
print("step3 started")

def run_pgd_with_lambda(my_lambda):
    theta_whole_linear, _ = dag.pgd(d, w, T, input_Xt, input_Y, ep_num=5000, lr=0.5, decay_ep=500,
                                    decay_num=0.9,my_penalty=dag.p_linear_circle,my_lambda=my_lambda,
                                    A_constraint=A_constraint, delta_lst=delta_lst)
    
    filename = f"fine_tune_lambda_2/theta_whole_linear_lambda_{my_lambda}.npy"
    np.save(filename, theta_whole_linear)
    
    return theta_whole  # or return any other relevant result


# # _, theta_whole = dag.pgd(d, w, T, input_Xt, input_Y, ep_num=ep_num, theta_whole_linear_previous = theta_whole,
# #                          lr=lr, decay_ep=decay_ep, my_proj=dag.proj_negative_treat_effect,
# #                          my_penalty=dag.p_linear_circle, my_lambda=0.1, A_constraint=A_constraint, delta_lst=delta_lst)


my_lambda_values = 10**np.linspace(-2, -6, num=501)  # Example list of lambda values to try
with Pool(processes=30) as pool:  # Adjust the number of processes as needed
    results = list(tqdm(pool.imap_unordered(run_pgd_with_lambda, my_lambda_values), total=len(my_lambda_values)))