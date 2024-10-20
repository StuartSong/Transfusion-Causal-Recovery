#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:08:25 2023

@author: vinsonwei
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.special import logit, expit
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from scipy.linalg import expm
from scipy.spatial.distance import hamming
from tqdm import tqdm
import pickle

#define the DAG constraint
def expit_d(x):
    return expit(x)*(1-expit(x))

def g(x):
    return x

def find_top_numbers_indices(numbers):
    # Calculate the total sum of the numbers.
    total_sum = sum(numbers)
    # Sort the numbers in descending order.
    sorted_numbers = sorted(enumerate(numbers), key=lambda x: x[1], reverse=True)
    # Initialize variables for the cumulative sum and selected indices.
    cumulative_sum = 0
    selected_indices = []
    # Iterate through the sorted numbers and add them to the selected indices
    # until the cumulative sum exceeds 90% of the total sum.
    for index, value in sorted_numbers:
        cumulative_sum += value
        selected_indices.append(index)
        if cumulative_sum >= 0.9 * total_sum:
            break
    
    return selected_indices


def process_item(number, data):
    Y = np.array(data["Y"])
    T = np.array(data["T"])
    
    # Real Data
    w = 1
    d = np.shape(Y)[1]

    Xt = []
    Yt = []

    start = 0
    for length in T:
        Xt.append(Y[start:start+length-1,:])
        Yt.append(Y[start+1:start+length,:])
        start += length

    input_Xt = np.concatenate(Xt)
    input_Xt = np.insert(input_Xt, 0, 1, axis=1)  # Add a column of 1s
    input_Y = np.concatenate(Yt)

    T = np.shape(input_Xt)[0]

    # PGD - step 1: rough estimation
    ep_num = 6000
    lr = 0.005
    theta_whole = np.zeros([d, 1 + w * d])

    for ep_idx in range(ep_num):
        tmperr = g(input_Xt @ theta_whole.T) - input_Y
        tmp_GD = tmperr.T @ input_Xt / (T - w)
        print(tmp_GD[:,-1])
        tmp_GD_norm = np.linalg.norm(tmp_GD, axis=1)
        theta_whole -= lr * tmp_GD / tmp_GD_norm[:, np.newaxis]

        # projection
        tmp_idx_matrix = theta_whole < 0
        tmp_idx_matrix[:,-1] = False
        
        theta_whole[tmp_idx_matrix] = 0
        if ep_idx % 2000 == 0:
            lr /= 2 

    # max_row_number = find_top_numbers_indices(tmp_GD_norm)

    # # Additional steps
    # for ep_idx in range(ep_num):
    #     tmperr = g(input_Xt @ theta_whole.T) - input_Y
    #     tmp_GD = tmperr.T @ input_Xt / (T - w)
    #     tmp_GD_norm = np.linalg.norm(tmp_GD, axis=1)

    #     for row_number in max_row_number:
    #         theta_whole[row_number,:] -= lr * tmp_GD[row_number,:] / tmp_GD_norm[row_number]

    #     # # set last feature to be 0
    #     # theta_whole[:,-1] = 0
    #     theta_whole[:,-1] = 0
    #     if ep_idx % 2000 == 0:
    #         lr /= 2
            
    # Return a dictionary for the number
    return {
        number: {
            # "max_row_number": max_row_number,
            # "theta_whole_before": theta_whole[max_row_number,:],
            # "tmp_GD_before": tmp_GD[max_row_number,:],
            # "theta_whole_after": theta_whole[max_row_number,:],
            # "tmp_GD_after": tmp_GD[max_row_number,:],
            "A_fitted": theta_whole[:,1:],
            "mu_fitted": theta_whole[:,0]
        }
    }

def main():
    # Load your data
    with open("Transfusion_bootstraped30_20240213_transbinary.pkl", 'rb') as pickle_file:
        loaded_dict = pickle.load(pickle_file)

    final_dict = {}

    # Use ProcessPoolExecutor to parallelize the computation
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_item, number, data): number for number, data in loaded_dict.items()}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            number = futures[future]
            try:
                result = future.result()
                final_dict.update(result)
            except Exception as exc:
                print(f'{number} generated an exception: {exc}')

    # Save the final dictionary
    with open("transfusion_bootstraped30_DAG_20240213_trans_binary.pkl", 'wb') as pickle_file:
        pickle.dump(final_dict, pickle_file)

if __name__ == "__main__":
    main()
   