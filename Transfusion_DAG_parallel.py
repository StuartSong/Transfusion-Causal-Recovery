#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script implements a parallelized process to estimate parameters using a bootstrapped dataset. It performs gradient descent optimization with projection to handle constraints in a dynamic DAG (Directed Acyclic Graph) setting.
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

# Define the expit function derivative used in gradient descent
def expit_d(x):
    return expit(x)*(1-expit(x))

# Define the g function (identity function used in gradient descent)
def g(x):
    return x

# Function to find the indices of the top numbers whose cumulative sum reaches 90% of the total sum.
def find_top_numbers_indices(numbers):
    # Calculate the total sum of the numbers.
    total_sum = sum(numbers)
    
    # Sort the numbers in descending order, keeping track of their original indices.
    sorted_numbers = sorted(enumerate(numbers), key=lambda x: x[1], reverse=True)
    
    # Initialize variables for cumulative sum and selected indices.
    cumulative_sum = 0
    selected_indices = []
    
    # Iterate through sorted numbers until the cumulative sum exceeds 90% of the total sum.
    for index, value in sorted_numbers:
        cumulative_sum += value
        selected_indices.append(index)
        if cumulative_sum >= 0.9 * total_sum:
            break
    
    return selected_indices

# Function to process each data item using gradient descent with projections
def process_item(number, data):
    # Extract data from dictionary for real data
    Y = np.array(data["Y"])
    T = np.array(data["T"])
    
    # Set parameters
    w = 1
    d = np.shape(Y)[1]

    Xt = []  # Input data sequence
    Yt = []  # Output data sequence
    
    # Split data into multiple time series using T (time lengths)
    start = 0
    for length in T:
        Xt.append(Y[start:start+length-1,:])  # Input is one step behind the output
        Yt.append(Y[start+1:start+length,:])  # Output is the next time step
        start += length

    # Concatenate sequences and insert a column of 1s for the intercept
    input_Xt = np.concatenate(Xt)
    input_Xt = np.insert(input_Xt, 0, 1, axis=1)  # Add bias term (column of 1s)
    input_Y = np.concatenate(Yt)

    # Get total number of observations
    T = np.shape(input_Xt)[0]

    # Initialize parameters for gradient descent
    ep_num = 6000  # Number of epochs
    lr = 0.005  # Learning rate
    theta_whole = np.zeros([d, 1 + w * d])  # Initialize theta (parameters) matrix

    # Perform projected gradient descent optimization
    for ep_idx in range(ep_num):
        tmperr = g(input_Xt @ theta_whole.T) - input_Y  # Calculate error
        tmp_GD = tmperr.T @ input_Xt / (T - w)  # Compute gradient
        print(tmp_GD[:,-1])  # Debugging: print the last column of gradient for monitoring
        tmp_GD_norm = np.linalg.norm(tmp_GD, axis=1)  # Compute norm of the gradient
        theta_whole -= lr * tmp_GD / tmp_GD_norm[:, np.newaxis]  # Update theta using normalized gradient

        # Apply projection to ensure non-negative constraints on theta (except bias term)
        tmp_idx_matrix = theta_whole < 0
        tmp_idx_matrix[:,-1] = False  # Do not apply projection to bias term
        theta_whole[tmp_idx_matrix] = 0  # Set negative values to 0 (projection)
        
        # Adjust learning rate after every 2000 epochs
        if ep_idx % 2000 == 0:
            lr /= 2 

    # Return the results for the current item
    return {
        number: {
            "A_fitted": theta_whole[:,1:],  # The estimated parameters (without bias)
            "mu_fitted": theta_whole[:,0]   # The bias/intercept term
        }
    }

def main():
    # Load the bootstrapped dataset from a pickle file
    with open("Transfusion_bootstraped30_20240213_transbinary.pkl", 'rb') as pickle_file:
        loaded_dict = pickle.load(pickle_file)

    # Initialize the final dictionary to store the results
    final_dict = {}

    # Use ProcessPoolExecutor to parallelize the processing of data items
    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor for parallel execution
        futures = {executor.submit(process_item, number, data): number for number, data in loaded_dict.items()}

        # Process the results as the tasks complete, updating the final dictionary
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            number = futures[future]
            try:
                result = future.result()
                final_dict.update(result)  # Store the result in final dictionary
            except Exception as exc:
                print(f'{number} generated an exception: {exc}')  # Handle exceptions

    # Save the final dictionary to a pickle file
    with open("transfusion_bootstraped30_DAG_20240213_trans_binary.pkl", 'wb') as pickle_file:
        pickle.dump(final_dict, pickle_file)

if __name__ == "__main__":
    main()  # Execute the main function
