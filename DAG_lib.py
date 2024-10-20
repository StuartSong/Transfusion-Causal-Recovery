import itertools
import time
import numpy as np
from scipy.linalg import expm

# Function to apply a transformation to the input (currently just identity)
def g(x):
    return x

# Load the dataset and segment it into subarrays
def load_data(data):
    Y = np.array(data["Y"])  # Response variables
    T_seg = np.array(data["T"])  # Time segments

    print("Data loaded")

    w = 1  # Window size
    d = np.shape(Y)[1]  # Number of features

    Xt, Yt = [], []

    # Create input-output pairs based on time segments
    start = 0
    for length in T_seg:
        Xt.append(Y[start:start+length-1,:])  # Input is the segment minus the last step
        Yt.append(Y[start+1:start+length,:])  # Output is the segment minus the first step
        start += length

    # Concatenate inputs and add a bias term (column of 1s)
    input_Xt = np.concatenate(Xt)
    input_Xt = np.insert(input_Xt, 0, 1, axis=1)  # Add bias term (1s)
    input_Y = np.concatenate(Yt)  # Concatenate outputs

    T = np.shape(input_Xt)[0]  # Total number of time steps

    return w, d, input_Xt, input_Y, T

# Projection function to enforce non-negativity and constraints
def proj(theta_whole_linear, d):
    theta_whole_linear[theta_whole_linear < 0] = 0  # Set negative values to 0
    np.fill_diagonal(theta_whole_linear[:, 1:], 0)  # Force diagonals to remain 0
    return theta_whole_linear

# Projection function excluding the last column
def proj_negative_treat_effect(theta_whole_linear, d):
    tmp_idx_matrix = theta_whole_linear < 0
    tmp_idx_matrix[:, -1] = False  # Exclude last column from projection
    np.fill_diagonal(theta_whole_linear[:, 1:], 0)
    theta_whole_linear[tmp_idx_matrix] = 0
    return theta_whole_linear

# Projected Gradient Descent (PGD) optimization
def pgd(d, w, T, input_Xt, input_Y, ep_num=6000, theta_whole_linear_previous=None, 
        lr=0.005, decay_ep=2000, decay_num=0.9, my_proj=proj, my_penalty=None, my_lambda=0.1, A_constraint=None, delta_lst=None):
    
    theta_whole_linear = np.zeros([d, 1 + w * d]) if theta_whole_linear_previous is None else theta_whole_linear_previous.copy()
    tmperr_norm = []  # To store error norms

    for ep_idx in range(ep_num):
        # Calculate error
        tmperr = g(input_Xt @ theta_whole_linear.T) - input_Y
        tmperr_norm.append(np.linalg.norm(tmperr))

        # Compute gradient and apply penalty if necessary
        tmp_GD = tmperr.T @ input_Xt / (T - w)
        if my_penalty:
            tmp_GD += my_penalty(theta_whole_linear, my_lambda, A_constraint, delta_lst)
        tmp_GD_norm = np.linalg.norm(tmp_GD)

        # Update the model parameters
        theta_whole_linear -= lr * tmp_GD / tmp_GD_norm
        theta_whole_linear = my_proj(theta_whole_linear, d)

        # Decay learning rate
        if ep_idx % decay_ep == 0:
            lr *= decay_num

    return theta_whole_linear, tmperr_norm

# Find cycles in the graph represented by A_after_thres
def cycle_find(A_after_thres, w, d):
    interested_walk_len = 4  # Maximum cycle length to search
    all_closed_walk_num = {walk_len: 0 for walk_len in range(2, interested_walk_len + 1)}
    all_closed_walk_weights, all_closed_walk_edge, all_closed_walk_edge_weight = {}, {}, {}

    for walk_len in range(2, interested_walk_len + 1):
        tmp_permutations = itertools.permutations(range(d), walk_len)

        all_closed_walk_weights[walk_len], all_closed_walk_edge[walk_len], all_closed_walk_edge_weight[walk_len] = [], [], []

        for node_lst in tmp_permutations:
            node_lst = list(node_lst)
            new_node_lst = node_lst + [node_lst[0]]  # Complete the cycle
            all_edge_weight = [A_after_thres[new_node_lst[i + 1], new_node_lst[i]] for i in range(walk_len)]
            
            # If all edges in the cycle exist
            if all(all_edge_weight):
                all_closed_walk_num[walk_len] += 1
                all_closed_walk_weights[walk_len].append(sum(all_edge_weight))
                all_closed_walk_edge[walk_len].append(new_node_lst)
                all_closed_walk_edge_weight[walk_len].append(all_edge_weight)

    # Define the penalty based on cycles found
    A_constraint = set()
    delta_lst = []
    
    for tmp_len in all_closed_walk_edge:
        if tmp_len <= 3:
            for tmp_closed_walk_edge, tmp_closed_walk_edge_weight in zip(all_closed_walk_edge[tmp_len], all_closed_walk_edge_weight[tmp_len]):
                tmp_constraint = np.zeros(d * (1 + w * d))
                for idx in range(tmp_len):
                    idxi, idxj = tmp_closed_walk_edge[idx+1], tmp_closed_walk_edge[idx]
                    tmp_constraint[idxi * (1 + w * d) + idxj + 1] = 1
                A_constraint.add(tuple(tmp_constraint.tolist()))
                delta_lst.append(sum(tmp_closed_walk_edge_weight) - min(tmp_closed_walk_edge_weight))

    A_constraint = np.array(list(A_constraint))  # Convert to array
    
    return {
        "delta_lst": delta_lst,
        "A_constraint": A_constraint,
        "all_closed_walk_num": all_closed_walk_num,
        "all_closed_walk_weights": all_closed_walk_weights,
        "all_closed_walk_edge": all_closed_walk_edge,
        "all_closed_walk_edge_weight": all_closed_walk_edge_weight
    }

# Penalty for linear cycle detection
def p_linear_circle(A, reg_coef, A_constraint, delta_lst):
    return reg_coef * (1 / np.array(delta_lst)) @ np.array(A_constraint).reshape(A.shape)

# Adaptive L1 penalty
def p_ada_l1(A, reg_coef, A_ada_l1):
    lasso_grad = np.zeros(A[:,1:].shape)
    pos_idx = (A[:,1:] > 0) & (A_ada_l1 != 0)
    neg_idx = (A[:,1:] < 0) & (A_ada_l1 != 0)

    lasso_grad[pos_idx] = 1 / A_ada_l1[pos_idx]
    lasso_grad[neg_idx] = -1 / A_ada_l1[neg_idx]
    np.fill_diagonal(lasso_grad, 0)  # No update on diagonal

    ans = np.zeros(A.shape)
    ans[:,1:] = lasso_grad

    return reg_coef * ans

# Standard L1 penalty
def p_l1(A, reg_coef):
    lasso_grad = np.sign(A[:,1:])  # Derivative of L1 is sign
    np.fill_diagonal(lasso_grad, 0)  # No update on diagonal

    ans = np.zeros(A.shape)
    ans[:,1:] = lasso_grad

    return reg_coef * ans

# Penalty for enforcing DAG structure (version 1)
def p_DAG_1(A, reg_coef):
    h_grad = expm(A[:,1:]).T  # Matrix exponential of adjacency matrix
    ans = np.zeros(A.shape)
    ans[:,1:] = reg_coef * h_grad

    return ans

# Penalty for enforcing DAG structure (version 2) with diagonal exclusion
def p_DAG_2(A, reg_coef):
    h_grad = expm(A[:,1:]).T
    np.fill_diagonal(h_grad, 0)  # Exclude diagonal elements

    ans = np.zeros(A.shape)
    ans[:,1:] = reg_coef * h_grad

    return ans
