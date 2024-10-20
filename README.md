# Transfusion Data Analysis and DAG Visualization Project

This repository contains code and data related to the analysis of transfusion data, the testing and training of models, and visualization of Directed Acyclic Graphs (DAGs). The focus of this project is to utilize DAG-based models and various loss functions to perform advanced data analysis on transfusion-related datasets.

## Repository Structure

- **Data**:  
  - `Transfusion Testing data_20240704.pkl`  
  - `Transfusion Training data_20240704.pkl`  
  - `Transfusion Validation data_20240704.pkl`  
  These pickle files contain the transfusion datasets used for training, testing, and validation purposes.
  
- **Notebooks**:  
  - `DAG visualization.ipynb`: A notebook for visualizing DAGs and their impact on transfusion data.
  - `Plot_loss_function.ipynb`: A notebook focusing on plotting various loss functions used in model evaluation.
  - `Validation for Hyperparameters.ipynb`: A notebook to validate the hyperparameters used in training models.
  
- **Python Scripts**:  
  - `DAG_lib.py`: A Python library containing functions related to DAG-based model processing.
  - `Transfusion_DAG_parallel.py`: A script for parallelizing the processing of DAG models for transfusion data.
  
- **Job Files**:  
  - `Transfusion_DAG_parallel.job`: A job file used to execute DAG processing in a parallelized environment.

## Project Overview

This project leverages DAG-based models to analyze transfusion data. It involves:
- Visualizing DAGs for better interpretability.
- Using loss functions to optimize the model.
- Training and validating models with transfusion data.
- Hyperparameter tuning to improve model performance.

## Key Features

- **DAG Visualization**: Provides clear visual insights into relationships and dependencies between variables in the transfusion datasets.
- **Loss Function Analysis**: Different loss functions are applied and analyzed for optimizing DAG models.
- **Parallelization**: The project includes methods to parallelize the execution of DAG models, improving computational efficiency.
