import numpy as np
from load_data import load_data
from preprocessing import *
from regression_tools import *
from implementations import *

# Read the data and set parameters
x_train,y_train,x_test,ids_test=load_data('train.csv','test.csv')
seed = 3

#Set parameters values
lambda_ = 1e-5
gamma = 5 * 1e-5
max_iters = 1000
degree = 12

# Clean the x vector from incorrect values 
x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)

# Features augmentation
x_train_cleaned,noaf=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)
''' To test quickly, comment the previous line and 
    set noaf = 0'''
#noaf = 0

# Build the polinomial to train
phi_train=build_polinomial(x_train_cleaned,degree,not_poly_features=noaf+nmc_tr+1,nm=-999,already_cleaned=True)

# Normalize the data
phi_train=norm_data(phi_train,not_norm_features=nmc_tr+1,skip_first_col=True)

# Normalize with the maximum value
phi_train = norm_max(phi_train)

# Initialize variables
initial_w = 5 * np.ones(phi_train.shape[1])
losses = []

# Calculate the optimal w and loss
w,loss = reg_logistic_regression(y_train, phi_train,lambda_,initial_w, max_iters, gamma)

# Calculate result on the train sample
result=(y_train==(sigmoid(phi_train.dot(w))>0.5)).sum()/y_train.shape[0]

# Retrieve the test sample
x_test_cleaned,nmc_te=cleaning_function(x_test,-999)

# Features augmentation
x_test_cleaned,noaf=features_augmentation(x_test_cleaned,not_augm_features=nmc_te+1)
''' To test quickly, comment the previous line and 
    set noaf = 0'''
#noaf = 0

# Build the polinomial to test
phi_test=build_polinomial(x_test_cleaned,degree=degree,not_poly_features=noaf+nmc_te+1,nm=-999,already_cleaned=True)

# Normalize the data
phi_test=norm_data(phi_test,not_norm_features=nmc_te+1,skip_first_col=True)

# Normalize with the maximum value
phi_test = norm_max(phi_test)

# Calculate the prediction
y_test=sigmoid(phi_test.dot(w))>0.5
y_pred=[]
for i in range(y_test.shape[0]):
    if y_test[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(-1)
        
create_csv_submission(ids_test, y_pred, 'submission_log_pen.csv')