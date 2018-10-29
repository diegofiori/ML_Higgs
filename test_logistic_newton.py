import numpy as np
from load_data import load_data
from implementations import *
from preprocessing import *


# Load data and select only a subset
x_train,y_train,x_test,ids_test=load_data('train.csv','test.csv')
seed=1
y_train,x_train=retrieve_subset(y_train, x_train, num_obs=2500, set_seed=1)

# Define variables
degree_= 3
logistic_type=0

# Clean data
x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)

# Feature augmentation
x_train_cleaned,noaf=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)

# Normalize data
phi_train = build_polinomial(x_train_cleaned,degree_,not_poly_features=noaf+nmc_tr+1,nm=-999,already_cleaned=True)
phi_train=norm_data(phi_train,not_norm_features=nmc_tr+1,skip_first_col=True)
phi_train = norm_max(phi_train)

# Define parameters
max_iter=1000
threshold=1e-7
gamma=1e-6
w,loss=logistic_regression_newton_method_demo(y_train, phi_train, max_iter, threshold, gamma)


# Calculate result on the train sample
result=(y_train==(sigmoid(phi_train.dot(w))>0.5)).sum()/y_train.shape[0]

y_test=sigmoid(phi_test.dot(w))>0.5
y_pred=[]
for i in range(y_test.shape[0]):
    if y_test[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(-1)
        
        
create_csv_submission(ids_test, y_pred, 'submission_log_gd.csv')