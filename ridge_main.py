import numpy as np
from cross_validation_ridge import *
from load_data import load_data
from preprocessing import *

# Load data
x_train,y_train,x_test,ids_test=load_data('train.csv','test.csv')

# Set parameters
seed = 1
degrees = np.arange(5,16)
k_fold = 4
lambdas = np.logspace(-8,-1,num=5)

# Cross validation to find best parameters
cost_te=cross_validation_demo(y_train,x_train,degrees,k_fold,lambdas,seed)

plot_cross_validation(lambdas,cost_te,degrees,'ridge')

# Retrieve the best
_,best_param_ind=find_the_maximum(cost_te)

# Clean and preprocess the train sample
x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)
x_train_cleaned,noac_tr=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)
x_train_cleaned=norm_data(x_train_cleaned,not_norm_features=nmc_tr+1)
phi_tr=build_polinomial(x_train_cleaned,degree=degrees[best_param_ind[1]],not_poly_features=nmc_tr+1+noac_tr)

# Use the whole dataset
w,loss=ridge_regression(y_train,phi_tr,lambdas[best_param_ind[0]])

#Clean and preprocess the test sample
x_test_cleaned,nmc_te=cleaning_function(x_test,-999)
x_test_cleaned,noac_te=features_augmentation(x_test_cleaned,not_augm_features=nmc_te+1)
x_test_cleaned=norm_data(x_test_cleaned,not_norm_features=nmc_te+1)
phi_te=build_polinomial(x_test_cleaned,degree=degrees[best_param_ind[1]],not_poly_features=nmc_te+1+noac_te)

# Calculate result on train sample
result = (y_train==(phi_tr.dot(w)>0.5)).sum()/y_train.shape[0]

# Result on test sample
y_test=phi_te.dot(w)
y_pred=[]
for i in range(y_test.shape[0]):
    if y_test[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(-1)
        
create_csv_submission(ids_test, y_pred, 'submission_ridge.csv')