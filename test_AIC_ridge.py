import numpy as np
from load_data import load_data
from preprocessing import *
from cross_validation_ridge import *
from regression_tools import *
from implementations import *
from AIC import *

# Read the data and set parameters
x_train,y_train,x_test,ids_test=load_data('train.csv','test.csv')
seed = 1

#Choose the lambda and the degree
lambda_ = 1e-2
degree = 4

# Clean the x vector from incorrect values 
x_train_cleaned,nmc_tr=cleaning_function(x_train, -999)

# Features augmentation
x_train_cleaned,noaf=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)
''' if you want to test, comment the previous line 
    set noaf = 0'''
#noaf = 0


# Build the polinomial to train
phi_train=build_polinomial(x_train_cleaned, degree, not_poly_features=noaf+nmc_tr+1, nm=-999, already_cleaned=True)

# Normalize the data
phi_train=norm_data(phi_train, not_norm_features=nmc_tr+1, skip_first_col=True)
        
    
# Retrieve a subset
#y_train,phi_train = retrieve_subset(y_train, phi_train, int(phi_train.shape[0]/250))

# Calculate the best model
model_feature = compare_aic_ridge(y_train, phi_train,lambda_)


_,w = ridge_regression(y_train, phi_train[:,model_feature],lambda_)

# ONLY TO TEST LOCALLY
#result=(y_train==(sigmoid(phi_train[:,model_feature].dot(w))>0.5)).sum()/y_train.shape[0]
#print(result)


# Retrieve the test sample
x_test_cleaned,nmc_te=cleaning_function(x_test,-999)

# Features augmentation
x_test_cleaned,noaf=features_augmentation(x_test_cleaned,not_augm_features=nmc_te+1)

# Build the polinomial to test
phi_test=build_polinomial(phi_test,degree,not_poly_features=noaf+nmc_te+1,nm=-999,already_cleaned=True)

# Normalize the data
phi_test=norm_data(phi_test,not_norm_features=nmc_te+1,skip_first_col=True)

# Apply the feature selection
phi_te = phi_test[:,model_feature]

# Calculate the prediction
y_test=sigmoid(phi_te.dot(w))>0.5
y_pred=[]
for i in range(y_test.shape[0]):
    if y_test[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(-1)
        
        
#create_csv_submission(ids_test, y_pred, 'submission_log_gd.csv')