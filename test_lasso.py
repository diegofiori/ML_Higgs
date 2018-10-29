import numpy as np
import csv
from regression_tools import * 
from cross_validation_ridge import *
from cross_validation_lasso import *
from preprocessing import *
from load_data import *
from implementations import *
import matplotlib.pyplot as plt

# Load data
x_train,y_train,x_test,ids_test=load_data('train.csv','test.csv')
# Setting parameters
k_fold=3
gamma=1e-7
lambdas=np.logspace(-8,-1,num=5)
seed=1
gammas=np.linspace(1e-4,5e-3,num=4)
degrees=np.arange(10,14)
max_iters=200
batch_size=1

# Cross validation to find best combination of parameters
mat3D=cross_validation_lasso_demo(y_train,x_train,degrees,k_fold,lambdas,gammas,max_iters,seed)
for i in range(len(gammas)):
    plot_cross_validation(lambdas,mat3D[i],degrees,'lasso'+str(i))

# Best parameters
result,[best_gamma_ind,best_lambda_ind,best_degree_ind]=find_the_maximum_3D(mat3D)

# Clean and preprocess the train sample
x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)
x_train_cleaned,noaf=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)
phi_train=build_polinomial(x_train_cleaned,degrees[best_degree_ind],not_poly_features=noaf+nmc_tr+1,nm=-999,already_cleaned=True)
phi_train=norm_data(phi_train,not_norm_features=nmc_tr+1,skip_first_col=True)

# Clean and preprocess the test sample
x_test_cleaned,nmc_te=cleaning_function(x_test,-999)
x_test_cleaned,noaf=features_augmentation(x_test_cleaned,not_augm_features=nmc_te+1)
phi_test=build_polinomial(x_test_cleaned,degrees[best_degree_ind],not_poly_features=noaf+nmc_te+1,nm=-999,already_cleaned=True)
phi_test=norm_data(phi_test,not_norm_features=nmc_te+1,skip_first_col=True)

# Calculate w with lasso regression using best parameters
w,loss=lasso_regression_GD(y_train, phi_train, lambdas[best_lambda_ind], initial_w, max_iters, gammas[best_gamma_ind])

# Calculate result on train sample
result = (y_train==(phi_train.dot(w)>0.5)).sum()/y_train.shape[0]

# Result on test sample
y_test=phi_test.dot(w)
y_pred=[]
for i in range(y_test.shape[0]):
    if y_test[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(-1)
        
create_csv_submission(ids_test, y_pred, 'submission_lasso_gd.csv')