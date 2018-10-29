import numpy as np
from regression_tools import * 
from cross_validation_ridge import *
from cross_validation_ridge_super import *
from cross_validation_lasso import *
from preprocessing import *
from load_data import *
from implementations import *
from AIC import *

x_train, y_train, x_test, ids_test = load_data('train.csv', 'test.csv')
# Setting parameters
seed = 1
degrees = np.arange(9,13)
k_fold = 3
# To use ridge regression
lambdas = np.logspace(-8,-1,num=4)

# Cross Validation
cost_te = cross_validation_super_demo(y_train, x_train, degrees, k_fold, lambdas, seed)
plot_cross_validation(lambdas, cost_te, degrees, 'ridge')
result_ridge, best_param_ind = find_the_maximum(cost_te)


# Train Model

x_train_cleaned, nmc_tr = cleaning_function(x_train, -999)

x_train_cleaned, super_col = super_features_augmentation(x_train_cleaned, y_train, lambdas[best_param_ind[0]], not_super_features = nmc_tr+1, is_train=True, augmentation=False)
num_super_col = len(super_col)

x_train_cleaned, noac_tr = features_augmentation(x_train_cleaned, not_augm_features = nmc_tr+1)
x_train_cleaned = norm_data(x_train_cleaned, not_norm_features = nmc_tr+1)

phi_tr = build_polinomial(x_train_cleaned, degree = degrees[best_param_ind[1]], not_poly_features = nmc_tr + 1 + noac_tr + num_super_col)
w ,loss = ridge_regression(y_train, phi_tr, lambdas[best_param_ind[0]])

# Prediction

x_test_cleaned,nmc_te=cleaning_function(x_test,-999)
x_test_cleaned,super_col=super_features_augmentation(x_test_cleaned,super_col,lambdas[best_param_ind[0]],not_super_features=nmc_tr+1,is_train=False,augmentation=False)
num_super_col=len(super_col)
x_test_cleaned,noac_te=features_augmentation(x_test_cleaned,not_augm_features=nmc_te+1)
x_test_cleaned=norm_data(x_test_cleaned,not_norm_features=nmc_te+1)
phi_te=build_polinomial(x_test_cleaned,degree=degrees[best_param_ind[1]],not_poly_features=nmc_te+1+noac_te+num_super_col)
y_test=phi_te.dot(w)
y_pred=[]
for i in range(y_test.shape[0]):
    if y_test[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(-1)
        
create_csv_submission(ids_test, y_pred, 'submission_super_ridge.csv')

