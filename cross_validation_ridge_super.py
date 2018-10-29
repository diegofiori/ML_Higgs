import numpy as np
import matplotlib.pyplot as plt
from regression_tools import * 
from preprocessing import *



def cross_validation_ridge_super(y, phi, k_indices, k, lambda_, degree, not_poly_features):
    """
    Return the proportion of correct classifications of ridge/linear regression in a step of k-fold cross-validation.
    """
    
    # Get k'th subgroup in test, others in train    
    train_indices = np.delete(k_indices , k , 0).reshape((k_indices.shape[0]-1) * k_indices.shape[1])
    x_test = phi[k_indices[k],:]
    x_train = phi[train_indices,:]
    y_test = y[k_indices[k]]
    y_train = y[train_indices]
    
    # Form data with polynomial degree
    tx_train = build_polinomial(x_train, degree, not_poly_features)
    tx_test = build_polinomial(x_test, degree, not_poly_features)

    # Ridge regression / Linear regression
    if lambda_!=0:
        w, loss = ridge_regression(y_train, tx_train, lambda_)
    else:
        w, loss = least_squares(y_train,tx_train)
   
    
    # Calculate proportion of correct classification for given lambda and degree
    result=(y_test==(tx_test.dot(w)>0.5)).sum()/y_test.shape[0]
    return result

def cross_validation_super_demo(y_train,x_train,degrees,k_fold,lambdas,seed):
    """
    Performs cross-validation with ridge regression.
    Returns a matrix which stores the proportion of correct classifications where:
        rows: lambda
        columns: degree of polynomial of the features.
    """

    # Split data in k fold
    k_indices = build_k_indices(y_train, k_fold, seed)
    # Clean data 
    x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)
    # Cross validation steps
    cost_te=np.zeros((lambdas.size,degrees.size))
    for ind_lamb,lambda_ in enumerate(lambdas):
        print(lambda_)
        if lambda_!=0:
            x_train_agm,super_col=super_features_augmentation(x_train_cleaned,y_train,lambda_,not_super_features=nmc_tr+1,is_train=True,augmentation=False)
            super_col_nb=len(super_col)
            x_train_agm,noaf=features_augmentation(x_train_agm,not_augm_features=nmc_tr+1)
            x_train_agm=norm_data(x_train_agm,not_norm_features=nmc_tr+1)
        for ind_deg, degree_ in enumerate(degrees):
            loss_te = np.zeros(k_fold)
            for k in range (k_fold):
                result = cross_validation_ridge_super(y_train, x_train_agm, k_indices, k , lambda_, degree_, nmc_tr+1+noaf+super_col_nb)
                loss_te[k]= result
            print('new deg')

            cost_te[ind_lamb,ind_deg]=loss_te.mean()
    return cost_te