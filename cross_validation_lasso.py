import numpy as np
import matplotlib.pyplot as plt
from regression_tools import * 
from preprocessing import *

def cross_validation_lasso(y, phi, k_indices, k, lambda_, not_poly_features):
    """
    Computes one step of cross validation with lasso regression and returns the fraction of correct classifications.
    """
    # Get k'th subgroup in test, others in train    
    train_indices = np.delete(k_indices , k , 0).reshape((k_indices.shape[0]-1) * k_indices.shape[1])
    x_test = phi[k_indices[k],:]
    x_train = phi[train_indices,:]
    y_test = y[k_indices[k]]
    y_train = y[train_indices]
    #print(tx_test.shape)
    #print(tx_train.shape)
    #print(y_train.shape)
    loss , w = lasso_regression_SGD(y_train,tx_train, lambda_)
    # Calculate results
    result=(y_test==(tx_test.dot(w)>0.5)).sum()/y_test.shape[0]
    #print('RESULT CALCULATED')
    return result


def cross_validation_demo(y_train,x_train,degrees,k_fold,lambdas,seed,key):
    """
    Return the matrix of proportion of correct classifications obtained by cross validation from lasso regression.
    """
    # split data in k fold
    k_indices = build_k_indices(y_train, k_fold, seed)
    #Clean data
    x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)
    #feature augmentation
    x_train_cleaned,noaf=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)

    # cross validation
    cost_te=np.zeros((lambdas.size,degrees.size))
    for ind_lamb,lambda_ in enumerate(lambdas):
        print(lambda_)
        if lambda_!=0:
            phi_train=build_polinomial(x_train_cleaned,degree,not_poly_features=noaf+nmc_tr+1)
            phi_train=norm_data(x_train_cleaned,not_norm_features=nmc_tr+1)
            loss_te = np.zeros(k_fold)
            for k in range (k_fold):
                result = cross_validation_lasso(y_train, phi_train, k_indices, k , lambda_, nmc_tr+1+noaf)
                loss_te[k]= result

            cost_te[ind_lamb,ind_deg]=loss_te.mean()
    return cost_te