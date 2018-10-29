import numpy as np
import matplotlib.pyplot as plt
from regression_tools import * 
from preprocessing import *

def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold cross-validation.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_ridge(y, phi, k_indices, k, lambda_, degree, not_poly_features):
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

def cross_validation_demo(y_train,x_train,degrees,k_fold,lambdas,seed):
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
    # Feature augmentation
    x_train_cleaned,noaf=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)
    # Cross validation steps
    cost_te=np.zeros((lambdas.size,degrees.size))
    for ind_lamb,lambda_ in enumerate(lambdas):
        if lambda_!=0:
            x_train_cleaned=norm_data(x_train_cleaned,not_norm_features=noaf+nmc_tr+1)
        for ind_deg, degree_ in enumerate(degrees):
            loss_te = np.zeros(k_fold)
            for k in range (k_fold):
                result = cross_validation_ridge(y_train, x_train_cleaned, k_indices, k , lambda_, degree_, nmc_tr+1+noaf)
                loss_te[k]= result

            cost_te[ind_lamb,ind_deg]=loss_te.mean()
    return cost_te


def plot_cross_validation(lambdas,cost_te,degrees,regression_type):
    plt.figure()
    string=[]
    for s in range(lambdas.size):
        plt.plot(degrees,cost_te[s])
        string.append(str(lambdas[s]))
    plt.xlabel('degree')
    plt.ylabel('train accuracy')
    plt.legend(string)
    plt.savefig('cross_validation '+regression_type+'.png')
    
def find_the_maximum(matrix):
    max_col=np.max(matrix,axis=0)
    max_col_ind=np.max(np.argmax(max_col))
    max_matrix=np.max(max_col)
    max_row_ind=np.min(np.argmax(matrix[:,max_col_ind]))
    return max_matrix,[max_row_ind,max_col_ind] 

def find_the_maximum_3D(tensor):
    max_mat=np.max(tensor,axis=0)
    depth_mat=np.argmax(tensor,axis=0)
    _,[ind_row,ind_col]=find_the_maximum(max_mat)
    ind_depth=depth_mat[ind_row,ind_col]
    max_tensor=np.max(tensor)
    return max_tensor,[ind_depth,ind_row,ind_col]