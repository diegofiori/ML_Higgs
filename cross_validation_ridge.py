import numpy as np
import matplotlib.pyplot as plt
from regression_tools import * 
from preprocessing import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def LOO_cross_validation(y, phi, k_indices, k, lambda_, degree, not_poly_features):
    """return the loss of ridge/linear regression."""
    """Probabilmente conviene implementare una per la regressione normale, in modo da capire 
    quale sia il grado massimo oltre il quale non ha senso andare e poi lavorare con lambda 
    per capire come eliminare feature"""
    
    
    # Get k'th subgroup in test, others in train    
    train_indices = np.delete(k_indices , k , 0).reshape((k_indices.shape[0]-1) * k_indices.shape[1])
    x_test = phi[k_indices[k],:]
    x_train = phi[train_indices,:]
    y_test = y[k_indices[k]]
    y_train = y[train_indices]
    
    # Form data with polynomial degree
    tx_train = build_polinomial(x_train, degree, not_poly_features)
    tx_test = build_polinomial(x_test, degree, not_poly_features)
    #print(tx_test.shape)
    #print(tx_train.shape)
    #print(y_train.shape)
    
    
    # Ridge regression / Linear regression
    #if tx_train.shape[1]<50:
    if lambda_!=0:
        loss , w = ridge_regression(y_train, tx_train, lambda_)
    else:
        loss , w = least_squares(y_train,tx_train)
            
    #else: 
        # forse è meglio implementarle all'esterno della funzione
        #initial_w=np.ones((tx_train.shape[1]))
        #batch_size=1
        #max_iters=100
        #gamma=0.01
        #if lambda_!=0:
        #    loss , w = ridge_regression_SGD(y_train, tx_train, lambda_,initial_w, batch_size, max_iters, gamma)
        #else:
        #    loss , w = least_squares_SGD(y_train,tx_train,initial_w, batch_size, max_iters, gamma)
        
    
    
    #print('REGRESSION DONE')
    #print(y_test.shape)
    #print(w.shape)
    
    # Calculate results
    result=(y_test==(tx_test.dot(w)>0.5)).sum()/y_test.shape[0]
    #print('RESULT CALCULATED')
    return result

def cross_validation_demo(y_train,x_train,degrees,k_fold,lambdas,seed):

    # split data in k fold
    k_indices = build_k_indices(y_train, k_fold, seed)


    # Clean data 
    x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)
    #feature augmentation
    x_train_cleaned,noaf=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)

    # cross validation
    cost_te=np.zeros((lambdas.size,degrees.size))
    for ind_lamb,lambda_ in enumerate(lambdas):
        print(lambda_)
        if lambda_!=0:
            x_train_cleaned=norm_data(x_train_cleaned,not_norm_features=noaf+nmc_tr+1)
        for ind_deg, degree_ in enumerate(degrees):
            #print('DEGREE IS: ')
            #print(degree_)
            loss_te = np.zeros(k_fold)
            for k in range (k_fold):
                #print('K CONSIDERED IS: ')
                #print(k)
                result = LOO_cross_validation(y_train, x_train_cleaned, k_indices, k , lambda_, degree_, nmc_tr+1+noaf)
                loss_te[k]= result

            cost_te[ind_lamb,ind_deg]=loss_te.mean()
    return cost_te


def plot_cross_validation(lambdas,cost_te,degrees):
    plt.figure
    string=[]
    for s in range(lambdas.size):
        plt.plot(degrees,cost_te[s])
        string.append(str(lambdas[s]))
    plt.xlabel('degree')
    plt.ylabel('train accuracy')
    plt.legend(string)
    plt.show()
    
def find_the_maximum(matrix):
    max_col=np.max(matrix,axis=0)
    max_col_ind=np.max(np.argmax(max_col))
    max_matrix=np.max(max_col)
    max_row_ind=np.min(np.argmax(matrix[:,max_col_ind]))
    return max_matrix,[max_row_ind,max_col_ind] 