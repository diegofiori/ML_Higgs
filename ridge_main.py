import numpy as np
from cross_validation_ridge import *
from load_data import load_data
from preprocessing import *

x_train,y_train,x_test,ids_test=load_data('train.csv','test.csv')
seed = 1
degrees = np.arange(5,16)
k_fold = 4
# To use ridge regression
lambdas = np.logspace(-8,-1,num=5)
print(lambdas.size)
cost_te=cross_validation_demo(y_train,x_train,degrees,k_fold,lambdas,seed)
plot_cross_validation(lambdas,cost_te,degrees,'ridge')
_,best_param_ind=find_the_maximum(cost_te)
print('best degree is '+str(degrees[best_param_ind[1]]))
print('best lambda is '+str(lambdas[best_param_ind[0]]))
x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)
x_train_cleaned,noac_tr=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)
x_train_cleaned=norm_data(x_train_cleaned,not_norm_features=nmc_tr+1)
phi_tr=build_polinomial(x_train_cleaned,degree=degrees[best_param_ind[1]],not_poly_features=nmc_tr+1+noac_tr)
loss,w=ridge_regression(y_train,phi_tr,lambdas[best_param_ind[0]])
x_test_cleaned,nmc_te=cleaning_function(x_test,-999)
x_test_cleaned,noac_te=features_augmentation(x_test_cleaned,not_augm_features=nmc_te+1)
x_test_cleaned=norm_data(x_test_cleaned,not_norm_features=nmc_te+1)
phi_te=build_polinomial(x_test_cleaned,degree=degrees[best_param_ind[1]],not_poly_features=nmc_te+1+noac_te)
y_test=phi_te.dot(w)
y_pred=[]
for i in range(y_test.shape[0]):
    if y_test[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(-1)
        #b=-1
        
create_csv_submission(ids_test, y_pred, 'submission.csv')