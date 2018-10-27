import numpy as np
from load_data import load_data
from preprocessing import *
from cross_validation_logistic import *
from cross_validation_lasso import plot_cross_validation

x_train,y_train,x_test,ids_test=load_data('train.csv','test.csv')
# Setting parameters
degrees=10
k_fold=5
lambdas=np.logspace(-8,-1,num=5)
seed=1
vec=cross_validation_demo(y_train,x_train,degrees,k_fold,lambdas,seed)
#plot_cross_validation(lambdas,cost_te,degrees,'ridge_regression')
best_lambda_idx=np.argmax(vec)

x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)
x_train_cleaned,noaf=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)
phi_train=build_polinomial(x_train_cleaned,degrees,not_poly_features=noaf+nmc_tr+1,nm=-999,already_cleaned=True)
phi_train=norm_data(phi_train,not_norm_features=nmc_tr+1,skip_first_col=True)

x_test_cleaned,nmc_te=cleaning_function(x_test,-999)
x_test_cleaned,noaf=features_augmentation(x_test_cleaned,not_augm_features=nmc_te+1)
phi_test=build_polinomial(x_test_cleaned,degrees,not_poly_features=noaf+nmc_te+1,nm=-999,already_cleaned=True)
phi_test=norm_data(phi_test,not_norm_features=nmc_te+1,skip_first_col=True)

loss,w=lasso_regression_SGD(y_train, phi_train, lambdas[best_lambda_idx])

y_test=phi_test.dot(w)
y_pred=[]
for i in range(y_test.shape[0]):
    if y_test[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(-1)
        
create_csv_submission(ids_test, y_pred, 'submission_log_gd.csv')