import numpy as np
from load_data import load_data
from preprocessing import *
from cross_validation_logistic import *
from cross_validation_ridge import plot_cross_validation

x_train,y_train,x_test,ids_test=load_data('train.csv','test.csv')
degrees=np.arange(1,20)
k_fold=5
lambdas=np.logspace(-10,-7)
seed=1
logistic_type=0
mat=cross_validation_logistic_demo(y_train,x_train,degrees,k_fold,lambdas,seed,logistic_type, max_iter=1000, threshold=1e-8)
plot_cross_validation(lambdas,mat,degrees,'logistic_gradient_descent')
best_deg=np.min(np.argmax(np.max(mat,axis=0)))
best_gamma=np.max(np.argmax(mat[:,best_deg]))

x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)
x_train_cleaned,noaf=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)
phi_train=build_polinomial(x_train_cleaned,degree=degrees[best_deg],not_poly_features=noaf+nmc_tr+1,nm=-999,already_cleaned=True)
phi_train=norm_data(phi_train,not_norm_features=nmc_tr+1,skip_first_col=True)

x_test_cleaned,nmc_te=cleaning_function(x_test,-999)
x_test_cleaned,noaf=features_augmentation(x_test_cleaned,not_augm_features=nmc_te+1)
phi_test=build_polinomial(x_test_cleaned,degree=degrees[best_deg],not_poly_features=noaf+nmc_te+1,nm=-999,already_cleaned=True)
phi_test=norm_data(phi_test,not_norm_features=nmc_te+1,skip_first_col=True)

loss,w=logistic_regression_gradient_descent_demo(y_train, phi_train, gamma=lambdas[best_gamma], max_iter=1000, threshold=1e-8)

y_test=sigmoid(phi_test.dot(w))>0.5
y_pred=[]
for i in range(y_test.shape[0]):
    if y_test[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(-1)
        #b=-1
        
create_csv_submission(ids_test, y_pred, 'submission.csv')





