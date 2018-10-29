import numpy as np
from load_data import load_data
from preprocessing import *
from cross_validation_logistic import *
from cross_validation_ridge import plot_cross_validation


# Load data
x_train,y_train,x_test,ids_test=load_data('train.csv','test.csv')

# Set parameter values
degrees=np.arange(1,12)
k_fold=4
gammas=np.logspace(-10,-7,5)
seed=5
logistic_type=0

# Cross validation to find best gamma and best degree
mat=cross_validation_logistic_demo(y_train,x_train,degrees,k_fold,gammas,seed,logistic_type, max_iter=1000, threshold=1e-8)
plot_cross_validation(gammas,mat,degrees,'logistic_gradient_descent')
best_deg_ind=np.min(np.argmax(np.max(mat,axis=0)))
best_gamma_ind=np.max(np.argmax(mat[:,best_deg_ind]))

# Now train the model on the whole dataset

# Clean and transform train data
x_train_cleaned,nmc_tr=cleaning_function(x_train,-999)
x_train_cleaned,noaf=features_augmentation(x_train_cleaned,not_augm_features=nmc_tr+1)
phi_train=build_polinomial(x_train_cleaned,degree=degrees[best_deg_ind],not_poly_features=noaf+nmc_tr+1,nm=-999,already_cleaned=True)
phi_train=norm_data(phi_train,not_norm_features=nmc_tr+1,skip_first_col=True)

# Clean and transform test data
x_test_cleaned,nmc_te=cleaning_function(x_test,-999)
x_test_cleaned,noaf=features_augmentation(x_test_cleaned,not_augm_features=nmc_te+1)
phi_test=build_polinomial(x_test_cleaned,degree=degrees[best_deg_ind],not_poly_features=noaf+nmc_te+1,nm=-999,already_cleaned=True)
phi_test=norm_data(phi_test,not_norm_features=nmc_te+1,skip_first_col=True)

# Calculate the best w training on the whole dataset
initial_w = np.zeros(phi_train.shape[1])
w,loss=logistic_regression(y_train, phi_train, initial_w ,1000, gamma=gammas[best_gamma_ind])

# Calculate result on the train sample
result=(y_train==(sigmoid(phi_train.dot(w))>0.5)).sum()/y_train.shape[0]

y_test=sigmoid(phi_test.dot(w))>0.5
y_pred=[]
for i in range(y_test.shape[0]):
    if y_test[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(-1)
       
        
create_csv_submission(ids_test, y_pred, 'submission_log_gd.csv')




