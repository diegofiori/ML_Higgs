import numpy as np
from implementations import *

def compare_aic_gradient_descent(y,tx,gamma,max_iter,threshold):
    dimx=tx.shape[1]
    aic_chosen=np.zeros(dimx) #contains best loss of model with m variables
    models=[] #list of best models
    variables=list(range(dimx)) #list of variables
    for ind in range(dimx):
        loss=np.zeros(dimx) #contains loss for all models with m variables
        loss.fill(np.inf)
        aic=np.zeros(dimx)  #contains aic for all models with m variables
        aic.fill(np.inf)
        w=np.zeros(dimx)
        for m in variables:
            temp=models.copy()
            temp.append(m)
            [loss[m],w]=logistic_regression(y, tx[:,temp], w, max_iter, gamma)
            aic[m] = AIC(w,loss[m])

        b=np.argmin(loss)
        models.append(b)
        variables.remove(b)
        aic_chosen[ind]=aic[b]

    idx_loss=np.argmin(aic_chosen)
    model_feature=models[:idx_loss+1]
    return model_feature


def compare_aic_ridge(y,tx,lambda_):
    dimx=tx.shape[1]
    aic_chosen=np.zeros(dimx) #contains best loss of model with m variables
    models=[] #list of best models
    variables=list(range(dimx)) #list of variables
    for ind in range(dimx):
        loss=np.zeros(dimx) #contains loss for all models with m variables
        loss.fill(np.inf)
        aic=np.zeros(dimx)  #contains aic for all models with m variables
        aic.fill(np.inf)
        w=np.zeros(dimx)
        for m in variables:
            temp=models.copy()
            temp.append(m)
            w, loss[m] =ridge_regression(y, tx[:,temp],lambda_)
            aic[m] = AIC(w,-loss[m])

        b=np.argmin(loss)
        models.append(b)
        variables.remove(b)
        aic_chosen[ind]=aic[b]

    idx_loss=np.argmin(aic_chosen)
    model_feature=models[:idx_loss+1]
    return model_feature

