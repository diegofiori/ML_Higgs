import numpy as np
from implementations import *

def compare_aic_gradient_descent(y,tx,gamma,max_iter,threshold):
    """
    Performs model selection using Akaike Information Criterion: starting from the null model,
    iteratively and greedily adds new predictors such that the loss is minimized. The best logit models with 
    0,1,2... p variables are thus constructed. 
    In the end, the best model is selected by AIC.
    Takes as input: y, objective variable
                    tx, input data
                    gamma, learning parameter of logistic regression with gradient descent
                    max_iter, maximum number of iterations of gradient descent
                    threshold, parameter of the stopping criterion in gradient descent
    Returns the best subset selection according to AIC.
    """
    dimx=tx.shape[1]
    aic_chosen=np.zeros(dimx) #contains best loss of model with m variables
    models=[] #list of best models
    variables=list(range(dimx)) #list of variables
    # iterate over all the possible dimensions of the model
    for ind in range(dimx):
        loss=np.zeros(dimx) #contains loss for all models with m variables
        loss.fill(np.inf)
        aic=np.zeros(dimx)  #contains aic for all models with m variables
        aic.fill(np.inf)
        w=np.zeros(ind+1)
        # select the best variable to be added to the model among the remaining ones
        for m in variables:
            temp=models.copy()
            temp.append(m)
            [w,loss[m]]=logistic_regression(y, tx[:,temp], w, max_iter, gamma)
            aic[m] = AIC(w,loss[m])

        b=np.argmin(loss) #pick the variable that minimises the loss in the model
        models.append(b) #add the variable to the ind-feature model
        variables.remove(b) #remove the variable which was selected
        aic_chosen[ind]=aic[b] #save the AIC of the model selected

    idx_loss=np.argmin(aic_chosen) #select the number of variables that minimise AIC
    model_feature=models[:idx_loss+1]  #retrieve the model
    return model_feature


def compare_aic_ridge(y,tx,lambda_):
    """
    Performs model selection using Akaike Information Criterion: starting from the null model,
    iteratively and greedily adds new predictors such that the loss is minimized. The best logit models with 
    0,1,2... p variables are thus constructed. 
    In the end, the best model is selected by AIC.
    Takes as input: y, objective variable
                    tx, input data
                    lambda_, regularising parameter of ridge regression
    Returns the best subset selection according to AIC.
    """
    dimx=tx.shape[1]
    aic_chosen=np.zeros(dimx) #contains best loss of model with m variables
    models=[] #list of best models
    variables=list(range(dimx)) #list of variables
    # iterate over all the possible dimensions of the model
    for ind in range(dimx):
        loss=np.zeros(dimx) #contains loss for all models with m variables
        loss.fill(np.inf)
        aic=np.zeros(dimx)  #contains aic for all models with m variables
        aic.fill(np.inf)
        w=np.zeros(dimx)
        # select the best variable to be added to the model among the remaining ones
        for m in variables:
            temp=models.copy()
            temp.append(m)
            w, loss[m] =ridge_regression(y, tx[:,temp],lambda_)
            aic[m] = AIC(w,-loss[m])

        b=np.argmin(loss) #pick the variable that minimises the loss in the model
        models.append(b) #add the variable to the ind-feature model
        variables.remove(b) #remove the variable which was selected
        aic_chosen[ind]=aic[b] #save the AIC of the model selected

    idx_loss=np.argmin(aic_chosen) #select the number of variables that minimise AIC
    model_feature=models[:idx_loss+1] #retrieve the model
    return model_feature

