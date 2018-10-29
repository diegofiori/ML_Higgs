import numpy as np
from regression_tools import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.
    Takes as input: y, the objective variable (batch)
                    tx, input data (batch)
                    initial_w, initial approximation of the vector of coefficients of the model
                    max_iters, maximum number of iterations of gradient descent method
                    gamma, learning parameter
    Returns: estimated coefficients and the loss of the model.
    """
    w = initial_w
    for n_iter in range(max_iters):
        d_L=compute_gradient_mse(y,tx,w)
        w=w-gamma*d_L
    loss=compute_loss(y,tx,w)
    return w,loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.
    Takes as input: y, the objective variable (batch)
                    tx, input data (batch)
                    initial_w, initial approximation of the vector of coefficients of the model
                    max_iters, maximum number of iterations of gradient descent method
                    gamma, learning parameter
    Returns: estimated coefficients and loss of the model.
    """
    batch_size=1
    w=initial_w
    for n_iter in range(max_iters):
        for mini_y,mini_tx in batch_iter(y,tx,batch_size):
            g=compute_stoch_gradient_mse(mini_y,mini_tx,w)
            w=w-gamma*g
    loss=compute_loss(y,tx,w)
    return w,loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations.
    Takes as input: y, the objective variable (batch)
                    tx, input data (batch)
    Returns: estimated coefficients and loss of the model.    
    """
    w=np.linalg.solve(np.matmul(tx.transpose(),tx),tx.transpose().dot(y))
    loss=compute_loss(y,tx,w)
    return w,loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.
    Takes as input: y, the objective variable (batch)
                    tx, input data (batch)
                    lambda_, regularising parameter
    Returns: estimated coefficients and loss of the model.
    """
    n=len(y)
    I=np.diag(np.ones(tx.shape[1]))
    A=np.matmul(tx.transpose(),tx)+2*n*lambda_*I
    b=tx.transpose().dot(y)
    w=np.linalg.solve(A,b)
    loss=np.linalg.norm(y-tx.dot(w))**2/tx.shape[0]/2+lambda_*(np.linalg.norm(w)**2)
    return w,loss


def ridge_regression_SGD(y, tx, lambda_, initial_w, batch_size, max_iters, gamma):
    """
    Ridge regression using stochastic gradient descent.
    Takes as input: y, the objective variable (batch)
                    tx, input data (batch)
                    lambda_, regularising parameter
                    initial_w, initial approximation of the vector of coefficients of the model
                    batch_size, size of the subsample used for the gradient descent
                    max_iters, maximum number of iterations of gradient descent
                    gamma, learning parameter in the gradient descent
    Returns: estimated coefficients and loss of the model.
    """
    w=initial_w
    for n_iter in range(max_iters):
        for mini_y,mini_tx in batch_iter(y,tx,batch_size):
            g=compute_stoch_gradient_ridge(mini_y,mini_tx,w,lambda_)
            w=w-gamma*g
    loss=np.linalg.norm(y-tx.dot(w))**2/(2*len(y))+lambda_*np.linalg.norm(w)**2
    return w,loss


def lasso_regression_GD(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Lasso regression using gradient descent.
    Takes as input: y, the objective variable (batch)
                    tx, input data (batch)
                    lambda_, regularising parameter
                    initial_w, initial approximation of the vector of coefficients of the model
                    max_iters, maximum number of iterations of gradient descent
                    gamma, learning parameter in the gradient descent
    Returns: estimated coefficients and loss of the model.
    """
    w=initial_w
    for n_iter in range(max_iters):
        g=compute_gradient_lasso(y,tx,w,lambda_)
        w=w-gamma*g
    loss=np.linalg.norm(y-tx.dot(w))**2/(2*tx.shape[0])+lambda_*np.absolute(w).sum()
    return w,loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Returns a model estimated by logistic regression gradient descent.
    Takes as input: y, the objective variable (batch)
                    tx, input data (batch)
                    initial_w, initial approximation of the vector of coefficients of the model
                    batch_size, size of the subsample used for the gradient descent
                    max_iters, maximum number of iterations of gradient descent
                    gamma, learning parameter in the gradient descent
    Returns: estimated coefficients and loss of the model.
    """
    # init parameters
    threshold=1e-8
    losses = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y.reshape(-1,1), tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2])/np.abs(losses[-1]) < threshold:
            break
    return w,loss

def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    """
    Returns a model estimated by regularised logistic regression gradient descent.
    Takes as input: y, the objective variable (batch)
                    tx, input data (batch)
                    initial_w, initial approximation of the vector of coefficients of the model
                    batch_size, size of the subsample used for the gradient descent
                    max_iters, maximum number of iterations of gradient descent
                    gamma, learning parameter in the gradient descent
    Returns: estimated coefficients and loss of the model.
    """
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    # start the penalized logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w,loss = learning_by_penalized_gradient_descent(y, tx, w, gamma, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2])/np.abs(losses[-1]) < threshold:
            break
    return w,loss


def logistic_regression_newton_method_demo(y, tx, max_iter, threshold, gamma):
    """
    Performs logistic regression using Newton method to find the coefficients w of the model.
    Takes as input: y, objective variable
                    tx, input data
                    max_iter, maximum number of iterations of Newton method
                    threshold, stopping criterion based on two consecutive losses
                    gamma, learning parameter
    Returns the loss and the updated w.
    """
    # init parameters
    losses = []
    w = np.zeros(tx.shape[1],)
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        w, loss = learning_by_newton_method(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2])/np.abs(losses[-1]) < threshold:
            break
    return w,loss