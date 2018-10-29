import numpy as np
from regression_tools import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.
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
    """
    w=np.linalg.solve(np.matmul(tx.transpose(),tx),tx.transpose().dot(y))
    loss=compute_loss(y,tx,w)
    return w,loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.
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
    """
    w=initial_w
    for n_iter in range(max_iters):
        g=compute_gradient_lasso(y,tx,w,lambda_)
        w=w-gamma*g
    loss=np.linalg.norm(y-tx.dot(w))**2/(2*tx.shape[0])+lambda_*np.absolute(w).sum()
    return w,loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Returns a model estimated by logistic regression using logistic regression.
    """
    # init parameters
    threshold=1e-8
    losses = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y.reshape(-1,1), tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2])/np.abs(losses[-1]) < threshold:
            break
    return w,loss

def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
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