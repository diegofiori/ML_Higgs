import numpy as np
import csv

### Generic tools

def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    Takes as input: x, input data
                    degree, maximum degree to which variables are raised.
    Returns: phi, model matrix with polynomial features.
    """
    phi_list=[]
    for i in range(degree+1):
        phi_list.append(x**i)
    phi=np.array(phi_list).transpose()
    return phi


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


### Tools for linear regression

def grid_search(y, tx, w0, w1):
    """
    Algorithm for grid search.
    Takes as input: y, the objective variable
                    w0 and w1, vectors forming the grid along which optimum is searched
                    tx, input data
    Returns: matrix of losses with an entry for each element in the grid.
    """
    losses = np.zeros((len(w0), len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            w=np.array([w0[i],w1[j]])
            losses[i,j]=compute_loss(y,tx,w)
    return losses


def compute_loss(y,tx,w):
    """
    Compute the MSE loss.
    Takes as input: y, the objective variable (batch)
                    tx, input data (batch)
                    w, vector of coefficients of the model.
    Returns: loss of the model.
    """
    loss=np.linalg.norm(y-tx.dot(w))**2/(2*y.shape[0])
    return loss


def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient of the MSE of the model.
    Takes as input: y, the objective variable
                    tx, input data.
                    w, vector of coefficients of the model
    Returns: gradient of the loss of the model.
    """
    err=y-tx.dot(w)
    d_L=-tx.transpose().dot(err)/tx.shape[0]
    return d_L
            

def compute_stoch_gradient_mse(y, tx, w):
    """
    Compute the gradient of the MSE relative to a batch.
    Takes as input: y, the objective variable (batch)
                    tx, input data (batch)
                    w, vector of coefficients of the model.
    Returns: stochastic gradient of the loss of the model.
    """
    err=y-tx.dot(w)
    if len(y)>1:
        dL=-tx.transpose().dot(err)/tx.shape[0]
    else:
        dL=-tx.reshape(-1,1)*err
    dL=-tx.transpose().dot(err)/tx.shape[0]
    return dL


### Tools for ridge regression and the lasso

def compute_stoch_gradient_ridge(y, tx, w, lambda_):
    """
    Computes the stochastic gradient in the ridge regression.
    Takes as input: y, objective variable (batch)
                    tx, input data (batch)
                    w, coefficients of the model
                    lambda_, regularisation parameter
    Returns: stochastic gradient of the loss.
    """
    err=y-tx.dot(w)
    if len(y)>1:
        dL=-tx.transpose().dot(err)/tx.shape[0]+2*lambda_*w
    else:
        dL=-tx.reshape(-1,1)*err+2*lambda_*w
    return dL


def sign(x):
    """
    Computes the sign function.
    """
    true_vec1=x[:]>0
    true_vec2=x[:]<0
    x=1*true_vec1-1*true_vec2
    return x


def compute_gradient_lasso(y,tx,w,lambda_):
    """
    Returns the gradient in the lasso model.
    Takes as input: y, objective variable
                    tx, input data
                    w, coefficients of the model
                    lambda_, regularisation parameter
    Returns: gradient of the loss in the lasso model.
    """
    err=y-tx.dot(w)
    dL=-tx.transpose().dot(err)/tx.shape[0]+lambda_*sign(w)
    return dL


### Tools for Logistic Regression

def sigmoid(t):
    """
    Apply sigmoid function to t.
    """
    return(np.exp(t)/(1+np.exp(t)))


def calculate_loss_logistic(y, tx, w):
    """
    Computes the cost by negative log likelihood.
    Takes as input: y, objective variable
                    tx, input data
                    w, coefficients of the model
    Returns: negative log-likelihood of the logit.
    """
    return -sum(y*(np.dot(tx,w))-np.log(1+np.exp(np.dot(tx,w))))


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Does one step of gradient descent using logistic regression.
    Takes as input: y, objective variable
                    tx, input data
                    w, coefficients of the model
                    gamma, learning parameter
    Return the loss and the updated w.
    """
    y = y.reshape(tx.shape[0],1)
    # compute the cost
    loss=calculate_loss_logistic(y, tx, w)
    # compute the gradient
    grad=np.transpose(tx).dot(sigmoid(np.dot(tx,w))-y)
    # update w
    w=w-gamma*grad
    return w,loss

def stochastic_gradient_descent_logistic(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Do gradient descent using logistic regression.
    Return the loss and the updated w.
    Takes as input: y, objective variable
                    tx, input data
                    initial_w, initial approximation of the coefficients of the model
                    batch_size, size of the batch for which the gradient is computed
                    max_iters, maximum number of iterations of gradient descent
                    gamma, learning parameter
    Return the loss and the updated w.
    """
    w=initial_w
    for n_iter in range(max_iters):
        for mini_y,mini_x in batch_iter(y,tx,batch_size):
            g=np.transpose(mini_x)*sigmoid(np.dot(mini_x,w)-mini_y)
            w=w-gamma*g
    loss=calculate_loss_logistic(y, tx, w)
    return w,loss
            
    
def learning_by_newton_method(y, tx, w, gamma):
    """
    Does one step of gradient descent using Newton's method.
    Takes as input: y, objective variable
                    tx, input data
                    w, coefficients of the model
                    gamma, learning parameter
    Returns the loss and the updated w.
    """
    N = tx.shape[0]
    D = tx.shape[1]
    y = y.reshape(N,1)
    # compute loss
    loss=calculate_loss_logistic(y, tx, w)
    # compute gradient
    grad=np.transpose(tx).dot(sigmoid(np.dot(tx,w))-y)
    # compute hessian
    s1=sigmoid(np.dot(tx,w))
    d = s1 * (1-s1)
    H = np.zeros((D,D))
    for n in range(N):
        c = tx[n,:].reshape(-1,1)
        H += c.dot(c.T) * d[n]
    # update w
    w=w-np.linalg.solve(H,gamma*grad)
    return w,loss


def learning_by_penalized_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Does one step of gradient descent, using the penalized logistic regression.
    Takes as input: y, objective variable
                    tx, input data
                    w, coefficients of the model
                    gamma, learning parameter
                    lambda_, regularising parameter
    Returns the loss and the updated w.
    """
    N = tx.shape[0]
    D = tx.shape[1]
    # return loss, gradient:
    loss=calculate_loss_logistic(y, tx, w)+0.5*lambda_*np.linalg.norm(w)**2
    grad=np.transpose(tx).dot(sigmoid(np.dot(tx,w))-y)+lambda_*w
    # update w
    w=w-gamma*grad
    return w,loss


### Akaike Information Criterion
              
def AIC(w,l):
    """
    Return the Akaike Information Criterion of a model with parameters w and negative log likelihood l.
    Takes as input: w, coefficients of the model,
                    l, negative loglikelihood.
    Returns: the AIC of the model estimated.
    """
    return -2*w.shape[0]-2*l


### Subsample Extraction

def retrieve_subset(y, x, num_obs, set_seed=1):
    """
    Extracts a subset of the data to speed up model estimation.
    Takes as input: y, the objective variable
                    x, input data
                    num_obs, size of the subsample
                    seed
    Returns: subsample of objective variable y and of model matrix x
    """
    # Select randomly a subset
    np.random.seed(set_seed)
    tot_observation = x.shape[0]
    idx = np.random.randint(tot_observation, size=num_obs)
    x_small = x[idx,:]
    y_small = y[idx]
    return y_small , x_small


### File Submission

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Takes as input: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
              