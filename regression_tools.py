import numpy as np

def grid_search(y, tx, w0, w1):
    """
    Algorithm for grid search.
    """
    losses = np.zeros((len(w0), len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            w=np.array([w0[i],w1[j]])
            losses[i,j]=compute_loss(y,tx,w)
    return losses

def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient mse.
    """
    err=y-tx.dot(w)
    d_L=-tx.transpose().dot(err)/tx.shape[0]
    return d_L

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        d_L=compute_gradient_mse(y,tx,w)
        loss=compute_loss(y,tx,w)
        w=w-gamma*d_L
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]

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



def compute_stoch_gradient_mse(y, tx, w):
    """
    Compute a stochastic gradient from just few examples n and their corresponding y_n labels.
    """
    err=y-tx.dot(w)
    dL=-tx.transpose().dot(err)/tx.shape[0]
    return dL

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic gradient descent algorithm.
    """
    w=initial_w
    for n_iter in range(max_iters):
        for mini_y,mini_tx in batch_iter(y,tx,batch_size):
            g=compute_stoch_gradient_mse(mini_y,mini_tx,w)
            w=w-gamma*g
    loss=compute_loss(y,tx,w)
    return loss, w


def least_squares(y, tx):
    """
    Calculate the least squares.
    """
    w=np.linalg.solve(np.matmul(tx.transpose(),tx),tx.transpose().dot(y))
    mse=compute_loss(y,tx,w)
    return mse,w

def compute_loss(y,tx,w):
    """
    Compute the MSE loss.
    """
    L=np.linalg.norm(y-tx.dot(w))**2/(2*y.shape[0])
    return L

def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    """
    phi_list=[]
    for i in range(degree+1):
        phi_list.append(x**i)
    phi=np.array(phi_list).transpose()
    return phi

def ridge_regression(y, tx, lambda_):
    """
    Implement ridge regression.
    """
    n=len(y)
    I=np.diag(np.ones(tx.shape[1]))
    A=np.matmul(tx.transpose(),tx)+2*n*lambda_*I
    b=tx.transpose().dot(y)
    w=np.linalg.solve(A,b)
    L=np.linalg.norm(y-tx.dot(w))**2/tx.shape[0]/2+lambda_*(np.linalg.norm(w)**2)
    return L,w

def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio.
    """
    np.random.seed(seed)
    np.random.shuffle(y)
    np.random.seed(seed)
    np.random.shuffle(x)
    nb_train=int(np.ceil(ratio*x.shape[0]))
    y_train=y[:(nb_train-1)]
    x_train=x[:nb_train-1]
    y_test=y[nb_train:y.shape[0]-1]
    x_test=x[nb_train:y.shape[0]-1]
    return x_train,y_train,x_test,y_test

def compute_stoch_gradient_ridge(y,tx,w,lambda_):
    """
    Returns the stochastic gradient in the ridge model.
    """
    err=y-tx.dot(w)
    if len(y)>1:
        dL=-tx.transpose().dot(err)/tx.shape[0]+2*lambda_*w
    else:
        dL=-tx.reshape(-1,1)*err+2*lambda_*w
    return dL

def ridge_regression_SGD(y, tx, lambda_, initial_w, batch_size, max_iters, gamma):
    """
    Returns the model of a ridge regression found with stochastic gradient descent.
    """
    w=initial_w
    for n_iter in range(max_iters):
        for mini_y,mini_tx in batch_iter(y,tx,batch_size):
            g=compute_stoch_gradient_ridge(mini_y,mini_tx,w,lambda_)
            w=w-gamma*g
    loss=np.linalg.norm(y-tx.dot(w))**2/(2*len(y))+lambda_*np.linalg.norm(w)**2
    return loss, w

def sign(x):
    """
    Computes the sign() function.
    """
    true_vec1=x[:]>0
    true_vec2=x[:]<0
    x=true_vec1-true_vec2
    return x

def compute_stoch_gradient_lasso(y,tx,w,lambda_):
    """
    Returns the stochastic gradient in the lasso model.
    """
    err=y-tx.dot(w)
    if len(y)>1:
        dL=-tx.transpose().dot(err)/tx.shape[0]+lambda_*sign(w)
    else:
        dL=-tx.reshape(-1,1)*err+lambda_*sign(w)
    return dL

def lasso_regression_SGD(y, tx, lambda_, initial_w, batch_size, max_iters, gamma):
    """
    Returns the model of a lasso regression found with stochastic gradient descent.
    """
    w=initial_w
    for n_iter in range(max_iters):
        for mini_y,mini_x in batch_iter(y,tx,batch_size):
            g=compute_stoch_gradient_lasso(y,tx,w,lambda_)
            w=w-gamma*g
    loss=np.linalg.norm(y-tx.dot(w))**2/(2*tx.shape[0])+lambda_*np.absolute(w).sum()
    return loss,w

def sigmoid(t):
    """
    Apply sigmoid function on t.
    """
    return(np.exp(t)/(1+np.exp(t)))

def calculate_loss_logistic(y, tx, w):
    """
    Compute the cost by negative log likelihood.
    """
    return -sum(y*(np.dot(tx,w))-np.log(1+np.exp(np.dot(tx,w))))

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    y = y.reshape(tx.shape[0],1)
    # compute the cost
    loss=calculate_loss_logistic(y, tx, w)
    # compute the gradient
    grad=np.transpose(tx).dot(sigmoid(np.dot(tx,w))-y)
    # update w
    w=w-gamma*grad
    return loss, w

def learning_by_stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    w=initial_w
    for n_iter in range(max_iters):
        for mini_y,mini_x in batch_iter(y,tx,batch_size):
            g=np.transpose(mini_x)*sigmoid(np.dot(mini_x,w)-mini_y)
            w=w-gamma*g
    loss=calculate_loss_logistic(y, tx, w)
    return loss, w
              
def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
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
    return loss, w


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    N = tx.shape[0]
    D = tx.shape[1]
    y = y.reshape(N,1)
    # return loss, gradient: TODO
    loss=calculate_loss_logistic(y, tx, w)+0.5*lambda_*np.linalg.norm(w)**2
    grad=np.transpose(tx).dot(sigmoid(np.dot(tx,w))-y)+lambda_*w
    # compute hessian
    s1=sigmoid(np.dot(tx,w))
    d = s1 * (1-s1)
    H = np.zeros((D,D))
    for n in range(N):
        c = tx[n,:].reshape(-1,1)
        H += c.dot(c.T) * d[n]
    # update w
    w=w-np.linalg.solve(H,gamma*grad)
    return loss, w


def logistic_regression_gradient_descent_demo(y, tx, gamma, max_iter, threshold):
    """
    Returns a model estimated by logistic regression using logistic regression.
    """
    # init parameters
    losses = []
    w = np.zeros((tx.shape[1],1))
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y.reshape(-1,1), tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return loss,w

              
def AIC(w,l):
    """
    Return the Akaike Information Criterion of a model with parameters w and negative log likelihood l
    """
    return -2*w.shape[0]-2*l
              