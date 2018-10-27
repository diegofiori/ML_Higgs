import numpy as np

def retrieve_subset(y,x,num_obs, seed_set=1):

    # Select randomly a subset
    np.random.seed(seed_set)
    tot_observation = x.shape[0]
    idx = np.random.randint(tot_observation, size=num_obs)
    x_small = x[idx,:]
    y_small = y[idx]

    return y_small , x_small

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


def compute_loss(y,tx,w):
    """
    Compute the MSE loss.
    """
    loss=np.linalg.norm(y-tx.dot(w))**2/(2*y.shape[0])
    return loss

def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    """
    phi_list=[]
    for i in range(degree+1):
        phi_list.append(x**i)
    phi=np.array(phi_list).transpose()
    return phi


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


def sign(x):
    """
    Computes the sign() function.
    """
    true_vec1=x[:]>0
    true_vec2=x[:]<0
    x=1*true_vec1-1*true_vec2
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
    return loss,w

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


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
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
        for m in variables:
            temp=models.copy()
            temp.append(m)
            [loss[m],w]=logistic_regression_gradient_descent_demo(y, tx[:,temp], gamma, max_iter, threshold)
            aic[m] = AIC(w,loss[m])

        b=np.argmin(loss)
        models.append(b)
        variables.remove(b)
        aic_chosen[ind]=aic[b]

    idx_loss=np.argmin(aic_chosen)
    model_feature=models[:idx_loss+1]
    return model_feature


              
def AIC(w,l):
    """
    Return the Akaike Information Criterion of a model with parameters w and negative log likelihood l
    """
    return -2*w.shape[0]-2*l
              