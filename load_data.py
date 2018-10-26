def load_data(train_path,test_path):
    """ 
    Load data
    """
    train_reader=np.genfromtxt(train_path,delimiter=',',skip_header=1,converters={1:lambda s: float(0) if s==b'b' else float(1)})
    y_train=train_reader[:,1]
    x_train=train_reader[:,2:]
    test_reader=np.genfromtxt(test_path,delimiter=',',skip_header=1)
    x_test=test_reader[:,2:]
    ids_test=test_reader[:,0]
    return x_train,y_train,x_test,ids_test