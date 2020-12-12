import os
import pickle
import numpy as np



def load_CIFAR_batch(filename):
    """Load a single batch of CIFAR."""
    with open(filename, 'rb') as file:
        datadict = pickle.load(file, encoding = 'latin1')

        X = datadict['data']
        Y = datadict['labels']
        
        X, Y = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float), np.array(Y)

        return X, Y



def load_CIFAR10(ROOT):
    """Load all of CIFAR."""
    Xs = []
    ys = []

    for batch in range(1, 6):
        filename = os.path.join(ROOT, f'data_batch_{batch}')
        
        X, y = load_CIFAR_batch(filename)

        Xs.append(X)
        ys.append(y)    

    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)

    del X, y
    
    X_test, y_test = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    return X_train, y_train, X_test, y_test
