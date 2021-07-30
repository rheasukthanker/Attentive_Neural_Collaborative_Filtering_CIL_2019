import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import tensorflow as tf

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def normalize_01(train, test):
    return train/5, test/5

def normalize_11(train_matrix, test_matrix):
    # to values between -1 and 1
    train_matrix = train_matrix/5 - 0.7
    train_matrix[train_matrix == -0.7] = 0
    test_matrix = test_matrix/5 - 0.7
    test_matrix[test_matrix == -0.7] = 0
    return train_matrix, test_matrix

def fill_zeros(train_matrix, val_new=0.6):
    # fill zeros with 0.6
    matrix = train_matrix.copy()
    matrix[matrix==0] = val_new
    return matrix

def noisy(train_matrix):
    # matrix = train_matrix.copy()
    matrix = np.random.normal(loc=0.5, scale=0.5, size=train_matrix.shape)
    np.copyto(matrix, train_matrix, where=train_matrix!=0)
    matrix=np.clip(matrix, 0.2, 1.0)
    return matrix

def noisy_labels(train_matrix, scale=0.1):
    # matrix = train_matrix.copy()
    matrix = np.random.normal(loc=0, scale=0.1, size=train_matrix.shape)
    matrix+=train_matrix
    matrix=np.around(np.clip(matrix, 0, 1.0),3)
    return matrix

def boltzmann_machine(train_matrix, n_comp, learning_rate=0.06,n_iter=20):
    from sklearn.neural_network import BernoulliRBM
    rbm = BernoulliRBM(n_components=n_com, learning_rate=learning_rate, n_iter=n_iter)
    rbm_transformed = rbm.fit_transform(train_matrix)
    print("successful RBM transform", rbm_transformed.shape)
    return rbm_transformed
