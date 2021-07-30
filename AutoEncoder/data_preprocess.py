import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import pickle


def get_datasplit(val_ids, data_path = "../data/"):
    data = pd.read_csv(data_path + "structured_data_train.csv")
    data.set_index("sample_id", inplace=True)
    train_ids = np.setdiff1d(data.index.values, val_ids)
    data = np.asarray(data)
    return data[train_ids], data[val_ids]

def load_data(path):
    # ## Read data
    data = pd.read_csv(path)
    data = np.asarray(data)
    np.random.shuffle(data)
    # mean_rating = np.mean(data[:,1])/5
    return data

def construct_matrix(train_inds, test_inds):
    train_inds = np.loadtxt(train_inds, delimiter=',').astype(int)
    test_inds = np.loadtxt(test_inds, delimiter=',').astype(int)
    omega = np.asarray(pd.read_csv(data))
    omega = omega[omega[:,0], 1:] # first column of omega as inds
    omega_train = omega[train_inds]
    omega_test = omega[test_inds]
    train_matrix = matrix_data_omega(omega_train)
    test_matrix = matrix_data_omega(omega_test)
    return train_matrix, test_matrix

def split_data(data, split=10):
    inds = np.random.permutation(len(data))
    test_inds = inds[:len(data)//split]
    train_inds = inds[len(data)//split:]

    # matrix = np.zeros((10000,1000))
    train_matrix = np.zeros((10000,1000))
    test_matrix = np.zeros((10000,1000))

    for entry in data[train_inds]:
        field = entry[0].split("_")
        row = int(field[0][1:])
        col = int(field[1][1:])
        # matrix[row-1, col-1] = entry[1]
        train_matrix[row-1, col-1] = entry[1]

    for entry in data[test_inds]:
        field = entry[0].split("_")
        row = int(field[0][1:])
        col = int(field[1][1:])
        # matrix[row-1, col-1] = entry[1]
        test_matrix[row-1, col-1] = entry[1]

    print("nonzeros:", "train:", np.count_nonzero(train_matrix), "test:", np.count_nonzero(test_matrix)) # all:", np.count_nonzero(matrix),
    assert(not np.any(np.logical_and(train_matrix, test_matrix)))
    return train_matrix, test_matrix

def load_data_omega(path="../../omega.npy"):
    omega = np.load(path)
    inds = np.random.permutation(len(omega))
    test_inds = inds[:len(omega)//10]
    train_inds = inds[len(omega)//10:]

    # ## Put into train and test matrix
    omega_train = omega[train_inds]
    omega_test = omega[test_inds]
    print(len(omega_train), len(omega_test))
    return omega_train, omega_test

def matrix_data_omega(omega):
    matrix = np.zeros((10000,1000))
    for row, col, entry in omega:
        matrix[row, col] = entry
    # assert(not np.any([np.all(r==0) for r in matrix]))
    return matrix

def submission_omega(submission):
    omega_sub = []
    for entry in submission:
        field = entry[0].split("_")
        row = int(field[0][1:])
        col = int(field[1][1:])
        omega_sub.append([row-1, col-1])
    omega_sub = np.asarray(omega_sub)
    return omega_sub

def make_submission(res_matrix, path="cil-collab-filtering-2019/sampleSubmission.csv", out_path="submission.csv"):
    submission = pd.read_csv(path)
    submission_arr = np.asarray(submission)
    omega_sub = submission_omega(submission_arr)
    out_list = []
    for row in omega_sub:
        pred = (res_matrix[row[0], row[1]]*5).clip(min=1, max=5)
        out_list.append(float(pred))
    submission["Prediction"] = out_list
    submission.to_csv(out_path, index=False)
