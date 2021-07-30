#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import regex as re
from sklearn.metrics import mean_squared_error
from math import sqrt
from Utils.DataSplits import DataSplit


def get_matrix_on_samples(subset):
    mat = np.zeros([10000, 1000])
    for i in range(0, subset.shape[0]):
        mat[subset.iloc[i, 1], subset.iloc[i, 2]] = subset.iloc[i, 3]
    return mat


def generate_movie_imputed(data):
    mean_per_movie = np.sum(data, axis=0) / np.sum(data != 0, axis=0)
    #print(mean_per_movie)
    temp = data.copy()
    for i in range(1000):
        temp[temp[:, i] == 0, i] = mean_per_movie[i]
    data_movie_imputed = temp
    where_are_NaNs = np.isnan(data_movie_imputed)
    data_movie_imputed[where_are_NaNs] = 0
    return data_movie_imputed


def return_svd(imputed_data, factors,i):
    U, d, Vt = np.linalg.svd(
        imputed_data)  #shapes U=10000*10000 , d=1000, V=1000*1000
    D = np.zeros([10000, 1000])
    D[range(0, 1000), range(0, 1000)] = d
    U_k = U[:, 0:factors]
    D_k = D[0:factors, 0:factors]
    Vt_k = Vt[0:factors, :]
    preds = np.matmul(np.matmul(U_k, D_k), Vt_k)
    svd_save_embeddings(U_k, Vt_k.T,i,factors)
    return preds


def clip_predictions(preds):
    preds[preds > 5] = 5
    preds[preds < 1] = 1
    return preds


def predict(test, preds_clipped):
    preds_val = []
    for i in range(0, test.shape[0]):
        preds_val.append(preds_clipped[test.iloc[i, 1], test.iloc[i, 2]])
    return preds_val


def svd_save_embeddings(U_k, V_k,i,factors):
    
    np.save("svd_user_embedding_"+str(factors)+"_"+str(i)+".npy", U_k)
    np.save("svd_movies_embedding_"+str(factors)+"_"+str(i)+".npy", V_k)


def main():
    data = pd.read_csv("data/structured_data_train.csv")
    #print(data.shape)
    split = DataSplit.load_from_pickle("s10.pickle")
    cross_val = []
    for i in range(0, 10):
        val_inds = split.get_validation(i)
        #print(val_inds)
        val_data = data.iloc[val_inds, :]
        data_new = data.drop(data.index[val_inds])
        factors=128
        #factors = 10 
        mat = get_matrix_on_samples(data_new)
        #print(mat)
        imputed_mat = generate_movie_imputed(mat)
        #print(imputed_mat)
        preds = return_svd(imputed_mat, factors,i)
        preds_clipped = clip_predictions(preds)
        test = val_data
        actual_preds = test.iloc[:, 3]
        preds_val = predict(test, preds_clipped)
        mean_rmse = sqrt(mean_squared_error(actual_preds, preds_val))
        print("MSE", mean_rmse)
        cross_val.append(mean_rmse)
    print("10 fold cv score is", np.mean(cross_val))


if __name__ == "__main__":
    main()
