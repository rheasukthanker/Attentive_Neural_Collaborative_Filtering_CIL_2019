#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:04:36 2019

@author: rheasukthanker
"""

import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split


def main():
    data = pd.read_csv("data/structured_data_train.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.iloc[0:100000]
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(data[['row_id', 'col_id', 'prediction']],
                                reader)
    benchmark = []
    # Iterate over all algorithms
    for algorithm in [
            SVD(),
            SVDpp(),
            SlopeOne(),
            NMF(),
            NormalPredictor(),
            KNNBaseline(),
            KNNBasic(),
            KNNWithMeans(),
            KNNWithZScore(),
            BaselineOnly(),
            CoClustering()
    ]:
        # Perform cross validation
        results = cross_validate(algorithm,
                                 data,
                                 measures=['RMSE'],
                                 cv=10,
                                 verbose=False)
        print(results)
        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(
            pd.Series([str(algorithm).split(' ')[0].split('.')[-1]],
                      index=['Algorithm']))
        benchmark.append(tmp)
    surprise_results = pd.DataFrame(benchmark).set_index(
        'Algorithm').sort_values('test_rmse')
    print(surprise_results.head())
    surprise_results.to_pickle("baselines_results.pkl")


if __name__ == "__main__":
    main()
