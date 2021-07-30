import numpy as np
import pandas as pd
import pickle
import os

class DataSplit():
    def __init__(self, split=10, shuffle=False, seed=123, data_folder=None):
        if data_folder is None:
            data_folder = os.path.join(os.getcwd(), "../data")

        self.data_folder = data_folder
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.out_list = self.reorder_dataset(split, data=os.path.join(data_folder, "structured_data_train.csv"))

    def reorder_dataset(self, split=10, data="data/structured_data_train.csv", nr_users =10000):
        omega = np.asarray(pd.read_csv(data))
        grouped_list = []
        np.random.seed(self.seed)
        for u in range(nr_users):
            samples = (np.where(omega[:,1]==u)[0]).tolist()
            if self.shuffle:
                np.random.shuffle(samples)
            grouped_list.append(samples)
        out_list = []
        for i in range(split):
            split_list = []
            for u in range(nr_users):
                user_ratings = grouped_list[u]
                batch = (len(user_ratings)//split)
                if i==split-1:
                    split_list.extend(user_ratings[i*batch:])
                else:
                    split_list.extend(user_ratings[i*batch:(i+1)*batch])
            out_list.append(split_list)
        return out_list

    def get_validation(self, val_part=1):
        return self.out_list[val_part]

    # @staticmethod
    def pickle(self, outfilename = "split.pickle"):
        outpath = os.path.join(self.data_folder, outfilename)

        with open(outpath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pickle(filename, path_to_folder=None):
        if path_to_folder is None:
            path_to_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

        filepath = os.path.join(path_to_folder, filename)

        with open(filepath, "rb") as f:
            return pickle.load(f)



if __name__=="__main__":
    # we only call this once and share the split over github

    s10 = DataSplit(10, shuffle=True)
    # lists = s10.out_list
    DataSplit.pickle(s10)
    # with open("../data/s10_lists2.pickle", "wb") as f:
    #     pickle.dump(lists, f)

    rec = DataSplit.load_from_pickle("s10.pickle")

    print(rec)
