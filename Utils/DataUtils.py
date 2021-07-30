import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
from scipy.sparse import csr_matrix as sp_matrix
import time
import argparse

def read_data_original(path="data_train.csv"):
    d = pd.read_csv(path)
    row_ids = list()
    column_ids = list()
    predictions = list()
    n = len(d)
    #     n=100
    for i in range(n):
        loc_string = d.iloc[i].loc["Id"]
        pred = d.iloc[i].loc["Prediction"]
        r, c = loc_string.split("_")
        row_id = int(r[1:])
        col_id = int(c[1:])
        row_ids.append(row_id)
        column_ids.append(col_id)
        predictions.append(pred)

        if (i % 10000) == 0:
            print(i / n)

    d = pd.DataFrame(data=dict(row_id=row_ids, col_id=column_ids, prediction=predictions))
    d.index.names = ["sample_id"]
    return d

def convert_original_to_structured_data_frame(original_path, output_path):
    d = read_data_original(original_path)
    d.col_id -= 1
    d.row_id -= 1

    d.to_csv(output_path)

def get_n_movies_users(data):
    n_movies = np.max(data.col_id.values) + 1
    n_user = np.max(data.row_id.values) + 1
    return n_movies, n_user

def structured_data_to_original(structured):
    """"structured is pd data frame"""
    assert (structured.columns.values.tolist() == ['row_id', 'col_id', 'prediction'])
    structured.col_id += 1
    structured.row_id += 1


class DataIter:
    @property
    def n_cols(self):
        return np.max(self.data.col_id.values) + 1

    @property
    def n_rows(self):
        return np.max(self.data.row_id.values) + 1

    def __init__(self, data_builder, data, sample_weightings=None, print_timings=False):
        """
        :param data: padnas dataframe with columns sample_id(index), row_id, col_id, prediction
        """
        self.print_timings = print_timings
        assert(data.columns.values.tolist() == ['row_id', 'col_id', 'prediction'])
        self.data = data.astype(int)
        self.data_builder = data_builder

        self.set_validation_indices_has_to_be_called_first = True

        self.sample_weightings = sample_weightings

    def set_validation_indices(self, val_ids):
        """
        :param ids: numpy array of shape [n_val_samples] with the ids (corresponding to data.sample_id) for ids to use for validation
        """
        assert(np.all(np.isin(val_ids, self.data.index.values)))

        self.set_validation_indices_has_to_be_called_first = False
        print("Validation indices set")

        self.validation_ids = val_ids
        self.training_ids = np.setdiff1d(self.data.index.values, val_ids)
        t0 = time.time()
        self.data_builder.set_data(self.data.loc[self.training_ids])
        if self.print_timings:
            print("Set Data Took {} seconds".format(time.time()-t0))

        assert(len(self.validation_ids) + len(self.training_ids) == len(self.data))


    def complete_training_data(self):
        assert(not self.set_validation_indices_has_to_be_called_first)

        return self.data_builder.present_all_training_samples()

    def complete_validation_data(self):
        assert(not self.set_validation_indices_has_to_be_called_first)

        return self.data_builder.present_new(self.data.loc[self.validation_ids])



    def training_epoch_iter(self, n_epochs, batchsize, shuffle):
        """
        Iterators for training

        :param n_epochs:
        :param batchsize:
        :param shuffle:
        :return:
        """
        assert(not self.set_validation_indices_has_to_be_called_first)

        if self.sample_weightings is not None:
            training_data = self.data.loc[self.training_ids]

            concrete_weightings = self.sample_weightings.get_batch_weights(user_ids=training_data.loc[:, "row_id"],
                                                                         item_ids= training_data.loc[:, "col_id"],
                                                                         ratings=training_data.loc[:, "prediction"])
        else:
            concrete_weightings=None

        for epoch_id in range(n_epochs):
            t0 = time.time()
            data, label = self.data_builder.present_all_training_samples()
            t1 = time.time()
            if self.print_timings:
                print("The Data iter took {} seconds to prepare the trainig data".format(t1-t0))
            yield self._generic_batch_iter(data=data, label=label, batchsize=batchsize, shuffle=shuffle, weightings=concrete_weightings)

    def prediction_iter(self, new_samples_pd, batchsize):
        """present new samples"""
        assert(not self.set_validation_indices_has_to_be_called_first)

        t0 = time.time()
        data, label = self.data_builder.present_new(new_samples_pd)
        t1 = time.time()
        if self.print_timings:
            print("The data builder took {} seconds to prepare the prediction data".format(t1-t0))

        return self._generic_batch_iter(data, label, batchsize=batchsize, shuffle=False)

    def validation_iter(self, batchsize):
        validation_samples = self.data.loc[self.validation_ids]

        return self.prediction_iter(new_samples_pd=validation_samples, batchsize=batchsize)

    def _concatenate_to_np_matrix(self, X, label, weightings=None):
        """

        :param X: a tuple of numpy arrays of shape [batchsize, <...other_dims...>]
        :param label: a numpy array of shape [batchsize]
        :return:
        """

        X_np = list()
        id_ranges = [0]

        list_of_stuff_to_code = X + (label,)
        if weightings is not None:
            list_of_stuff_to_code += (weightings,)

        for elem in list_of_stuff_to_code:
            if len(elem.shape) == 1:
                X_np.append(np.expand_dims(elem, 1))

                id_ranges.append(id_ranges[-1]+1)
            elif len(elem.shape) == 2:
                X_np.append(elem)
                id_ranges.append(id_ranges[-1]+elem.shape[1])

        concatenated = np.concatenate(X_np, axis=1)

        assert(concatenated.shape[1] == id_ranges[-1])
        return concatenated, id_ranges

    def _np_matrix_to_tuple(self, X_np, id_ranges, has_weightings):
        tuple_collector = tuple()
        for i in range(len(id_ranges)-1):
            tuple_collector += (np.squeeze(X_np[:, id_ranges[i]: id_ranges[i+1]]) ,)

        if has_weightings:
            X = tuple_collector[:-2]
            label = tuple_collector[-2]
            weightings = tuple_collector[-1]
        else:
            X = tuple_collector[:-1]
            label = tuple_collector[-1]
            weightings = None
        return X, label, weightings

    def _generic_batch_iter(self, data, label, batchsize, shuffle, allow_smaller_last_batch=True, weightings=None):
        # data is a tuple of numpy arrays

        t0 = time.time()
        np_matrix, id_ranges = self._concatenate_to_np_matrix(data, label, weightings=weightings)

        num_batches_per_epoch = int((len(label) - 1) / batchsize) #rounded down number of batches

        if allow_smaller_last_batch:
            num_batches_per_epoch += 1 # one more smaller batch

        if shuffle:
            shuffled_ids = np.random.permutation(len(label))
            shuffled_np_matrix = np_matrix[shuffled_ids]
        else:
            shuffled_np_matrix = np_matrix

        has_weightings = weightings is not None
        t1 = time.time()
        t_overhead = t1-t0

        total_time_batches = 0
        for b_id in range(num_batches_per_epoch):
            t0 = time.time()
            start_id = b_id * batchsize
            end_id = min((b_id+1)*batchsize, len(label))

            batch_data = shuffled_np_matrix[start_id: end_id]

            batch_X, batch_label, weightings = self._np_matrix_to_tuple(batch_data, id_ranges, has_weightings=has_weightings)
            t1 = time.time()

            total_time_batches += (t1-t0)

            yield batch_X, batch_label, weightings

        if self.print_timings:
            print("Preperation overhead: {} Total time for batch generation {}".format(t_overhead, total_time_batches))

def avg_csv_files(exp_dir, submssion_folders, filename="submission.csv"):

    collector = list()

    for folder in submssion_folders:
        full_path = os.path.join(exp_dir, folder, filename)
        df = pd.read_csv(full_path)

        subm = df.Prediction.values

        collector.append(np.expand_dims(subm, 0))

    all = np.concatenate(collector, axis=0)
    print(all.shape)

    avg_sub = np.mean(all, axis=0)

    df.loc[:, "Prediction"] = avg_sub

    df.set_index("Id", inplace=True)
    df.to_csv("averaged_submissions.csv")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    action = parser.add_mutually_exclusive_group()
    action.add_argument("--average", action="store_true")
    action.add_argument("--process_data", action="store_true")

    parser.add_argument("--submission_folders", type=str, nargs="*")
    parser.add_argument("--exp_dir", type=str)

    parser.add_argument("--raw_csv", type=str)
    parser.add_argument("--out", type=str)

    args = parser.parse_args()

    if args.average:
        avg_csv_files(exp_dir = args.exp_dir, submssion_folders=args.submission_folders)

    if args.process_data:
        convert_original_to_structured_data_frame(args.raw_csv, args.out)


