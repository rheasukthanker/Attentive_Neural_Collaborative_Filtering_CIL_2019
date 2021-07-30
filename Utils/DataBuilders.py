import pandas as pd
import abc
from scipy.sparse import csr_matrix as sp_matrix
import numpy as np
import numba

class AbstractDataBuilder(metaclass=abc.ABCMeta):
    """Each Subclass of AbstractEmbedding has a AbstractDataBuilder associated."""

    @abc.abstractmethod
    def set_data(self, training_data):
        """
        present_training_ids and present_new can only be caled after this method was called once

        :param training_data: pandas Data Frame with columns 'sample_id,row_id,col_id,prediction' where sample_id is the index column
        """
        raise NotImplementedError("Needs to be overwritten")

    @abc.abstractmethod
    def present_all_training_samples(self):
        """
        It is guaranteed that set_data was called before this method is called for the first time

        present all samples that were previously set with set_data (should be in the same order as what was past to set_data)

        :return: (X, label), where
            X is a tuple of numpy arrays of shape [len(ids), <...>] where <...> is the shape that the corresponding embedding expects per sample.
            label is a numpy array of shape [len(ids)] and values in {1...5} with the label
        """
        raise NotImplementedError("Needs to be overwritten")

    @abc.abstractmethod
    def present_new(self, data):
        """
        It is guaranteed that set_data was called before this method is called for the first time

        :param data: pandas Data Frame with columns 'row_id,col_id,prediction'. this is new data that should be presented
        :return:  (X, label), where
            X is a tuple of numpy arrays of shape [len(ids), <...>] where <...> is the shape that the corresponding embedding expects per sample.
            label is a numpy array of shape [len(ids)] and values in {1...5} with the label
        """
        raise NotImplementedError("needs to be implemented error")


class SimpleEmbeddingDataBuilder(AbstractDataBuilder):

    def set_data(self, training_data):
        self.data = training_data

    def present_all_training_samples(self):
        X, label = self._make_sample_triple_from_pd(self.data)
        return X, label

    def present_new(self, data):
        X, label = self._make_sample_triple_from_pd(data)
        return X, label

    def _make_sample_triple_from_pd(self, df):
        row_ids = df.row_id.values
        col_ids = df.col_id.values
        label = df.prediction.values
        return (row_ids, col_ids), label


class AttentionEmbeddingDataBuilder(AbstractDataBuilder):

    def __init__(self, n_context_movies_train, n_contect_movies_predict):

        self.n_context_movies_train = n_context_movies_train
        self.n_context_movies_predict = n_contect_movies_predict

    def set_data(self, training_data):


        self.row_ids = training_data.row_id.values.astype(int)
        self.col_ids = training_data.col_id.values.astype(int)
        self.label = training_data.prediction.values

        matrix = sp_matrix((self.label, (self.row_ids, self.col_ids)), dtype=int)

        self.print_matrix_stats(matrix)


        self.other_watched_movie_ids, self.other_watched_movie_ratings =  self.csr_matrix2three_np(matrix)

    def print_matrix_stats(self, matrix):
        min_context = self.min_n_context_movies(matrix)
        max_context = self.max_n_context_movies(matrix)

        print("Min n movies per user {}, max n movies per user {}".format(min_context, max_context))

    def max_n_context_movies(self, matrix):
        n_movies_per_user = matrix.getnnz(axis=1)
        return np.max(n_movies_per_user)

    def min_n_context_movies(self, matrix):
        n_movies_per_user = matrix.getnnz(axis=1)
        return np.min(n_movies_per_user)

    def present_all_training_samples(self):
        X = self.make_input_x(user_ids=self.row_ids, movie_ids=self.col_ids, num_context_movies=self.n_context_movies_train)
        label = self.label
        return X, label

    def present_new(self, data):
        user_ids = data.row_id.values
        movie_ids = data.col_id.values

        X = self.make_input_x(user_ids=user_ids, movie_ids=movie_ids, num_context_movies=self.n_context_movies_predict)
        label = data.prediction.values
        return X, label

    def make_input_x(self, user_ids, movie_ids, num_context_movies):
        return _np_pos_and_other_columns(other_movie_ids=self.other_watched_movie_ids,
                                         other_movie_ratings=self.other_watched_movie_ratings,
                                         target_user_ids=user_ids, target_movie_ids=movie_ids,
                                         num_context_movies=num_context_movies)


    def csr_matrix2three_np(self, matrix):
        n_users, n_movies = matrix.shape

        n_context_movies = self.max_n_context_movies(matrix)

        other_movie_ids = -1 * np.ones((n_users, n_context_movies), dtype=int)
        other_movie_ratings = -1 * np.ones((n_users, n_context_movies), dtype=int)
        n_other_movies_per_row = np.zeros(n_users)
        for user_id in range(n_users):
            other_movies_rated_by_user = matrix.getrow(user_id)

            other_movie_ids_this_user = other_movies_rated_by_user.indices
            other_movie_ratings_this_user = other_movies_rated_by_user.data

            n_other_movies_this_user = len(other_movie_ratings_this_user)

            assert(n_other_movies_this_user == len(other_movie_ids_this_user))
            assert(n_other_movies_this_user <= n_context_movies)

            n_other_movies_per_row[user_id] = n_other_movies_this_user

            other_movie_ids[user_id, :n_other_movies_this_user] = other_movie_ids_this_user
            other_movie_ratings[user_id, :n_other_movies_this_user] = other_movie_ratings_this_user


        return other_movie_ids, other_movie_ratings

@numba.jit(nopython=True, parallel=False)
def _np_pos_and_other_columns(other_movie_ids, other_movie_ratings, target_user_ids, target_movie_ids, num_context_movies):
    batchsize = target_user_ids.shape[0]
    batch_other_watched_ids = -1 * np.ones((batchsize, num_context_movies), dtype=np.int32)
    batch_other_watched_ratings = -1 * np.ones((batchsize, num_context_movies), dtype=np.int32)

    for s in range(batchsize):
        user_id = target_user_ids[s]
        target_movie_id = target_movie_ids[s]

        other_movies_rated_by_user_id = other_movie_ids[user_id]
        other_movies_rated_by_user_rating = other_movie_ratings[user_id]

        (usable_reference_ids,) = np.where(np.logical_and((other_movies_rated_by_user_id != target_movie_id), (other_movies_rated_by_user_id != -1)))

        n_other_movies = len(usable_reference_ids)
        if n_other_movies <= num_context_movies: # we have to use all ratings
            reference_ids_to_use = usable_reference_ids
        else:
            # we have to select a subset to use
            reference_ids_to_use = np.random.choice(usable_reference_ids, num_context_movies, replace=False)

        other_movies_rated_by_user_global_ids = other_movies_rated_by_user_id[reference_ids_to_use]
        other_movies_rated_by_user_ratings = other_movies_rated_by_user_rating[reference_ids_to_use]


        batch_other_watched_ids[s, 0:n_other_movies] = other_movies_rated_by_user_global_ids
        batch_other_watched_ratings[s, 0:n_other_movies] = other_movies_rated_by_user_ratings


    return target_user_ids, target_movie_ids, batch_other_watched_ids, batch_other_watched_ratings
