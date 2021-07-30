import unittest
from Utils.DataUtils import *
from Utils.DataBuilders import *
import numpy as np

class TestDataUtils(unittest.TestCase):


    def setUp(self):
        data = pd.read_csv("../data/structured_data_train.csv")

        data.set_index("sample_id", inplace=True)

        train_subset_size = 300

        self.subset_train = data[(data.row_id<train_subset_size) & (data.col_id<train_subset_size)]


        self.data_iter_simple = DataIter(SimpleEmbeddingDataBuilder(), self.subset_train, print_timings=True)


        self.n_context_training = 32
        self.n_context_predict = 64
        self.attention_db = AttentionEmbeddingDataBuilder(n_contect_movies_predict=self.n_context_predict, n_context_movies_train=self.n_context_training)
        self.data_iter_attention = DataIter(self.attention_db, self.subset_train, print_timings=True)

        all_ids = self.subset_train.index.values

        self.n_validation = 90
        self.n_training = len(self.subset_train)-self.n_validation

        self.val_ids = all_ids[np.random.choice(len(all_ids), 90, replace=False)]
        self.validation_pd = self.subset_train.loc[self.val_ids]


        self.data_iter_simple.set_validation_indices(self.val_ids)
        self.data_iter_attention.set_validation_indices(self.val_ids)

        data_test = pd.read_csv("../data/structured_sample_submission.csv")
        data_test.set_index('sample_id', inplace=True)
        # test_subset_size = 100
        self.subset_test = data_test[(data_test.row_id < train_subset_size) & (data_test.col_id < train_subset_size)]
        #
        #
        # n_max_context = self.dh.max_n_context_movies

        # t = d.read_data("/data/sentences.eval")

        print("")

    def test_presenting_test_data(self):

        new_attention_db = AttentionEmbeddingDataBuilder(n_context_movies_train=self.n_context_training, n_contect_movies_predict=self.n_context_predict)
        new_attention_db.set_data(self.subset_train)

        input_data, label = new_attention_db.present_new(self.subset_test)

        row_ids, col_ids, other_watched_col_ids, other_watched_ratings = input_data

        samples_to_test = 100
        for s in range(len(row_ids)):
            if s > samples_to_test:
                break
            target_col_id = col_ids[s]
            user = row_ids[s]

            other_watched = other_watched_col_ids[s, :]
            self.assertFalse(target_col_id in other_watched) # make sure that the target movie is not accidently given to network

            for other_watched_movie_id, other_watched_movie_rating in zip(other_watched_col_ids[s], other_watched_ratings[s]):
                if other_watched_movie_id==-1: # rest is padding
                    break
                p = self._get_prediction_by_row_col(user, other_watched_movie_id) # make sure other watched movie has correct prediction
                self.assertEqual(p, other_watched_movie_rating)

                # make sure other watched context movie is not in test set
                id_context_movie_in_test = self._get_prediction_by_row_col(user, other_watched_movie_id, pandas_frame_to_get_from=self.subset_test)
                self.assertTrue(id_context_movie_in_test == -1)



    def test_epoch_iter_other_watched_movies(self):
        n_batches_to_test=23

        for epoch in self.data_iter_attention.training_epoch_iter(batchsize=23, n_epochs=1, shuffle=True):# .epoch_iter(batch_size=23, n_epochs=1, shuffle=True, allow_smaller_last_batch=True, num_col_samples=10):
            for b_id, batch in enumerate(epoch):
                if b_id >= n_batches_to_test:
                   return
                input_data, label, weight = batch
                row_ids, col_ids, other_watched_col_ids, other_watched_ratings = input_data
                for s in range(len(row_ids)):
                    target_col_id = col_ids[s]
                    user = row_ids[s]

                    other_watched = other_watched_col_ids[s, :]
                    self.assertFalse(target_col_id in other_watched) # make sure that the target movie is not accidently given to network

                    for other_watched_movie_id, other_watched_movie_rating in zip(other_watched_col_ids[s], other_watched_ratings[s]): #make sure the other watched movies have correct label
                        if other_watched_movie_id==-1:
                            break
                        p = self._get_prediction_by_row_col(user, other_watched_movie_id)
                        self.assertEqual(p, other_watched_movie_rating)



                print()

    def _get_prediction_by_row_col(self, row, col, pandas_frame_to_get_from = None):
        if pandas_frame_to_get_from is None:
            pandas_frame_to_get_from = self.subset_train

        row_filter = pandas_frame_to_get_from.row_id.values == row
        col_filter = pandas_frame_to_get_from.col_id.values == col

        pred = pandas_frame_to_get_from[(row_filter & col_filter)].prediction.values
        if len(pred)==1:
            return pred[0] #found
        else:
            return -1 #not found

    def test_epoch_iter(self):
        all_presented_data_pairs = set()
        for epoch in self.data_iter_simple.training_epoch_iter(batchsize=23, shuffle=True, n_epochs=1):
            for batch in epoch:
                (row_ids, col_ids), label, weight = batch
                # sample to check that I didn't mix up label
                check_pos= 12
                for i  in range(len(row_ids)):
                    row_id = row_ids[ i]
                    col_id = col_ids[i]
                    pred = label[i]

                    from_table = self._get_prediction_by_row_col(row=row_id, col=col_id)
                    self.assertEqual(from_table, pred)

                    from_validation = self._get_prediction_by_row_col(row=row_id, col=col_id, pandas_frame_to_get_from=self.validation_pd )
                    self.assertEqual(from_validation, -1) # make sure sample is not in validaiton data


                # add all presented pairs
                pairs = zip(row_ids, col_ids)
                all_presented_data_pairs.update(pairs)

        self.assertEqual(len(all_presented_data_pairs), self.n_training) # test that all training pairs have been presented


    @unittest.skip("function deprecated")
    def test_all_data(self):
        data, label = self.dh.all_data()

        self.assertTrue(np.all(data[:,0]==self.subset_train.row_id.values))
        self.assertTrue(np.all(data[:,1]==self.subset_train.col_id.values))
        self.assertTrue(np.all(label==self.subset_train.prediction.values))


