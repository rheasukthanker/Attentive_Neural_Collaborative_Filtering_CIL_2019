import abc
import tensorflow as tf
import numpy as np
import os

class AbstractEmbedding(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, embedding_size, n_movies, n_user, *args, **kwargs):
        """
        :param embedding_size:
        """
        raise NotImplementedError("has to be overwritten")

    @abc.abstractmethod
    def get_user_embedding(self):
        """returns a tensorflow tensor of shape [batchsize, embedding_size] (that is [?, embedding_size]"""
        raise NotImplementedError("has to be overwritten")

    @abc.abstractmethod
    def get_movie_embedding(self):
        """returns a tensorflow tensor of shape [batchsize, embedding_size] (that is [?, embedding_size]"""
        raise NotImplementedError("has to be overwritten")

    @abc.abstractmethod
    def make_input_feeddict(self, X):
        """
        :param X: is the tuple returned by the corresponding subclass of AbstractDataBuilder
        :return: a tensorflow feed dict set appropriatly
        """
        raise NotImplementedError("has to be overwritten")

    def get_additional_val_summaries(self):
        if hasattr(self, 'val_summaries'):
            return self.val_summaries
        else:
            return []

    def add_val_summary(self, sum):
        if hasattr(self, 'val_summaries'):
            self.val_summaries.append(sum)
        else:
            self.val_summaries = [sum]

class SimpleEmbedding(AbstractEmbedding):

    def __init__(self, embedding_size, n_movies, n_user, batch_size=None):
        self.user_indxs = tf.placeholder(tf.int32, [batch_size], name="input_user_indices")
        self.movie_indxs = tf.placeholder(tf.int32, [batch_size], name="input_movie_indices")

        with tf.name_scope("embedding"):
            self.movie_embedding_w = tf.get_variable("row_embedding", [n_movies, embedding_size],
                                                     regularizer=tf.contrib.layers.l2_regularizer(0.5),
                                                     dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            self.user_embedding_w = tf.get_variable("col_embedding", [n_user, embedding_size], dtype=tf.float32,
                                                    regularizer=tf.contrib.layers.l2_regularizer(0.5),
                                                    initializer=tf.contrib.layers.xavier_initializer())


            self.final_user_embeddings = tf.nn.embedding_lookup(self.movie_embedding_w, self.movie_indxs)  # shape [batchsize, seqlength, embed_dim]
            self.final_movie_embeddings = tf.nn.embedding_lookup(self.user_embedding_w, self.user_indxs)

    def get_user_embedding(self):
        return self.final_user_embeddings

    def get_movie_embedding(self):
        return self.final_movie_embeddings

    def make_input_feeddict(self, X):
        (user_ids, movie_ids) = X
        d = {self.user_indxs: user_ids, self.movie_indxs: movie_ids}
        return d

class AutoencoderEmbedding(AbstractEmbedding):

    def __init__(self, embedding_size, n_movies, n_user, split_nr, batch_size=None, weights_path = "Weights/ae_embeddings/"):
        self.user_indxs = tf.placeholder(tf.int32, [batch_size], name="input_user_indices")
        self.movie_indxs = tf.placeholder(tf.int32, [batch_size], name="input_movie_indices")
        use_embed = np.load(os.path.join(weights_path, "user_embed_ae_"+str(split_nr)+".npy")).astype(np.float32)
        mov_embed = np.load(os.path.join(weights_path, "movie_embed_ae_"+str(split_nr)+".npy")).astype(np.float32)
        assert(embedding_size==use_embed.shape[1] and embedding_size==mov_embed.shape[1])
        self.user_embedding_w = tf.get_variable("row_embedding", dtype=tf.float32,
                                                regularizer=tf.contrib.layers.l2_regularizer(0.5),
                                                initializer=use_embed) # tf.Variable(use_embed)
        self.movie_embedding_w =  tf.get_variable("col_embedding", dtype=tf.float32,
                                                regularizer=tf.contrib.layers.l2_regularizer(0.5),
                                                initializer=mov_embed)# tf.Variable(mov_embed)
        self.final_user_embeddings = tf.nn.embedding_lookup(self.movie_embedding_w, self.movie_indxs)  # shape [batchsize, seqlength, embed_dim]
        self.final_movie_embeddings = tf.nn.embedding_lookup(self.user_embedding_w, self.user_indxs)

    def get_user_embedding(self):
        return self.final_user_embeddings

    def get_movie_embedding(self):
        return self.final_movie_embeddings

    def make_input_feeddict(self, X):
        (user_ids, movie_ids) = X
        d = {self.user_indxs: user_ids, self.movie_indxs: movie_ids}
        return d

class AbstractAttentionEmbedding(AbstractEmbedding):
    def __init__(self):
        self.padding_id = -1

        batch_size = None
        n_context_movies = None #just for debug to make the sizes inspectable

        with tf.name_scope("input"):
            self.user_indxs = tf.placeholder(tf.int32, [batch_size], name="input_user_indices")
            self.movie_indxs = tf.placeholder(tf.int32, [batch_size], name="input_movie_indices")
            self.indx_other_movies_watched_by_user = tf.placeholder(tf.int32, [batch_size, n_context_movies], name="indices_of_other_movies_watched_by_user")
            self.ratings_of_movies_watched_by_user = tf.placeholder(tf.int32, [batch_size, n_context_movies], name="ratings_of_other_movies_watched_by_user")

    def make_input_feeddict(self, X):
        user_ids, movie_ids, other_watched_ids, other_watched_ratings = X

        is_padding = (other_watched_ids == self.padding_id)
        n_padding = np.count_nonzero(is_padding, axis=1)
        assert(len(n_padding) == len(user_ids))

        feed_dict = {self.user_indxs: user_ids, self.movie_indxs: movie_ids, self.indx_other_movies_watched_by_user: other_watched_ids, self.ratings_of_movies_watched_by_user: other_watched_ratings}
        return feed_dict

    def _inner_product(self, a, b, axis_in_which_to_dot_product=-1):
        return tf.reduce_sum(tf.multiply(a, b), axis=axis_in_which_to_dot_product)


    def get_possibly_learnable_parameter_with_summary(self, flag, default_init_val, name, **kwargs):
        if type(flag) == str:
            arglist = flag.split("_")
            if arglist[0] !="train":
                raise ValueError("")

            if len(arglist) == 1:
                init_val = default_init_val
            else:
                init_val = float(arglist[1])

            var = tf.get_variable(name, [], dtype=tf.float32, trainable=True, **kwargs)

            sum = tf.summary.scalar(name=name, tensor=var)
            self.add_val_summary(sum)
        else:
            var = float(flag)

        return var

    def get_id_mask(self, ids):
        """
        :param ids:
        :return: a tensor of same shape as ids with True wherever ids is a proper id (not padding)
        """
        return tf.not_equal(ids, self.padding_id)

    def attention(self, query, keys, values, values_mask, default_vector=None, type="cosine"):
        """

        :param query:
        :param keys:
        :param values:
        :param values_mask:
        :param default_vector: the vector that should be used if there are no keys for a query
        :param type:
        :return:
        """

        print("Attention >>>")
        assert(len(query.shape) == 2) #batchsize, embedding_size
        assert(len(keys.shape) ==3) # batchsize,  n_keys, embedding_size
        assert(len(values.shape) ==3) # batchsize, n_keys, value_size
        assert(len(values_mask.shape) == 2) # batchsize, n_keys


        values = tf.cast(values, tf.float32)
        print("vals {}".format(values))

        if default_vector is None:
            default_vector_expanded = tf.zeros_like(values[:, 0, :])
            default_vector = tf.zeros_like(values[0, 0, :])
        else:
            default_vector_expanded = tf.ones_like(values[:, 0, :]) * tf.expand_dims(default_vector, axis=0)

        _logit_similarities = self.similarity_logits(query=query, keys=keys, type=type)

        logit_similarities = tf.identity(_logit_similarities)

        weightings, everything_was_masked = self.masked_softmax(logit_similarities, mask=values_mask, axis=-1)

        weighting_over_values = tf.expand_dims(weightings, axis=-1)

        combined_raw = tf.reduce_sum((weighting_over_values * tf.cast(values, tf.float32)), axis=1)

        combined = tf.where(everything_was_masked, default_vector_expanded, combined_raw)

        combined_final = tf.identity(combined)

        return combined_final, weightings


    def masked_softmax(self, logits, mask, axis):
        """

        :param logits:
        :param mask: mask should be FALSE if something has to be ignored
        :param axis:
        :return: p_dist, all_values_masked
            p_dist has same shape as logits and is a probability distribution (i.e. sums up to 1 along the given axis)
            all_values_masked has all dimensions the same as logits, expect the one indicating by axis is missing, wherever it is true that sample in the batch had all values masked, then the probability distribution will be constant zero
        """
        assert(len(logits.shape) == len(mask.shape))

        logits_min_max = logits - tf.reduce_max(logits, axis=axis, keep_dims=True)

        exps = tf.exp(logits_min_max)
        float_mask = tf.cast(mask, tf.float32)

        masked_exp = exps * float_mask

        normalizers = tf.reduce_sum(masked_exp, axis=axis)

        all_values_masked = tf.reduce_all(tf.logical_not(mask), axis=axis) # indicates, for which samples in the batch all values are masked. i.e. no context movie

        normalizers_no_zero = normalizers + tf.cast(all_values_masked, tf.float32) * 1 # add one to the normalizers that were completely masked to avoid deviding by zero, the result will be a 'prob distr' that is all zeros

        result = masked_exp / tf.expand_dims(normalizers_no_zero, axis=axis)

        return result, all_values_masked


    def similarity_logits(self, query, keys, type):
        if type=="dim_inner_prod":
            return self.dim_scaled_inner_prod_sim(query=query, keys=keys)
        elif type=="inner_prod":
            return self.inner_prod_sim(query=query, keys=keys)
        else:
            raise ValueError("No Attention type {}".format(type))

    def inner_prod_sim(self, query, keys):
        return self._inner_product(tf.expand_dims(query, 1), keys)

    def dim_scaled_inner_prod_sim(self, query, keys):
        d = tf.cast(query.shape[1], tf.float32)

        cosine_similarity = self._inner_product(tf.expand_dims(query, 1)
                                                , keys)

        scaled_sim = cosine_similarity / tf.sqrt(d)
        return scaled_sim

    def make_weighting_summary(self, weightings):
        print(weightings.shape)
        assert(len(weightings.shape)==2)

        exp_w = tf.expand_dims(tf.expand_dims(weightings, axis=-1), axis=0)

        self.image_summary = tf.summary.image(name="attention_weightings", tensor=exp_w)
        self.add_val_summary(self.image_summary)



class AttentionWeightedContextRaitings(AbstractAttentionEmbedding):

    def __init__(self, path_to_movie_embeddings, trainable=False, attention="cosine", image_sum=False, **kwargs):
        super(AttentionWeightedContextRaitings, self).__init__()


        mov_embed = np.load(path_to_movie_embeddings)


        self.movie_embedding_matrix = tf.Variable(mov_embed, trainable=trainable)

        self.final_movie_embedding_target = tf.nn.embedding_lookup(self.movie_embedding_matrix, self.movie_indxs)

        self.movie_embeddings_context = tf.nn.embedding_lookup(self.movie_embedding_matrix, tf.abs(self.indx_other_movies_watched_by_user))

        query = self.final_movie_embedding_target
        keys = self.movie_embeddings_context
        values = tf.expand_dims(self.ratings_of_movies_watched_by_user, -1)
        value_mask = self.get_id_mask(self.indx_other_movies_watched_by_user)

        defauld_vector = tf.get_variable("rating_unkown_user", [1], dtype=tf.float32, trainable=True, initializer=tf.constant_initializer([3.5]))

        self.predicted_rating, self.weightings_over_context = self.attention(query=query, keys=keys, values=values, values_mask=value_mask, type=attention, default_vector=defauld_vector)

        print("Predicted Rating shape >>{}".format(self.predicted_rating.shape))
        if image_sum:
            self.make_weighting_summary(self.weightings_over_context)

    def get_movie_embedding(self):
        return tf.squeeze(self.predicted_rating)

    def get_user_embedding(self):
        return tf.squeeze(self.predicted_rating)

class AttentionEmbedding(AbstractAttentionEmbedding):
    def get_movie_embedding(self):
        return self.final_movie_embeddings

    def get_user_embedding(self):
        return self.final_user_embeddings


    def __init__(self, n_users, n_movies, embedding_size, batchnorm=False,
                 image_sum=False, user_specific_embedding_strategy="add", is_training=False,
                 l2_reg_coefficient=0, attention_type="dim_inner_prod", lambda_mov_spec = 1,
                 query_key_scalling=1, context_key_matrix="same"):
        """
        :param n_users:
        :param n_movies:
        :param embedding_size:
        :param n_context_movies:
        :param batch_size:
        :param padding_id:
        :param user_specific_embedding_strategy: in ['add', 'append']
        :param weighting_trad_user weighting_movie_specific both are either a number or 'train'
        :param context_key_matrix: {same, seperate}

        Afterwards you can use self.final_user_embedding and self.final_movie_embedding in all kinds of models on top.
        """
        super(AttentionEmbedding, self).__init__()
        assert(user_specific_embedding_strategy in ['add', 'append'])

        self.batchnorm = batchnorm

        self.is_training = is_training

        if l2_reg_coefficient == 0:
            get_regularizer = lambda: None
            print("no regularisation")
        else:
            print("Regularizing in Attention with {}".format(l2_reg_coefficient))
            get_regularizer = lambda: tf.contrib.layers.l2_regularizer(l2_reg_coefficient)

        with tf.name_scope("embedding_weights"):
            if user_specific_embedding_strategy == "append":
                assert(embedding_size%2==0)
                user_part_embedding_size = embedding_size/2
            elif user_specific_embedding_strategy == "add":
                user_part_embedding_size=embedding_size
            else:
                raise ValueError("user_specific_embedding_strategy does not exist")

            if lambda_mov_spec != 1 or user_specific_embedding_strategy != "add":
                self.user_structure_embedding_u_weights = tf.get_variable("user_structure_embedding_u_weights", [n_users, user_part_embedding_size],
                                                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), regularizer=get_regularizer())

                self.user_structure_embedding_u = tf.nn.embedding_lookup(self.user_structure_embedding_u_weights, self.user_indxs)



            self.movie_structure_embedding_m_weights = tf.get_variable("movie_structure_embedding_u_weigths", [n_movies, embedding_size],
                                                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), regularizer=get_regularizer())
            self.movie_structure_embedding_m_target = tf.nn.embedding_lookup(self.movie_structure_embedding_m_weights, self.movie_indxs)

            if context_key_matrix == 'same':
                self.context_key_embedding_matrix = self.movie_structure_embedding_m_weights
            elif context_key_matrix == 'seperate':
                self.context_key_embedding_matrix = tf.get_variable("seperate_context_movie_embedding_weights", [n_movies, embedding_size],
                                                                    dtype=tf.float32,
                                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                                    regularizer=get_regularizer())
            else:
                raise ValueError("unkown value for context_key_matrix {}".format(context_key_matrix))


            self.movie_keys_embedding_m_context = tf.nn.embedding_lookup(self.context_key_embedding_matrix,
                                                                         tf.abs(self.indx_other_movies_watched_by_user)) # the tf.abs
            # the tf abs is a bit sneaky. we use -1 to pad if the movie has not watched enough movies previously. those padding movies now get resolved to the movie with id +1
            # but it doesn't matter since the corresponding positions later get zeroed out.

            self.movie_about_user_embedding_t_weights = tf.get_variable("movie_about_user_embedding_t_weights", [n_movies, user_part_embedding_size],
                                                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), regularizer=get_regularizer())

            self.movie_about_user_embedding_t  = tf.nn.embedding_lookup(self.movie_about_user_embedding_t_weights,
                                                                        tf.abs(self.indx_other_movies_watched_by_user))


        with tf.name_scope("target_specific_user_embedding"):

            self.query_key_scalar = self.get_possibly_learnable_parameter_with_summary(flag = query_key_scalling, default_init_val=1.0, name="query_key_scalar")

            values_mask = self.get_id_mask(self.indx_other_movies_watched_by_user)
            query = self.movie_structure_embedding_m_target * self.query_key_scalar
            keys = self.movie_keys_embedding_m_context * self.query_key_scalar

            values = tf.expand_dims(self.standardize_rating(self.ratings_of_movies_watched_by_user), axis=-1) * self.movie_about_user_embedding_t

            default_movie_about_user_embed = tf.get_variable("default_movie_about_user_embed", [user_part_embedding_size])

            self.what_movies_say_about_user_embedding, self.context_movie_attention_dist = self.attention(query=query, keys=keys, values=values, values_mask=values_mask, default_vector=default_movie_about_user_embed, type=attention_type)


            if user_specific_embedding_strategy == "add":

                if lambda_mov_spec == 1:
                    self.target_specific_user_embedding = self.what_movies_say_about_user_embedding
                else:
                    lambda_mov_spec_float = self.get_possibly_learnable_parameter_with_summary(flag = lambda_mov_spec, default_init_val=0.5, name="lambda_mov_specific", constraint=lambda t: tf.clip_by_value(t, 0, 1))
                    self.target_specific_user_embedding = (1-lambda_mov_spec_float)*self.user_structure_embedding_u + (lambda_mov_spec_float)*self.what_movies_say_about_user_embedding


            elif user_specific_embedding_strategy == "append":
                self.target_specific_user_embedding = tf.concat([self.user_structure_embedding_u, self.what_movies_say_about_user_embedding], axis=-1)

        if image_sum:
            self.make_weighting_summary(self.context_movie_attention_dist)

        if self.batchnorm:
            with tf.name_scope("normalisation"):
                final_user_embeddings = tf.layers.batch_normalization(self.target_specific_user_embedding, training=self.is_training)
                final_movie_embeddings = tf.layers.batch_normalization(self.movie_structure_embedding_m_target, training=self.is_training)
            print("using batch_normalisation")
        else:
            final_user_embeddings  = self.target_specific_user_embedding
            final_movie_embeddings = self.movie_structure_embedding_m_target

        self.final_user_embeddings = final_user_embeddings
        self.final_movie_embeddings = final_movie_embeddings

    def standardize_rating(self, r):
        """standardise ratings r which are [1, 5] to [-1, 1]"""
        float_r = tf.cast(r, tf.float32)
        return (float_r-3)/2

# class SVD_AE_Embedding(AbstractEmbedding):
#     def __init__(self, embedding_size, n_movies, n_user, batch_size=None):
#         if embedding_size==138:
#             svd_use_file = "Weights/svd_user_embedding_10.npy"
#             svd_mov_file = "Weights/svd_movies_embedding_10.npy"
#         elif embedding_size == 256:
#             svd_use_file = "Weights/svd_user_embedding_128.npy"
#             svd_mov_file = "Weights/svd_movies_embedding_128.npy"
#         else:
#             print("ERROR: with SVD_AE only embedding size 138 or 256 are possible so far, because only these were pretrained")
#             exit()
#         self.user_indxs = tf.placeholder(tf.int32, [batch_size], name="input_user_indices")
#         self.movie_indxs = tf.placeholder(tf.int32, [batch_size], name="input_movie_indices")
#         use_embed1 = np.load(svd_use_file).astype(np.float32)
#         mov_embed1 = np.load(svd_mov_file).astype(np.float32)
#         use_embed2 = np.load("Weights/user_embed_ae.npy").astype(np.float32)
#         mov_embed2 = np.load("Weights/movie_embed_ae.npy").astype(np.float32)
#         use_embed = np.concatenate((use_embed1, use_embed2), axis=1)
#         mov_embed = np.concatenate((mov_embed1, mov_embed2), axis=1)
#         self.user_embedding_w = tf.get_variable("row_embedding", dtype=tf.float32,
#                                                 regularizer=tf.contrib.layers.l2_regularizer(0.5),
#                                                 initializer=use_embed)
#         self.movie_embedding_w = tf.get_variable("col_embedding", dtype=tf.float32,
#                                                 regularizer=tf.contrib.layers.l2_regularizer(0.5),
#                                                 initializer=mov_embed)
#         self.final_user_embeddings = tf.nn.embedding_lookup(self.movie_embedding_w, self.movie_indxs)  # shape [batchsize, seqlength, embed_dim]
#         self.final_movie_embeddings = tf.nn.embedding_lookup(self.user_embedding_w, self.user_indxs)
#
#     def get_user_embedding(self):
#         return self.final_user_embeddings
#
#     def get_movie_embedding(self):
#         return self.final_movie_embeddings
#
#     def make_input_feeddict(self, X):
#         (user_ids, movie_ids) = X
#         d = {self.user_indxs: user_ids, self.movie_indxs: movie_ids}
#         return d
