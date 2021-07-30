import abc
import tensorflow as tf

class AbstractEmbeddingBasedPredictionModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, embbedding_model, *args, **kwargs):
        """

        :param embbedding_model: an instance of a subclass of AbstractEmbedding, it has methods get_user_embedding and get_movie_embedding
        """
        raise NotImplementedError("has to be overwritten")

    @property
    @abc.abstractmethod
    def loss(self):
        """ONE LOSS PER SAMPLE, gets the loss-tensor that should be minimised, possibly the same as get_batch_mse"""
        raise NotImplementedError("has to be overwritten")

    @property
    @abc.abstractmethod
    def batch_mse(self):
        """gets the batch_mse tensor (not root!)"""
        raise NotImplementedError("has to be overwritten")

    @property
    @abc.abstractmethod
    def label_placeholder(self):
        """gets the placeholder for the label (type int, elements {1 ... 5}"""
        raise NotImplementedError("has to be overwritten")

    @property
    @abc.abstractmethod
    def predictions(self):
        """gets the tensor containing the predictions, tf.tensor of shape [batchsize] with float values"""
        raise NotImplementedError("has to be overwritten")

    def get_additional_val_summaries(self):
        return []

    def per_sample_mse(self, labels, predictions):
        print("per_sample_mse")
        res =  (tf.cast(labels, tf.float32) - predictions)**2

        # res_mse = tf.losses.mean_squared_error(labels, predictions, reduction=Reduction.NONE) + 1
        #
        # with tf.control_dependencies([tf.assert_near(res, res_mse, message="mse_implementation wrong")]):
        #     out = tf.identity(res)

        out = res

        return out

class PassThroughModel(AbstractEmbeddingBasedPredictionModel):
    """A hacky model that just passes through the embeddings as predictions. """
    def __init__(self, embbedding_model, *args, **kwargs):

        self.label = tf.placeholder(tf.float32, [None], name='target_label')

        self.user_embed = embbedding_model.get_user_embedding()
        self.movie_embed = embbedding_model.get_movie_embedding()

        # tf.assert_equal(self.user_embed, self.movie_embed)

        # self.constant_pred = tf.get_variable(dtype=tf.float32, shape=[], name="const_output")

        self.model_preds = tf.squeeze(self.user_embed)

        self.loss_per_sample = self.per_sample_mse(labels=self.label, predictions=self.model_preds)


    @property
    def loss(self):
        return self.loss_per_sample

    @property
    def batch_mse(self):
        return tf.reduce_mean(self.loss_per_sample)

    @property
    def label_placeholder(self):
        return self.label

    @property
    def predictions(self):
        return self.model_preds



class Inner_Product_Model(AbstractEmbeddingBasedPredictionModel):
    def __init__(self, embedding_model, batch_size=None):


        self.label = tf.placeholder(tf.float32, [batch_size], name='target_label')

        self.user_embed = embedding_model.get_user_embedding()
        self.movie_embed = embedding_model.get_movie_embedding()

        with tf.name_scope("prediction_model"):
            self.model_preds = tf.sigmoid(self._inner_product(self.user_embed, self.movie_embed))*4 + 1

            self.mse_loss_per_sample = self.per_sample_mse(labels=self.label, predictions=self.model_preds)

    def _inner_product(self, a, b, axis_in_which_to_dot_product=-1):
        return tf.reduce_sum(tf.multiply(a, b), axis=axis_in_which_to_dot_product)

    @property
    def loss(self):
        return self.mse_loss_per_sample

    @property
    def batch_mse(self):
        return tf.reduce_mean(self.mse_loss_per_sample)

    @property
    def label_placeholder(self):
        return self.label

    @property
    def predictions(self):
        return self.model_preds

class MLP1_model(AbstractEmbeddingBasedPredictionModel):
    def __init__(self, embedding_model,l2_reg=0.0,batch_size=None):
        self.user_embed = embedding_model.get_user_embedding()
        self.movie_embed = embedding_model.get_movie_embedding()

        self.embedding_dim = self.movie_embed.get_shape().as_list()[1]
        self.label = tf.placeholder(tf.float32, [batch_size], name='target_label')

        # self.user_indxs = tf.placeholder(tf.int32, [batch_size])
        # self.movie_indxs = tf.placeholder(tf.int32, [batch_size])
        # -> you don't need these as self.user_embed and self.movie_embed already contain the properly embeded users and movies for the current batch

        with tf.name_scope("prediction_model"):
            # item_emb_req = tf.gather(self.item_embed, self.movie_indxs)
            # user_emb_req = tf.gather(self.user_embed, self.user_indxs)
            #  --> You don't need gather any more since self.user_embed and self.movie_embed already contains the embeded elements for the indices
            item_emb_req = self.movie_embed
            user_emb_req = self.user_embed

            u_dot_i = tf.multiply(item_emb_req, user_emb_req)
            input_nn = tf.concat([item_emb_req, user_emb_req, u_dot_i], axis=1)
            W1 = tf.get_variable("w1", [200, 3 * self.embedding_dim], initializer=tf.contrib.layers.xavier_initializer(),
                                 regularizer=tf.contrib.layers.l2_regularizer(l2_reg), trainable=True)
            b1 = tf.get_variable("b1", [200, 1],regularizer=tf.contrib.layers.l2_regularizer(l2_reg), trainable=True)
            out_1 = tf.nn.relu(tf.matmul(W1, tf.transpose(input_nn)) + b1)
            W2 = tf.get_variable("w2", [100, 200], initializer=tf.contrib.layers.xavier_initializer(),
                                 regularizer=tf.contrib.layers.l2_regularizer(l2_reg), trainable=True)
            b2 = tf.get_variable("b2", [100, 1],regularizer=tf.contrib.layers.l2_regularizer(l2_reg), trainable=True)
            out_2 = tf.nn.relu(tf.matmul(W2, out_1) + b2)
            W3 = tf.get_variable("w3", [1, 100], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(l2_reg),trainable=True)
            b3 = tf.get_variable("b3", [1, 1],regularizer=tf.contrib.layers.l2_regularizer(l2_reg), trainable=True)
            out3 = tf.sigmoid(tf.matmul(W3, out_2) + b3)*4+1
            self.model_preds = out3
            self.mse_loss_per_sample = self.per_sample_mse(tf.squeeze(self.label), tf.squeeze(self.model_preds))

    @property
    def loss(self):
        return self.mse_loss_per_sample

    @property
    def batch_mse(self):
        return tf.reduce_mean(self.mse_loss_per_sample)

    @property
    def label_placeholder(self):
        return self.label

    @property
    def predictions(self):
        return tf.squeeze(tf.clip_by_value(self.model_preds, 1, 5))


class MLP2_model(AbstractEmbeddingBasedPredictionModel):

    @property
    def loss(self):
        return self.cross

    @property
    def label_placeholder(self):
        return self.label_inp_placeholder

    @property
    def batch_mse(self):
        return tf.reduce_mean(self.per_sample_mse(tf.squeeze(self.rating), tf.squeeze(self.model_predictions)))

    @property
    def predictions(self):
        return tf.squeeze(self.model_predictions)



    def __init__(self, embedding_model,
                 l2_reg=0.0):

        emb_u = tf.expand_dims(embedding_model.get_user_embedding(), axis=1)
        print(emb_u.shape)
        emb_m = tf.expand_dims(embedding_model.get_movie_embedding(), axis=1)
        print(emb_m.shape)

        # self.user_embs_size = user_embs_size
        # self.movies_embs_size = user_embs_size
        self.l2_reg = float(l2_reg)

        regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg)

        with tf.name_scope("Inputs"):
            self.label_inp_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name='rating')
            self.rating = tf.expand_dims(self.label_inp_placeholder, axis=1)


        with tf.variable_scope("Projection_of_user_embeddings"):
            projection_u = tf.layers.dense(inputs=emb_u, units=30, kernel_regularizer=regularizer, \
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                   seed=40),
                                           activation=tf.nn.relu)

        with tf.variable_scope("Projection_of_movie_embeddings"):
            projection_m = tf.layers.dense(inputs=emb_m, units=30, kernel_regularizer=regularizer, \
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                   seed=42),
                                           activation=tf.nn.relu)

        with tf.name_scope("Element_wise_multipl_of_projections"):
            element_wise_mult = projection_u * projection_m

        with tf.name_scope("Concatenation_of_projections"):
            concat = tf.concat([projection_u, projection_m], axis=2)

        with tf.variable_scope("Dense_layer_for_concatenation"):
            extra_layer = tf.layers.dense(inputs=concat, units=40, activation=tf.nn.relu,
                                          kernel_regularizer=regularizer, \
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                  seed=43))

        with tf.name_scope("Concatenation_of_output_of_last_Dense_and_Element_wise_multip-of_projections"):
            extra_concat = tf.concat([element_wise_mult, extra_layer], axis=2)

        with tf.variable_scope("Feed_Forward_network"):
            mlp_layer = tf.layers.dense(inputs=extra_concat, units=20, activation=tf.nn.relu,
                                        kernel_regularizer=regularizer, \
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=44))

            self.logits = tf.layers.dense(inputs=mlp_layer, units=5, activation=None, kernel_regularizer=regularizer, \
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                  seed=45))

        with tf.name_scope("categorical_cross_entropy_loss_function"):
            self.cross = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=(self.rating-1))

        with tf.name_scope("Predictions"):
            self.softs = tf.nn.softmax(logits=self.logits)
            # weigted average class*probability
            self.model_predictions = tf.reduce_sum(tf.multiply(tf.constant([1, 2, 3, 4, 5], dtype=tf.float32), self.softs), axis=2)
