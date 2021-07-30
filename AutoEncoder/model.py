import tensorflow as tf
import numpy as np

class AEmodel:
    def __init__(self, nr_users, nr_movies):
        self.nr_users=nr_users
        self.nr_movies=nr_movies

    def encode(self, x, rbm=None, keep_prob=1.0, n_hidden=512, n_embed=128, act_hidden=tf.nn.relu, act_out=tf.nn.relu, variational=False):
        with tf.name_scope("encoder"):
            if rbm is not None:
                concat = tf.concat([x, rbm_inp],1)
            else:
                concat = x
            hidden = tf.layers.dense(concat, n_hidden, activation=act_hidden)
            hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
            Z_mu = tf.layers.dense(hidden, n_embed, activation=act_out)

            if variational:
                print("Variational encoder used")
                Z_logvar = tf.layers.dense(hidden, 128)
                return Z_mu, Z_logvar
            else:
                print("non variational encoder")
                return Z_mu

    def decode(self, z, rbm=None, keep_prob=1.0, n_hidden=512, act_hidden=tf.nn.relu, act_out=None):
        with tf.name_scope("decoder"):
            hidden = tf.layers.dense(z, n_hidden, activation=act_hidden)
            hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
            outputs = tf.layers.dense(hidden, self.nr_movies, activation=act_out)
        return outputs


    def reparameterize(self, Z_mu, Z_logvar):
        Z_std = tf.exp(Z_logvar/2)
        mvn = tf.initializers.random_normal()
        eps = mvn(shape=tf.shape(Z_mu)) # statt call
        Z_sample = Z_mu + eps*Z_std
        return Z_sample

    def movie_encoder_loaded(self, movie, embedding_size=128, keep_prob=1):
        use_embed = np.load("../Weights/user_embed_ae.npy")
        W1 = tf.get_variable("row_embedding", dtype=tf.float32, initializer=use_embed)
        b1 = tf.get_variable("bmov",[128,1],trainable=True)
        encode_m = tf.transpose(tf.nn.sigmoid(tf.matmul(tf.transpose(W1),tf.transpose(movie))+b1))
        return encode_m

    def user_encoder__loaded(self,user, embedding_size=128, keep_prob=1):
        mov_embed = np.load("../Weights/movie_embed_ae.npy")
        W1 = tf.get_variable("col_embedding", dtype=tf.float32, initializer=mov_embed)
        b1 = tf.get_variable("buse",[128,1],trainable=True)
        encode_u = tf.transpose(tf.nn.sigmoid(tf.matmul(tf.transpose(W1),tf.transpose(user))+b1)) # tf.transpose(W1),tf.transpose(user)
        return encode_u

    def movie_encoder(self, movie, embedding_size=128, keep_prob=1):
        hidden_m = tf.layers.dense(movie, 128, activation=tf.nn.sigmoid)
        # hidden_m = tf.nn.dropout(hidden_m, keep_prob=keep_prob)
        encode_m = tf.layers.dense(hidden_m, embedding_size, activation=tf.nn.sigmoid)
        encode_m = tf.nn.dropout(encode_m, keep_prob=keep_prob)
        return encode_m

    def movie_decoder(self,encode_m, keep_prob=1):
        hidden_m = tf.layers.dense(encode_m, 128, activation=tf.nn.sigmoid)
        # hidden_m = tf.nn.dropout(hidden_m, keep_prob=keep_prob)
        outputs_m = tf.layers.dense(hidden_m, self.nr_users, activation=tf.nn.sigmoid)
        outputs_m = tf.nn.dropout(outputs_m, keep_prob=keep_prob)
        return outputs_m

    def user_encoder(self,user, embedding_size=128, keep_prob=1):
        # user model
        hidden_u = tf.layers.dense(user, 512, activation=tf.nn.relu)
        # hidden_u = tf.nn.dropout(hidden_u, keep_prob=keep_prob)
        encode_u = tf.layers.dense(hidden_u, embedding_size, activation=tf.nn.relu)
        encode_u = tf.nn.dropout(encode_u, keep_prob=keep_prob)
        return encode_u

    def user_decoder(self, encode_u, keep_prob=1):
        hidden_u = tf.layers.dense(encode_u, 512, activation=tf.nn.relu)
        # hidden_u = tf.nn.dropout(hidden_u, keep_prob=keep_prob)
        outputs_u = tf.layers.dense(hidden_u, self.nr_movies, activation=None)
        outputs_u = tf.nn.dropout(outputs_u, keep_prob=keep_prob)
        return outputs_u

    def rating_predictor(self, encode_u, encode_m, keep_prob=1):
        # middle model
        u_dot_i = tf.multiply(encode_m, encode_u)
        concat = tf.concat([encode_m, encode_u, u_dot_i],1)
        concat = tf.layers.dense(concat, 128, activation=tf.nn.relu)
        concat = tf.nn.dropout(concat, keep_prob=keep_prob)
        output = tf.layers.dense(concat, 1, activation= tf.nn.sigmoid)
        output = tf.nn.dropout(output, keep_prob=keep_prob)
        return output

    def rating_predictor_rhea(self, user_emb_req, item_emb_req, embed_size=128):
         u_dot_i = tf.multiply(item_emb_req,user_emb_req)
         inputX = tf.concat([item_emb_req,user_emb_req,u_dot_i],axis=1)
         W1 = tf.get_variable("wa",[200,3*embed_size],initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
         b1=tf.get_variable("ba",[200,1],trainable=True)
         out_1=tf.nn.relu(tf.matmul(W1,tf.transpose(inputX))+b1)
         W2=tf.get_variable("wb",[100,200],initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
         b2=tf.get_variable("bb",[100,1],trainable=True)
         out_2=tf.nn.relu(tf.matmul(W2,out_1)+b2)
         W3=tf.get_variable("w3",[1,100],initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
         b3=tf.get_variable("b3",[1,1],trainable=True)
         out_3=tf.matmul(W3,out_2)+b3
         out_3 = tf.nn.sigmoid(out_3)
         return out_3

    def sparse(self, Z_mu, k=20):
        # sparse AE
        print("SPARSE AE")
        topk, inds = tf.nn.top_k(Z_mu, k=k, sorted=False)
        min_topk = tf.reduce_min(topk, axis=1)
        subtracted = tf.transpose(tf.transpose(Z_mu) - min_topk)
        Z_mu = tf.where(tf.less(subtracted, 0), tf.zeros_like(Z_mu), Z_mu)
        return Z_mu

    def masked_loss(self, mask, y):
        """
        masks entries that are zero in input (mask) and computes loss only between the entries of reconstructed input (y) which is nonzero in mask
        """
        # mask= tf.where(tf.equal(x,0.0), tf.zeros_like(x), x) # indices of 0 values in the training set
        num_train_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32) # number of non zero values in the training set
        bool_mask=tf.cast(mask,dtype=tf.bool) # boolean mask
        masked_y = tf.where(bool_mask, y, tf.zeros_like(y)) # set the output values to zero if corresponding input values are zero
        return tf.sqrt(tf.div(tf.reduce_sum(tf.square(tf.subtract(masked_y*5,mask*5))),num_train_labels))

    def kld_loss(self, Z_mu, Z_logvar):
        # Return the KL divergence between Z and a Gaussian prior N(0, I)
        kld = - 0.5 * tf.reduce_sum(1 + Z_logvar - Z_mu**2 - tf.exp(Z_logvar), axis=1)
        return tf.reduce_mean(kld, axis=0)

    def mean_rating(self, outputs_u, outputs_m, output, user_mask_rat, movie_mask_rat):
        """
        Mean of the rating reconstructed from the movie embedding and the rating reconstructed from the user embedding
        """
        ## FOR MEAN RAT
        rat_use = tf.reshape(tf.reduce_sum(user_mask_rat * outputs_u, axis=1), (-1,1)) # mask and take sum to get this one
        rat_mov = tf.reshape(tf.reduce_sum(movie_mask_rat * outputs_m, axis=1), (-1,1))
        rat_mean = tf.concat([rat_use, rat_mov, output], 1)
        rat_mean = tf.reshape(tf.reduce_mean(rat_mean, axis=1), (-1,1))
        return rat_mean
