import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import tensorflow as tf
from utils import *
from data_preprocess import split_data, load_data, get_datasplit, matrix_data_omega,load_data_omega, make_submission
from model import AEmodel
from Utils.DataSplits import DataSplit
import os


class Trainer():
    def __init__(self, FLAGS, nr_movies=1000, nr_users=10000):
        self.FLAGS = FLAGS
        self.nr_movies=nr_movies
        self.nr_users=nr_users

    def del_all_flags(self, FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)

    def train(self, train_matrix, test_matrix, variational=False, return_embedding=False):

        tf.reset_default_graph()

        # placeholders
        with tf.name_scope("Input"):
            x_mask = tf.placeholder(tf.float32, (None, self.nr_movies), name="maskInput")
            x = tf.placeholder(tf.float32, (None, self.nr_movies), name = "input")
            beta = tf.placeholder(tf.float32, name="beta")
            drop_rate = tf.placeholder(tf.float32, name="dropoutPlaceholder")

        with tf.name_scope("model"):
            model = AEmodel(self.nr_users, self.nr_movies)

            # forward, loss and train
            if not variational:
                Z = model.encode(x, n_hidden=self.FLAGS.n_hidden, n_embed=self.FLAGS.n_embed, act_hidden=eval(self.FLAGS.act_hidden_e), act_out=eval(self.FLAGS.act_out_e), keep_prob=drop_rate, variational=variational)
                # Z = model.sparse(Z, k=self.FLAGS.k_sparse)
                outputs = model.decode(Z, n_hidden=self.FLAGS.n_hidden, act_hidden=eval(self.FLAGS.act_hidden_d), act_out=eval(self.FLAGS.act_out_d), keep_prob=drop_rate)
                loss_op = model.masked_loss(x_mask, outputs)
                tf.summary.scalar("loss", loss_op)
            else:
            ## variational:
                Z_mu, Z_logvar = model.encode(x, n_hidden=self.FLAGS.n_hidden, n_embed=self.FLAGS.n_embed, act_hidden=eval(self.FLAGS.act_hidden_e), act_out=eval(self.FLAGS.act_out_e), keep_prob=drop_rate, variational=variational)
                Z = model.reparameterize(Z_mu, Z_logvar)
                outputs = model.decode(Z, n_hidden=self.FLAGS.n_hidden, act_hidden=eval(self.FLAGS.act_hidden_d), act_out=eval(self.FLAGS.act_out_d), keep_prob=drop_rate)
                # loss
                loss_recon = model.masked_loss(x_mask, outputs)
                loss_kld = model.kld_loss(Z_mu, Z_logvar)
                loss_op = loss_recon + beta * loss_kld # BETA
                tf.summary.scalar("loss", loss_op)
                tf.summary.scalar("loss_recon", loss_recon)
                tf.summary.scalar("loss_kld",loss_kld)

        with tf.name_scope("loss_and_train"):
            if self.FLAGS.regularize==True:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                loss_op_reg = loss_op + l2_loss * self.FLAGS.lamba
            else:
                loss_op_reg = loss_op

            train_op = tf.train.RMSPropOptimizer(self.FLAGS.learning_rate).minimize(loss_op_reg)

        saver = tf.train.Saver()

        ### Run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_losses=[]

            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.FLAGS.summaries_dir, sess.graph)

            # matrix = train_matrix.copy()
            # print("training matrix vae shape", matrix.shape)

            for epoch in range(self.FLAGS.epochs):
                # if self.FLAGS.DAE: # For denoising AE
                #    matrix = noisy_labels(train_matrix)
                ## TRAIN
                train_losses=[]

                for user in range(self.nr_users//self.FLAGS.batch_size):
                    sta = user*self.FLAGS.batch_size
                    end = (user+1)*self.FLAGS.batch_size
                    loss, _  = sess.run([loss_op, train_op], {x: train_matrix[sta:end], x_mask: train_matrix[sta:end], beta:epoch/self.FLAGS.epochs, drop_rate: self.FLAGS.dropout_rate}) # rbm_inp:rbm_transformed[sta:end]
                    train_losses.append(loss)

                ## TEST
                if not variational:
                    test_loss, summary = sess.run([loss_op, merged_summary], {x: train_matrix, x_mask: test_matrix, drop_rate:1.0}) # rbm_inp:rbm_transformed,
                    print("train", np.mean(train_losses), "test", test_loss)
                else:
                ## Variational:
                    test_loss_kld, test_loss_recon, test_loss, summary = sess.run([loss_kld, loss_recon, loss_op, merged_summary], {x: train_matrix, x_mask: test_matrix, beta:epoch/self.FLAGS.epochs, drop_rate:1.0}) # rbm_inp:rbm_transformed,
                    print("train", np.mean(train_losses), "test", test_loss, "test_kld", test_loss_kld, "test_recon", test_loss_recon)

                test_losses.append(test_loss)

                summary_writer.add_summary(summary, epoch)

            if variational:
                ## SAMPLE NEW DATA (can be used to train model)
                nr_samples = 1000
                Z_sample = np.random.normal(size=(nr_samples, self.FLAGS.n_embed))
                augment = sess.run(outputs, feed_dict={Z:Z_sample, drop_rate:1.0})
                print(augment.shape)
                np.save("VAE_sample.npy", augment)

            if return_embedding:
                embed = sess.run(Z,{x: train_matrix, drop_rate:1.0})
                sess.close()
                # self.del_all_flags(self.FLAGS)
                return embed
            else:
                # run on full set to get final reconstructed matrix
                out_matrix = sess.run(outputs,{x: train_matrix, drop_rate:1.0}) #rbm_inp:rbm_transformed,
                sess.close()
                self.del_all_flags(self.FLAGS)
                return out_matrix


if __name__ == "__main__":

    tf.flags.DEFINE_bool("CV_save_embed", False, "if you want to save all the embeddings")
    tf.flags.DEFINE_string("embedding", "user", "user or movie embedding")
    tf.flags.DEFINE_integer("val_part", 9, "")
    tf.flags.DEFINE_integer("batch_size", 8, "")
    tf.flags.DEFINE_integer("epochs", 200, "")
    tf.flags.DEFINE_float("learning_rate", 0.0001, "")
    tf.flags.DEFINE_float("lamba", 0.001, "")
    tf.flags.DEFINE_bool("regularize", True, "")
    tf.flags.DEFINE_integer("n_embed", 128, "")
    tf.flags.DEFINE_integer("n_hidden", 512, "")
    tf.flags.DEFINE_string("act_hidden_e", "tf.nn.relu", "")
    tf.flags.DEFINE_string("act_out_e", "tf.nn.relu", "")
    tf.flags.DEFINE_string("act_hidden_d", "tf.nn.relu", "")
    tf.flags.DEFINE_string("act_out_d", "None", "")
    tf.flags.DEFINE_integer("k_sparse", 20, "")
    tf.flags.DEFINE_float("dropout_rate", 1.0, "")
    tf.flags.DEFINE_string("summaries_dir", "logs", "")
    tf.flags.DEFINE_string("save_path", "../Weights/model3/model", "")
    tf.flags.DEFINE_string("embedding_save_path", "../Weights/ae_embeddings/", "")
    tf.flags.DEFINE_string("train_data_path", "../data/", "")

    split = DataSplit.load_from_pickle("s10.pickle")

    if tf.flags.FLAGS.CV_save_embed:
        for i in range(10):
            val_ids = split.get_validation(val_part=i)
            omega_train, omega_test = get_datasplit(val_ids, data_path=tf.flags.FLAGS.train_data_path)
            train_matrix = matrix_data_omega(omega_train)
            test_matrix = matrix_data_omega(omega_test)
            train_matrix, test_matrix = normalize_01(train_matrix, test_matrix)
            print(np.unique(train_matrix), np.unique(test_matrix))
            print(train_matrix.shape, test_matrix.shape)

            if tf.flags.FLAGS.embedding=="user":
                trainer = Trainer(tf.flags.FLAGS, nr_users=10000, nr_movies=1000)
                embed = trainer.train(train_matrix, test_matrix, variational=False, return_embedding=True)
            elif tf.flags.FLAGS.embedding=="movie":
                trainer = Trainer(tf.flags.FLAGS, nr_users=1000, nr_movies=10000)
                embed = trainer.train(train_matrix.T, test_matrix.T, variational=False, return_embedding=True)
            else:
                print("ERROR: embedding must be user or movie")
                exit()
            print(embed.shape)
            np.save(os.path.join(tf.flags.FLAGS.embedding_save_path, tf.flags.FLAGS.embedding+"_embed_ae_"+str(i)+".npy"), embed)

    else:
        val_ids = split.get_validation(val_part=tf.flags.FLAGS.val_part)
        omega_train, omega_test = get_datasplit(val_ids, data_path=tf.flags.FLAGS.train_data_path)
        train_matrix = matrix_data_omega(omega_train)
        test_matrix = matrix_data_omega(omega_test)
        train_matrix, test_matrix = normalize_01(train_matrix, test_matrix)
        print(np.unique(train_matrix), np.unique(test_matrix))
        print(train_matrix.shape, test_matrix.shape)

        trainer = Trainer(tf.flags.FLAGS)
        out_matrix = trainer.train(train_matrix, test_matrix, variational=False)

        make_submission(out_matrix, path = os.path.join(tf.flags.FLAGS.train_data_path, "sampleSubmission.csv"))
