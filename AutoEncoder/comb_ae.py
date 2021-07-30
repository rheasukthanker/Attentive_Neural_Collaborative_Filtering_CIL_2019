import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import tensorflow as tf
from utils import *
from model import AEmodel
from data_preprocess import submission_omega, get_datasplit, matrix_data_omega #load_data_omega, ,
from Utils.DataSplits import DataSplit

tf.flags.DEFINE_integer("batch_size", 100, "") # BEFORE: 8
tf.flags.DEFINE_integer("EPOCHS", 40, "")
tf.flags.DEFINE_integer("nr_movies", 1000, "Movies in dataset")
tf.flags.DEFINE_integer("nr_users", 10000, "Users in dataset")
tf.flags.DEFINE_float("learning_rate", 0.00005, "")
tf.flags.DEFINE_string("act_hidden", "tf.nn.relu", "")
tf.flags.DEFINE_float("lamba", 0.001, "")
tf.flags.DEFINE_bool("regularize", False, "")
tf.flags.DEFINE_integer("embedding_size_m", 128, "")
tf.flags.DEFINE_integer("embedding_size_u", 128, "")
tf.flags.DEFINE_integer("k_sparse_m", 40, "")
tf.flags.DEFINE_integer("k_sparse_u", 20, "")
tf.flags.DEFINE_string("act_function_end", "None", "")
tf.flags.DEFINE_integer("rat_loss_weight", 2, "")
tf.flags.DEFINE_integer("test_batch_size", 20000, "")
tf.flags.DEFINE_float("drop_rate", 0.6, "")
tf.flags.DEFINE_string("save_path", ".", "path where to save the submission")
tf.flags.DEFINE_string("sample_submission_path", "../data/sampleSubmission.csv", "path where sampleSubmission file is")
submission_way = 3
FLAGS = tf.flags.FLAGS

# ###  LOAD DATA
split = DataSplit.load_from_pickle("s10.pickle")
val_ids = split.get_validation(val_part=9)
omega_train, omega_test = get_datasplit(val_ids)
train_matrix = matrix_data_omega(omega_train)
test_matrix = matrix_data_omega(omega_test)
train_matrix, test_matrix = normalize_01(train_matrix, test_matrix)
assert(not np.any(np.logical_and(train_matrix, test_matrix))) # Verify that no cells are nonzero in train and test
# ### INPUT

with tf.name_scope("input"):
    user = tf.placeholder(tf.float32, (None, FLAGS.nr_movies), name = "input_user")
    user_mask = tf.placeholder(tf.float32, (None, FLAGS.nr_movies), name = "mask_user")
    movie = tf.placeholder(tf.float32, (None, FLAGS.nr_users), name = "input_movie")
    movie_mask = tf.placeholder(tf.float32, (None, FLAGS.nr_users), name = "mask_movie")
    rating = tf.placeholder(tf.float32, (None, 1), name="input_rating")
    drop_rate = tf.placeholder(tf.float32, name="dropoutPlaceholder")

# FORWARD PASS
with tf.name_scope("model"):
    model = AEmodel(FLAGS.nr_users, FLAGS.nr_movies)
    encode_m = model.movie_encoder(movie, embedding_size=FLAGS.embedding_size_m, keep_prob=drop_rate)
    # encode_m = model.sparse(encode_m, k=FLAGS.k_sparse_m) # added sparseness
    outputs_m = model.movie_decoder(encode_m, keep_prob=drop_rate)
    encode_u = model.user_encoder(user, embedding_size=FLAGS.embedding_size_u, keep_prob=drop_rate)
    # encode_u = model.sparse(encode_u, k=FLAGS.k_sparse_u) # added sparseness
    outputs_u = model.user_decoder(encode_u, keep_prob=drop_rate)
    output = model.rating_predictor(encode_u, encode_m, keep_prob=drop_rate)

# LOSS
with tf.name_scope("loss"):
    mov_loss = model.masked_loss(movie_mask, outputs_m)
    use_loss = model.masked_loss(user_mask, outputs_u)
    # rat_mean = model.mean_rating(outputs_u, outputs_m, output, user_mask_rat, movie_mask_rat)
    # rat_loss_mean = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(rat_mean*5,rating*5))))
    rat_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(output*5,rating*5))))

    loss_op = FLAGS.rat_loss_weight*rat_loss + mov_loss + use_loss # + rat_loss_mean

    if FLAGS.regularize:
        print("REGULARIZING ", FLAGS.lamba)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss_op_reg = loss_op + l2_loss * FLAGS.lamba
    else:
        loss_op_reg=loss_op
    train_op=tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(loss_op_reg)

# ### Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(FLAGS.EPOCHS):
        print("EPOCH", epoch)
        train_losses = []
        test_losses_user = []
        test_losses_movie = []

        try:
            for i in range(len(omega_train)//FLAGS.batch_size):
                batch = omega_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size] # fine
                user_inp = train_matrix[batch[:, 0]]
                movie_inp = train_matrix[:, batch[:,1]].T
                rat_inp = np.asarray([batch[:,2]/5]).T
                train_loss, _= sess.run([loss_op, train_op], {user: user_inp, movie: movie_inp, rating:rat_inp, user_mask:user_inp, movie_mask:movie_inp, drop_rate:FLAGS.drop_rate}) # , user_mask_rat:user_mask_rat_inp, movie_mask_rat:movie_mask_rat_inp})
                train_losses.append(train_loss)

                if (i+1)%500==0:
                    test_mse = 0
                    for j in range(len(omega_test)//FLAGS.test_batch_size+1):
                        start = j*FLAGS.test_batch_size
                        end = (j+1)*FLAGS.test_batch_size
                        if end>len(omega_test):
                            end = len(omega_test)
                        batch = omega_test[start:end]
                        user_inp = train_matrix[batch[:, 0]]
                        movie_inp = train_matrix[:, batch[:,1]].T
                        test_rat_inp = np.asarray([batch[:,2]/5]).T
                        rat_out = sess.run(output, {user: user_inp, movie: movie_inp, drop_rate:1}) # 16*10000, 16*1000
                        # rat_out = (rat_out[:,0]*5).clip(min=1, max=5)
                        test_mse += np.sum((test_rat_inp[:,0]*5 - rat_out[:,0]*5)**2)
                    print("Train loss", np.mean(train_losses), "Validation", np.sqrt(test_mse/len(omega_test)))
                    # print("Train loss", np.mean(train_losses), "Test", np.sqrt(np.sum((test_rat_inp[:,0]*5-rat_out[:,0]*5)**2)/FLAGS.test_batch_size))

        except KeyboardInterrupt:
            print("Interrupted - type 1 for movie recon, 2 for user recon, 3 for rating, 4 for mean")
            submission_way = int(input())
            break

        np.random.shuffle(omega_train)

    # SUBMISSION
    print("Submission started, submission way:", submission_way)
    submission = pd.read_csv(FLAGS.sample_submission_path)
    submission_arr = np.asarray(submission)
    omega_sub = submission_omega(submission_arr)
    out_list = []
    for j in range(len(omega_sub)//FLAGS.batch_size+1):
        batch = omega_sub[j*FLAGS.batch_size:(j+1)*FLAGS.batch_size]
        user_inp = train_matrix[batch[:, 0]]
        movie_inp = train_matrix[:, batch[:,1]].T

        mov_embed, use_embed, rat_out = sess.run([outputs_m, outputs_u, output], {user: user_inp, movie: movie_inp, drop_rate:1}) # 16*10000, 16*1000
        rat_mov = np.asarray([mov_embed[i, batch[i,0]] * 5 for i in range(len(batch))])
        rat_use = np.asarray([use_embed[i, batch[i,1]] * 5 for i in range(len(batch))])

        if submission_way==1: # use movie embedding results
            rat_out = rat_mov.clip(min=1, max=5)
        elif submission_way==2: # use user embedding results
            rat_out = rat_use.clip(min=1, max=5)
        elif submission_way==3: # use predicted rating as result
            rat_out = (rat_out[:,0]*5).clip(min=1, max=5)
        elif submission_way==4: # use mean of all three as result
            concats = np.asarray([rat_mov, rat_use, rat_out[:,0]*5])
            rat = np.mean(concats, axis=0)
            rat_out = rat.clip(min=1, max=5)
        else:
            print("error: wrong submission way")
        out_list.extend(rat_out.tolist())

sess.close()

submission["Prediction"] = out_list
submission.to_csv(os.path.join(FLAGS.save_path,"submission.csv"), index=False)
