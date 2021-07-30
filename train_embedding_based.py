from Utils.DataUtils import *
import tensorflow as tf
from EmbeddingModels.Embeddings import SimpleEmbedding, AutoencoderEmbedding, AttentionEmbedding, AttentionWeightedContextRaitings# , SVD_AE_Embedding
from EmbeddingModels.PredictionModels import Inner_Product_Model, MLP1_model, MLP2_model, PassThroughModel
from Utils.DataBuilders import SimpleEmbeddingDataBuilder, AttentionEmbeddingDataBuilder
from Utils.DataSplits import DataSplit
from EmbeddingModels.ModelWrapper import ModelWrapper
from EmbeddingModels.SampleWeighting import Weighted_Loss_Users, Weighted_Loss_Ratings, Weighted_Debug
import time
from datetime import datetime
import argparse
import json
import sys

def save_json(obj, path):
    with open(path, "w+") as f:
        json.dump(obj, f, indent=4)


def read_csv(path):
    data = pd.read_csv(path)
    data.set_index("sample_id", inplace=True)
    return data

def opt_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_config(path, type):
    with open(path, "r") as f:
        config= json.load(f)

    if config["type"] != type:
        raise ValueError("The config file {} containing type {} was given when a config file for type {} was expected".format(path, config["type"], type))

    return config["name"], config["params"]

def load_part_of_weights(path, namespace_prefix, sess):
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}.*".format(namespace_prefix))

    saver = tf.train.Saver(var_list=params, max_to_keep=1, save_relative_paths=True)

    path = os.path.join(path, "checkpoints")
    # Restore the latest checkpoint found in experiment_dir.
    ckpt_path = tf.train.latest_checkpoint(path)

    if ckpt_path:
        print("Loading model checkpoint {0}".format(ckpt_path))
        print("with params {}".format(params))
        saver.restore(sess, ckpt_path)
    else:
        raise ValueError("could not load checkpoint, probably because {} does not exist".format(path))


def build_and_train(args):
    training_data = read_csv(os.getcwd() + "/data/structured_data_train.csv")
    n_movies, n_user = get_n_movies_users(training_data)

    if args.make_submissions:
        submission_data = read_csv(os.getcwd() + "/data/structured_sample_submission.csv")
    else:
        submission_data = None

    dataSplit = DataSplit.load_from_pickle("s10.pickle")

    if args.embedding_config != "":
        embedding_name, embedding_params = read_config(args.embedding_config, "embedding")
        if args.embed_size != -1:
            embedding_params["embedding_size"] = args.embed_size
    else:
        embedding_name = args.embedding_version
        if args.embed_size == -1:
            raise ValueError("Either a config file containing the embedding_size has to be given for the embedding or the --embed_size flag has to be present")
        embedding_params = dict(embedding_size = args.embed_size)

    if args.prediction_config != "":
        prediction_name, prediction_params = read_config(args.prediction_config, "prediction")
    else:
        prediction_name = args.prediction_version
        prediction_params = dict()

    def build_f(sess, is_training_mode_placeholder, split_nr=9):

        with tf.variable_scope("Embedding"):
            if embedding_name == "simple":
                data_builder = SimpleEmbeddingDataBuilder()
                embedding_model = SimpleEmbedding(n_movies=n_movies, n_user=n_user, **embedding_params)
            elif embedding_name == "autoencoder_embedding":
                data_builder = SimpleEmbeddingDataBuilder()
                embedding_model = AutoencoderEmbedding(n_movies=n_movies, n_user=n_user, split_nr=split_nr, weights_path = args.load_embedding_weights_numpy, **embedding_params)
            elif embedding_name == "attention":
                data_builder = AttentionEmbeddingDataBuilder(n_contect_movies_predict=args.n_context_movies_pred, n_context_movies_train=args.n_context_movies)
                embedding_model = AttentionEmbedding(n_movies=n_movies, n_users=n_user, is_training=is_training_mode_placeholder, **embedding_params)
            elif embedding_name == "attention_rating":
                data_builder = AttentionEmbeddingDataBuilder(n_contect_movies_predict=args.n_context_movies_pred, n_context_movies_train=args.n_context_movies)
                embedding_model = AttentionWeightedContextRaitings(**embedding_params)

        with tf.variable_scope("Prediction"):
            if prediction_name == "inner_prod":
                pred_model = Inner_Product_Model(embedding_model, **prediction_params)
            elif prediction_name == "mlp1":
                pred_model = MLP1_model(embedding_model, **prediction_params)
            elif prediction_name == "mlp2":
                pred_model = MLP2_model(embedding_model, **prediction_params)
            elif prediction_name == "pass_through":
                pred_model = PassThroughModel(embedding_model, **prediction_params)


        if args.sample_weighting == "rating":
            sample_weighting = Weighted_Loss_Ratings(training_data)
            weighted_samples = True
        elif args.sample_weighting == "user":
            sample_weighting = Weighted_Loss_Users(training_data)
            weighted_samples = True
        elif args.sample_weighting == "debug":
            sample_weighting = Weighted_Debug(training_data)
            weighted_samples = True
        elif args.sample_weighting == "none":
            sample_weighting = None
            weighted_samples = False

        data_iter = DataIter(data_builder=data_builder, data=training_data, print_timings=(args.verbosity >= 2), sample_weightings=sample_weighting)
        data_iter.set_validation_indices(dataSplit.get_validation(split_nr))

        return ModelWrapper(embedding=embedding_model, prediction=pred_model, weighted_samples=weighted_samples), data_iter

    exp_name =  (datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S") * (1-args.no_date)) + args.experiment_name
    exp_path = os.path.join(args.out_dir, exp_name)

    opt_make_dir(exp_path)

    summary_json = vars(args)
    summary_json["command"] = " ".join(["python"] + sys.argv)
    summary_json["embedding_name"] = embedding_name
    summary_json["embedding_params"] = embedding_params

    summary_json["prediciton_name"] = prediction_name
    summary_json["prediction_params"] = prediction_params

    print("The complte configuration: ")
    print(json.dumps(summary_json, indent=4))

    save_json(summary_json, os.path.join(exp_path, "summary.json"))

    if args.crossvalidate:
        def opt_add_split(path_string, split_id):
            if path_string != "":
                return os.path.join(path_string, "Split_{}".format(split_id))
            else:
                return ""

        val_scores_collector = np.zeros((dataSplit.split, args.n_epochs+1))

        other_params = vars(args).copy()
        del other_params["load_embedding_weights"]
        del other_params["load_prediction_weights"]

        for split_id in range(dataSplit.split):
            load_embedding_weights_split = opt_add_split(args.load_embedding_weights, split_id)
            load_prediction_weights_split = opt_add_split(args.load_prediction_weights, split_id)

            split_exp_path = opt_add_split(exp_path, split_id)

            opt_make_dir(split_exp_path)

            build_f_split = lambda sess, is_training_mode_placeholder: build_f(sess=sess, is_training_mode_placeholder=is_training_mode_placeholder, split_nr=split_id)

            val_scores_split = train_model(builder_function=build_f_split, exp_path=split_exp_path,
                                           submission_data=submission_data,
                                           load_embedding_weights=load_embedding_weights_split, load_prediction_weights=load_prediction_weights_split,
                                           **other_params)

            val_scores_collector[split_id, :] = val_scores_split

            summary_json["validation_scores_split{}".format(split_id)] = val_scores_split.tolist()

        avg_val_scores = np.mean(val_scores_collector, axis=0)

        summary_json["avg_val_scores"] = avg_val_scores.tolist()

        best_epoch = np.argmin(avg_val_scores)
        best_score = avg_val_scores[best_epoch]

        summary_json["best_avg_score"] = float(best_score)
        summary_json["best_epoch"] = int(best_epoch)
    else:
        val_scores = train_model(builder_function=build_f, exp_path=exp_path, submission_data=submission_data, **vars(args))

        best_epoch = np.argmin(val_scores)
        best_score = val_scores[best_epoch]
        summary_json["best_epoch"] = int(best_epoch)
        summary_json["best_score"] = float(best_score)
        summary_json["epoch_validation_scores"] = val_scores.tolist()

    save_json(summary_json, os.path.join(exp_path, "summary.json"))


def train_model(builder_function, reg_constant, learning_rate, batchsize, n_epochs, verbosity, optimizer, tensorboard, exp_path, submission_data, load_embedding_weights, load_prediction_weights, force_gpu, checkpoint, *args, **kwargs):
    assert(len(args)==0)
    global min_validation_rmse
    min_validation_rmse = 1e9

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        device_count = {"GPU": 1} if force_gpu else {"GPU": 0}

        sess =  tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count,
                                              allow_soft_placement=True))

        with sess.as_default():

            if force_gpu:
                assert(tf.test.is_gpu_available())

            is_training_mode_placeholder = tf.placeholder(tf.bool, name="is_training_mode")

            model, data_iter = builder_function(sess, is_training_mode_placeholder=is_training_mode_placeholder)

            global_step = tf.Variable(0, name="global_step", trainable=False)

            if optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif optimizer == "rms":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

            if reg_constant != 0:
                print("regularizing with {}".format(reg_constant))
                regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                print(regularizers)
                combined_loss = model.loss + reg_constant * tf.reduce_sum(regularizers)
            else:
                combined_loss = model.loss

            train_op = optimizer.minimize(combined_loss, global_step=global_step)

            print("Writing to {}\n".format(exp_path))

            # Summaries for loss and accuracy
            additional_samples_val_summaries_op = None
            if tensorboard==1:
                mse_summary = tf.summary.scalar("mse_train", model.batch_mse)
                mse_val=tf.placeholder(tf.float32,())
                mse_val_summary=tf.summary.scalar("rmse_validation",mse_val)
                train_summary_dir = os.path.join(exp_path, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                train_summary_op = mse_summary

                val_summary_dir = os.path.join(exp_path, "summaries", "validation")
                val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
                val_summary_op = mse_val_summary

                additional_samples_val_summaries_op = model.get_additional_val_summaries()

            else:
                train_summary_op = None

            if checkpoint:
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=2, save_relative_paths=True)
                checkpoint_path = os.path.join(exp_path, "checkpoints", "model.ckpt")

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            if load_embedding_weights != "":
                load_part_of_weights(load_embedding_weights, namespace_prefix="Embedding", sess=sess)
            if load_prediction_weights != "":
                load_part_of_weights(load_prediction_weights, namespace_prefix="Prediction", sess=sess)

            sess.graph.finalize()

            # Define training and dev steps (batch)
            def train_step(training_data, label, weightings=None):
                """
                A single training step
                """
                feed_dict = model.make_input_feeddict(training_data, label, weightings=weightings)
                feed_dict[is_training_mode_placeholder] = True

                if tensorboard==1:
                   fetches = [train_op, global_step, model.batch_mse,train_summary_op]
                else:
                   fetches= [train_op, global_step, model.batch_mse]
                t0 = time.time()
                out = sess.run(fetches, feed_dict)
                _, step, loss = out[:3]
                t1 = time.time()

                if verbosity >= 3:
                    print("step {}, mse_loss {:g}".format( step, loss))

                if tensorboard==1:
                    summaries = out[3]
                    train_summary_writer.add_summary(summaries, step)

                return t1-t0

            def val_step(step, epoch):
                mse_collector = 0
                n_samples = 0

                min_pred = 1e9
                max_pred = -1e9
                for batch_id, batch in enumerate(data_iter.validation_iter(batchsize=batchsize)):
                    X, label, _weights = batch

                    feed_dict = model.make_input_feeddict(X, label, None)
                    feed_dict[is_training_mode_placeholder] = False


                    fetches = [model.batch_mse, model.predictions]
                    if additional_samples_val_summaries_op is not None and batch_id==3:
                        # for one arbitrary batch execute some extra summaries
                        fetches += [additional_samples_val_summaries_op]

                    out= sess.run(fetches, feed_dict)

                    if additional_samples_val_summaries_op is not None and batch_id==3:
                        sums = out[2]
                        val_summary_writer.add_summary(sums, step)


                    mse,preds = out[:2]
                    min_pred = min(min_pred, np.min(preds))
                    max_pred = max(max_pred, np.max(preds))


                    mse_collector += mse * len(label)
                    n_samples += len(label)

                root_mse = np.sqrt(float(mse_collector)/float(n_samples))
                if tensorboard==1:
                   val_sum=sess.run(val_summary_op,{mse_val:root_mse})
                   summaries =val_sum
                   val_summary_writer.add_summary(summaries,step)

                if verbosity >=1:
                    print("Validation Epoch {}: step {}, root_mse {:g}".format(
                        epoch, step, root_mse))
                if verbosity >=2:
                    print("min-pred: {}, max-pred: {}".format(min_pred, max_pred))

                global min_validation_rmse
                if root_mse < min_validation_rmse:
                    min_validation_rmse = root_mse
                    if verbosity >= 1:
                        print("New Best Epoch with {} at Epoch {}".format(root_mse, epoch))

                    if submission_data is not None and root_mse<1.98:
                        print("start submission")
                        submission = pd.read_csv("data/sampleSubmission.csv")
                        predictions = prediction_step(submission_data)
                        submission["Prediction"] = predictions
                        submission.to_csv(os.path.join(exp_path, "submission.csv"), index=False) # "+str(epoch)+"_s"+str(round(root_mse*10000)/10000)+"
                        print("Finished submission, saved in ", exp_path)

                    if checkpoint:
                        saver.save(sess, checkpoint_path, global_step=step)

                return root_mse

            def prediction_step(pred_data):
                preds_collector = list()

                for batch in data_iter.prediction_iter(pred_data, batchsize=batchsize):
                    X, label, weights = batch

                    feed_dict = model.make_input_feeddict(X, None, None)
                    feed_dict[is_training_mode_placeholder] = False

                    fetches = [model.predictions]

                    (preds,) = sess.run(fetches, feed_dict)
                    preds_collector.extend(preds)

                return preds_collector

            time_total_collector = 0
            time_tf_collector = 0

            all_val_scores = np.zeros(n_epochs+1)
            all_val_scores[0] = val_step(step=0, epoch=0) # to make a submission or compute the predicitons without training

            for epoch_id, epoch in enumerate(data_iter.training_epoch_iter(batchsize=batchsize, shuffle=True, n_epochs=n_epochs)):
                t0 = time.time()
                time_tf = 0
                for batch in epoch:  # epoch itself is a generator for batches
                    input_data, label, weighting = batch

                    t = train_step(input_data, label=label, weightings=weighting)
                    current_step = tf.train.global_step(sess, global_step)
                    time_tf += t

                t1 = time.time()
                time_total_collector += (t1-t0)
                time_tf_collector += time_tf
                if verbosity >= 2:
                    print("Average Total Time per Epoch {}, Average tensorflow time per epoch {}".format(time_total_collector/(epoch_id+1), time_tf_collector/(epoch_id+1)))


                all_val_scores[epoch_id+1] = val_step(step=current_step, epoch=epoch_id+1)

    print("Best Score {}".format(min_validation_rmse))
    return all_val_scores

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batchsize", default=512, type=int)
    parser.add_argument("--n_epochs", default=5, type=int)


    parser.add_argument("--embed_size", default=128, type=int)
    parser.add_argument("--n_context_movies", default=64, type=int)
    parser.add_argument("--n_context_movies_pred", default=128, type=int)

    parser.add_argument("--reg_constant", default=0.0, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)

    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument("--embedding_version", default="", choices=["simple", "attention", "autoencoder_embedding", "attention_rating", "svd", "svdAE"])
    embedding_group.add_argument("--embedding_config", default="", type=str)

    prediction_group = parser.add_mutually_exclusive_group()
    prediction_group.add_argument("--prediction_version", default="", choices=["inner_prod", "mlp1", "mlp2", "pass_through"])
    prediction_group.add_argument("--prediction_config", default="", type=str)

    parser.add_argument("--sample_weighting", default="none", choices=["none", "user", "rating", "debug"])
    parser.add_argument("--optimizer", default="adam", choices=["adam", "rms"])


    parser.add_argument("--verbosity", default=1, type=int)
    parser.add_argument("--tensorboard", default=0,type=int)
    parser.add_argument("--no_date", action="store_true")
    parser.add_argument("--make_submissions", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")

    parser.add_argument("--load_embedding_weights", default="", type=str)
    parser.add_argument("--load_prediction_weights", default="", type=str)
    parser.add_argument("--load_embedding_weights_numpy", default="", type=str)

    parser.add_argument("--out_dir", default="runs", type=str)
    parser.add_argument("--experiment_name", default="test", type=str)

    parser.add_argument("--force_gpu", action="store_true")

    parser.add_argument("--crossvalidate", action="store_true")

    args = parser.parse_args()

    print(vars(args))

    build_and_train(args)
