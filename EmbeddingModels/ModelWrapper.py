import tensorflow as tf

class ModelWrapper:

    def __init__(self, embedding, prediction, weighted_samples=False):
        self.embedding_model = embedding
        self.prediction_model = prediction

        self.batch_mse = self.prediction_model.batch_mse

        self.predictions = self.prediction_model.predictions

        self.weighted_samples = weighted_samples

        if self.weighted_samples:
            print(">>>> With Sample Weighting")
            self.sample_weights_placeholder = tf.placeholder(tf.float32, [None], name="sample_weights")

    @property
    def loss(self):
        if self.weighted_samples:
            weighted_per_sample_loss = self.prediction_model.loss * self.sample_weights_placeholder
        else:
            weighted_per_sample_loss = self.prediction_model.loss

        return tf.reduce_mean(weighted_per_sample_loss)

    def make_input_feeddict(self, X, label, weightings=None):
        d = self.embedding_model.make_input_feeddict(X)
        if label is not None:
            d[self.prediction_model.label_placeholder] = label

        if weightings is None:
            pass
        elif weightings is not None and self.weighted_samples:
            d[self.sample_weights_placeholder] = weightings
        else:
            raise RuntimeError("")

        return d

    def get_additional_val_summaries(self):
        all_sums = self.embedding_model.get_additional_val_summaries() + self.prediction_model.get_additional_val_summaries()

        if len(all_sums) != 0:
            return tf.summary.merge(all_sums)