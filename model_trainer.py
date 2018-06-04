
import sys
import argparse
import functools
import tensorflow as tf

import model_zoo as zoo
from loss import CrossEntropyLoss
from optimizer import OptimizerBuilder
from data_provider import DataProvider
# import tf_data_zoo

from io import StringIO
import matplotlib.pyplot as plt
import numpy as np


class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)


    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    Learning TensorFlow, pp 212.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class ModelTrainer(object):

    def __init__(self,
                 model,
                 data,
                 loss,
                 optimizer,
                 log_dir):

        self.model = model

        self.data = data

        self.loss = loss.add_loss_to_graph(self.model)

        self.optimizer = optimizer.add_optimizer_to_graph(self.loss)

        self.error

        # Merge the summary.
        # tf.summary.merge_all()

    @define_scope
    def error(self):

        mistakes = tf.not_equal(tf.argmax(self.model.target_placeholder, 1),
                                tf.argmax(self.model.inference, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        tf.summary.scalar('error', error)
        return(error)

    # @define_scope
    def train_epoch(self):

        print("------------training_epoch-------------")

        # Initialize the training dataset.
        self.data.initialize_training_iterator(self.sess)

        # Run one training epoch.
        while True:

            try:

                images, labels = self.data.get_next_training_element(self.sess)

                # Make a dict to load the batch onto the placeholders.
                feed_dict = {self.stimulus_placeholder: images,
                             self.target_placeholder: labels,
                             self.keep_prob: FLAGS.keep_prob}

                # Compute error over the training set.
                # train_error = sess.run(model_trainer.error,
                #                        feed_dict=train_dict)

                # Compute loss over the training set.
                train_loss = self.sess.run(self.loss, feed_dict=feed_dict)

                # print("train loss = %f" % train_loss)

                self.sess.run(self.optimizer, feed_dict=feed_dict)

            except tf.errors.OutOfRangeError:

                break

        # print("Training Epoch " + str(i) + " complete.")
        print(train_loss)

        # Initialize the validation dataset.
        self.data.initialize_validation_iterator(self.sess)

        # Run one validation epoch.
        while True:

            try:

                images, labels = self.data.get_next_validation_element(
                    self.sess)

                # Make a dict to load the batch onto the placeholders.
                feed_dict = {self.model.stimulus_placeholder: images,
                             self.model.target_placeholder: labels,
                             self.model.keep_prob: FLAGS.keep_prob}

                # Compute error over the training set.
                # train_error = sess.run(model_trainer.error,
                #                        feed_dict=train_dict)

                # Compute loss over the training set.
                validation_loss = self.sess.run(self.loss, feed_dict=feed_dict)

                # print("validation_loss = %f" % validation_loss)

            except tf.errors.OutOfRangeError:

                break

        # print("Validation Epoch " + str(i) + " complete.")
        print(validation_loss)

        print("----------------------------------------")


def example_usage(_):

    # Clear the log directory, if it exists.
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Reset the default graph.
    tf.reset_default_graph()

    # tf.train.create_global_step()

    print("------------model_output-------------")

    # Instantiate the model zoo and retrieve a model.
    model_zoo = zoo.TensorFlowModelZoo()

    model = model_zoo.get_model('lenet')

    # Get a data provider.
    training_filenames = FLAGS.data_dir + "/" + FLAGS.train_file
    validation_filenames = FLAGS.data_dir + "/" + FLAGS.validation_file
    batch_size = FLAGS.train_batch_size

    data_provider = DataProvider(training_filenames=training_filenames,
                                 validation_filenames=validation_filenames,
                                 batch_size=batch_size)

    logger = Logger(FLAGS.log_dir)

    # data_zoo = tf_data_zoo.TensorFlowDataZoo()

    # training_data = data_zoo.get_dataset('mnist',
    #                                      partition='training',
    #                                      data_path=training_filenames)

    # validation_data = data_zoo.get_dataset('mnist',
    #                                        partition='validation',
    #                                        data_path=validation_filenames)

    loss = CrossEntropyLoss()

    # loss_zoo = tf_loss_zoo.TensorFlowLossZoo()

    # loss = loss_zoo.get_loss("cross_entropy_loss")

    optimizer = OptimizerBuilder(FLAGS.learning_rate)

    # optimizer_zoo = tf_optimizer_zoo.TensorFlowOptimizerZoo()

    # optimizer = optimizer_zoo.get_optimizer("adam")

    # Build a model trainer.
    model_trainer = ModelTrainer(model=model,
                                 data=data_provider,
                                 optimizer=optimizer,
                                 loss=loss,
                                 log_dir=FLAGS.log_dir)

    print("-------------------------------------")

    # model_trainer.train_epoch()

    # Merge the summary.
    tf.summary.merge_all()

    # validation_loss = tf.scalar("validation_loss")
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_summaries_secs=20.0)

    with sv.managed_session() as sess:

        # train_writer = tf.summary.FileWriter(FLAGS.log_dir +
        #                                      '/train', sess.graph)

        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        print("------------training_output-------------")

        # Iterate until the epoch limit has been reached.
        for i in range(FLAGS.num_epochs):

            train_loss = 0
            validation_loss = 0

            # Initialize the training dataset.
            data_provider.initialize_training_iterator(sess)

            # Run one training epoch.
            while True:

                try:

                    images, labels = data_provider.get_next_training_element(
                        sess)

                    # Make a dict to load the batch onto the placeholders.
                    feed_dict = {model_trainer.model.stimulus_placeholder: images,
                                 model_trainer.model.target_placeholder: labels,
                                 model_trainer.model.keep_prob: FLAGS.keep_prob}

                    # Compute error over the training set.
                    # train_error = sess.run(model_trainer.error,
                    #                        feed_dict=train_dict)

                    # Compute loss over the training set.
                    train_loss += sess.run(model_trainer.loss,
                                           feed_dict=feed_dict)

                    # print("train loss = %f" % train_loss)

                    sess.run(model_trainer.optimizer,
                             feed_dict=feed_dict)

                except tf.errors.OutOfRangeError:

                    break

            print("Training Epoch " + str(i) + " complete.")
            print(train_loss)

            logger.log_scalar('training_loss', train_loss, i)

            # Initialize the validation dataset.
            data_provider.initialize_validation_iterator(sess)

            # Run one validation epoch.
            while True:

                try:

                    images, labels = data_provider.get_next_validation_element(
                        sess)

                    # Make a dict to load the batch onto the placeholders.
                    feed_dict = {model_trainer.model.stimulus_placeholder: images,
                                 model_trainer.model.target_placeholder: labels,
                                 model_trainer.model.keep_prob: FLAGS.keep_prob}

                    # Compute error over the training set.
                    # train_error = sess.run(model_trainer.error,
                    #                        feed_dict=train_dict)

                    # Compute loss over the training set.
                    validation_loss += sess.run(model_trainer.loss,
                                                feed_dict=feed_dict)

                    # print("validation_loss = %f" % validation_loss)

                except tf.errors.OutOfRangeError:

                    break

            print("Validation Epoch " + str(i) + " complete.")
            print(validation_loss)

            logger.log_scalar('validation_loss', validation_loss, i)

        print("----------------------------------------")

        # Close the summary writers.
        # test_writer.close()
        # train_writer.close()
        sv.stop()
        sess.close()

    return()


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.

    # These flags are often, but not always, overwritten by the launcher.
    parser.add_argument('--log_dir', type=str,
                        default='../log/tensorflow_experiment/templog',
                        help='Summaries log directory.')

    parser.add_argument('--pause_time', type=float,
                        default=0.0,
                        help='Number of seconds to pause before execution.')

    parser.add_argument('--log_filename', type=str,
                        default='deep_sa_generalization_experiment.csv',
                        help='Summaries log directory.')

    parser.add_argument('--keep_prob', type=float,
                        default=1.0,
                        help='Keep probability for output layer dropout.')

    parser.add_argument('--train_batch_size', type=int,
                        default=128,
                        help='Training set batch size.')

    parser.add_argument('--batch_interval', type=int,
                        default=1,
                        help='Interval between training batch refresh.')

    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to run trainer.')

    parser.add_argument('--test_interval', type=int, default=10,
                        help='Number of steps between test set evaluations.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')

    # These flags specify the data used in the experiment.
    parser.add_argument('--data_dir', type=str,
                        default='C:/Users/Justin Fletcher/Research/data/mnist',
                        help='Directory from which to pull data TFRecords.')

    parser.add_argument('--train_file', type=str,
                        default='train.tfrecords',
                        help='Training dataset filename.')

    parser.add_argument('--validation_file', type=str,
                        default='validation.tfrecords',
                        help='Validation dataset filename.')

    parser.add_argument('--input_size', type=int,
                        default=28 * 28,
                        help='Dimensionality of the input space.')

    parser.add_argument('--label_size', type=int,
                        default=10,
                        help='Dimensinoality of the output space.')

    parser.add_argument('--val_batch_size', type=int,
                        default=10000,
                        help='Validation set batch size.')

    # These flags specify placekeeping variables.
    parser.add_argument('--rep_num', type=int,
                        default=0,
                        help='Flag identifying the repitition number.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # # Run the main function as TF app.
    tf.app.run(main=example_usage, argv=[sys.argv[0]] + unparsed)
