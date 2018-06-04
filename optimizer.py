import tensorflow as tf


class OptimizerBuilder(object):

    def __init__(self, learning_rate):

        self.learning_rate = learning_rate

    def add_optimizer_to_graph(self, loss):

        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return(opt)
