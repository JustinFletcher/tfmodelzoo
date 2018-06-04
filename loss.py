
import tensorflow as tf


class Loss(object):

    def add_loss_to_graph(self, model):

        raise NotImplementedError


class CrossEntropyLoss(Loss):

    def add_loss_to_graph(self, model):

        # Get a loss.
        # Compute the cross entropy.
        xe = tf.nn.softmax_cross_entropy_with_logits(
            labels=model.target_placeholder,
            logits=model.inference,
            name='xentropy')
        # Take the mean of the cross entropy.
        loss = tf.reduce_mean(xe, name='xentropy_mean')

        return(loss)
