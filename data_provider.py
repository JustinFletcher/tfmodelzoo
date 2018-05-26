
import tensorflow as tf


def _read_and_decode_mnist(serialized_example):

    # Parse that example into features.
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # resize the image tensors to add channels, 1 in this case
    # required to pass the images to various layers upcoming in the graph
    image = tf.reshape(image, [28, 28, 1])

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label_batch = features['label']

    label = tf.one_hot(label_batch,
                       10,
                       on_value=1.0,
                       off_value=0.0)

    return image, label


class DataProvider(object):

    def __init__(self, training_filenames, validation_filenames, batch_size):

        # First, we construct the training Dataset and Iterator objects.
        training_dataset = tf.data.TFRecordDataset(training_filenames)
        training_dataset = training_dataset.map(_read_and_decode_mnist)
        training_dataset = training_dataset.batch(batch_size)
        training_dataset = training_dataset.shuffle(100)
        training_iterator = training_dataset.make_one_shot_iterator()

        self._training_iterator = training_iterator
        self._training_init_op = self._training_iterator.make_initializer(training_dataset)

        # Then do the same for the validation set.
        validation_dataset = tf.data.TFRecordDataset(validation_filenames)
        validation_dataset = validation_dataset.map(_read_and_decode_mnist)
        validation_dataset = validation_dataset.shuffle(100)
        validation_dataset = validation_dataset.batch(batch_size)
        validation_iterator = validation_dataset.make_one_shot_iterator()

        self._validation_iterator = validation_iterator
        self._validation_init_op = self._validation_iterator.make_initializer(validation_dataset)

        # A feedable iterator is defined by a handle placeholder and its
        # structure. We could use the `output_types` and `output_shapes`
        # properties of either training_dataset` or `validation_dataset` here,
        # because they have identical structure.
        self._handle_placeholder = tf.placeholder(tf.string, shape=[])
        output_types = training_dataset.output_types
        output_shapes = training_dataset.output_shapes

        # Create an Iterator that will get the next element from the Iterator
        # for which the string_handle property matches the string in the handle
        # placeholder.
        iterator = tf.data.Iterator.from_string_handle(
            self._handle_placeholder, output_types, output_shapes)

        # Assign the instantiated iterator to a private property.
        self._next_element = iterator.get_next()
        self._training_handle = []
        self._validation_handle = []

    def initialize_iterator(self, sess, partition="training"):

        # Select the partition handle based on the input partition.
        if partition == "training":

            sess.run(self._training_init_op)

        elif partition == "validation":

            sess.run(self._validation_init_op)

        else:

            print(partition + " is not a recognized partition.")
            raise NotImplementedError

    def initialize_training_iterator(self, sess):

        return(self.initialize_iterator(sess=sess, partition="training"))

    def initialize_validation_iterator(self, sess):

        return(self.initialize_iterator(sess=sess, partition="validation"))

    def get_next_element(self, sess, partition="training"):

        # If we have not cached the string_handles for known paritions, do so.
        if not self._training_handle:

            self._training_handle = sess.run(
                self._training_iterator.string_handle())

            self._validation_handle = sess.run(
                self._validation_iterator.string_handle())

        # Select the partition handle based on the input partition.
        if partition == "training":

            partition_handle = self._training_handle

        elif partition == "validation":

            partition_handle = self._validation_handle

        else:

            print(partition + " is not a recognized partition.")
            raise NotImplementedError

        # Return the next element of the Iterator corresponging to the selected
        # partition.
        feed_dict = {self._handle_placeholder: partition_handle}
        element = sess.run(self._next_element, feed_dict=feed_dict)
        return(element)

    def get_next_training_element(self, sess):

        return(self.get_next_element(sess=sess, partition="training"))

    def get_next_validation_element(self, sess):

        return(self.get_next_element(sess=sess, partition="validation"))
