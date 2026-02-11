import tensorflow as tf
from PhaseplateNetwork.Losses.Losses import get_classification_vector

import matplotlib.pyplot as plt

class SoftmaxCrossentropyFromImages(tf.keras.metrics.Metric):
    def __init__(self, class_images, **kwargs):
        '''
        Creates the metric.
        :param class_images: Images of the shape [num, X,Y,C], that are used as masks to be multiplied with the incoming images to get the classification vectors
        '''
        super(SoftmaxCrossentropyFromImages, self).__init__(**kwargs)
        self.class_images = class_images
        self.xent_metric = tf.keras.metrics.CategoricalCrossentropy(from_logits=False)

    def update_state(self, y_true, y_pred, sample_weight = None):
        output_images = y_pred
        wanted_images = y_true
        if output_images.dtype == tf.complex64:
            output_images = tf.math.abs(output_images)
        if wanted_images.dtype == tf.complex64:
            wanted_images = tf.math.abs(wanted_images)

        class_vec = tf.math.softmax(get_classification_vector(output_images, self.class_images), axis = 1)
        wanted_output_vec = tf.math.softmax(get_classification_vector(wanted_images, self.class_images), axis = 1)
        #print(wanted_output_vec[0])
        #print(class_vec[0])
        #print(wanted_output_vec[0])

        self.xent_metric.update_state(wanted_output_vec, class_vec)

    def result(self):
        return self.xent_metric.result()

    def reset_states(self):
        self.xent_metric.reset_states()