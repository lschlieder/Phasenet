import tensorflow as tf
from PhaseplateNetwork.Losses.Losses import get_classification_vector

import matplotlib.pyplot as plt

class AccuracyFromImages(tf.keras.metrics.Metric):
    def __init__(self, class_images,**kwargs):
        '''
        Creates the metric.
        :param class_images: Images of the shape [num, X,Y,C], that are used as masks to be multiplied with the incoming images to get the classification vectors
        '''
        super(AccuracyFromImages, self).__init__(**kwargs)
        self. class_images = class_images
        self.acc_metric = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight = None):
        wanted_images = y_true
        output_images = y_pred
        if output_images.dtype == tf.complex64:
            output_images = tf.math.abs(output_images)
        if wanted_images.dtype == tf.complex64:
            wanted_images = tf.math.abs(wanted_images)
        class_vec = tf.math.argmax(get_classification_vector(output_images, self.class_images), axis = 1)
        wanted_output_vec = tf.math.argmax(get_classification_vector(wanted_images, self.class_images), axis = 1)
        '''
        print(class_vec.numpy())
        print(wanted_output_vec.numpy())
        plt.imshow(wanted_images[0,:,:,0])
        plt.figure()
        plt.imshow(self.class_images[0,:,:,0])
        plt.show()
        input()
        '''
        #class_vec = tf.random.uniform([10],0, 3, dtype = tf.int32)

        self.acc_metric.update_state( wanted_output_vec, class_vec)

    def result(self):
        return self.acc_metric.result()

    def reset_state(self):
        self.acc_metric.reset_states()