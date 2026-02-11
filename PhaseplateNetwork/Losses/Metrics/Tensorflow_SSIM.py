import tensorflow as tf
from skimage.metrics import structural_similarity as ssim


class Tensorflow_SSIM(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        '''
        Creates the metric.
        '''
        super(Tensorflow_SSIM, self).__init__(**kwargs)
        #self.class_images = class_images
        #self.running_ssim = 0.0
        #self.running_n = 0
        self.running_ssim = self.add_weight('running_ssim', initializer='zeros', dtype=tf.float32)
        self.running_n = self.add_weight('running_n', initializer='zeros', dtype=tf.float32)
        #self.acc_metric = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight = None):
        output_images = y_pred
        wanted_images = y_true
        if output_images.dtype == tf.complex64:
            output_images = tf.math.abs(output_images)
        if wanted_images.dtype == tf.complex64:
            wanted_images = tf.math.abs(wanted_images)

        values = tf.reduce_mean(tf.image.ssim(output_images,wanted_images, max_val = tf.math.reduce_max(wanted_images)))
        #print(tf.size(values))
        self.running_ssim.assign_add( values)
            #print(self.running_mse)
        self.running_n.assign_add(tf.cast(tf.size(values),dtype = tf.float32))
        #if not wanted_images.shape[0] == None:
        #    self.running_ssim = self.running_ssim + tf.reduce_sum(tf.image.ssim(output_images,wanted_images, max_val = tf.math.reduce_max(wanted_images)))
        #    self.running_n = self.running_n + wanted_images.shape[0]


    def result(self):
        return tf.math.divide_no_nan(self.running_ssim, self.running_n)

    def reset_states(self):
        self.running_ssim.assign(0.0)
        self.running_n.assign(0.0)