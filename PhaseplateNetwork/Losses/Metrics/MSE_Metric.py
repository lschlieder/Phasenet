import tensorflow as tf
from PhaseplateNetwork.Losses.Losses import mse

class MSE_Metric(tf.keras.metrics.Metric):



    def __init__(self, **kwargs):
        '''
        Creates the metric.
        '''
        super(MSE_Metric, self).__init__(**kwargs)
        #self.class_images = class_images
        #self.running_mse = tf.Variable(0.0)
        #self.running_n = tf.Variable(0.0)
        self.running_mse = self.add_weight('running_mse', initializer='zeros', dtype=tf.float32)
        self.running_n = self.add_weight('running_n', initializer='zeros', dtype=tf.float32)

        #self.acc_metric = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight = None):
        wanted_images = y_true
        output_images = y_pred
        if output_images.dtype == tf.complex64:
            output_images = tf.math.abs(output_images)
        if wanted_images.dtype == tf.complex64:
            wanted_images = tf.math.abs(wanted_images)

        #mse_loss_per_image_standardized
        #print(self.running_mse)
        #if not wanted_images.shape[0] == None:
            #self.running_snr =  self.running_snr + tf.reduce_sum(tf.numpy_function(calc_SNR, (output_images[:,:,:,0],wanted_images[:,:,:,0]), tf.float64) )
            #self.running_n = self.running_n + wanted_images.shape[0]
        values = mse(output_images, wanted_images)
        self.running_mse.assign_add( values)
            #print(self.running_mse)
        self.running_n.assign_add(tf.cast(tf.size(values),dtype = tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.running_mse, self.running_n)

    def reset_state(self):
        self.running_mse.assign(0.0)
        self.running_n.assign(0.0)