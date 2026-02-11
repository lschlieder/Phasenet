import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

def sk_ssim(pred, wanted, window_size = 5, data_range = 1):
    return ssim(pred,wanted, window_size = window_size, data_range = 1)

class SKImage_SSIM(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        '''
        Creates the metric.
        '''
        super(SKImage_SSIM, self).__init__(**kwargs)
        #self.class_images = class_images
        self.running_ssim = 0.0
        self.running_n = 0
        #self.acc_metric = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight = None):
        wanted_images = y_true
        output_images = y_pred
        if output_images.dtype == tf.complex64:
            output_images = tf.math.abs(output_images)
        if wanted_images.dtype == tf.complex64:
            wanted_images = tf.math.abs(wanted_images)
        temp_ssim = 0.0
        #print(output_images.dtype)
        #print(wanted_images.dtype)
        #print(wanted_images.shape)
        if not wanted_images.shape[0] == None:
            for i in range(0, wanted_images.shape[0]):
                temp_ssim = temp_ssim + tf.numpy_function(sk_ssim, (output_images[i,:,:,0], wanted_images[i,:,:,0]), tf.float64)
            self.running_ssim = self.running_ssim + temp_ssim
            self.running_n = self.running_n + wanted_images.shape[0]

    def result(self):
        if self.running_n == 0:
            ret =  0
        else:
            ret= self.running_ssim/self.running_n
        return ret

    def reset_states(self):
        self.running_ssim = 0.0
        self.running_n = 0