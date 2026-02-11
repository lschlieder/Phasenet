import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
import numpy as np


def get_image_fg(im):
    if im.ndim <= 2:
        t1 = threshold_otsu(im)
        return im > t1
    elif im.ndim == 3:
        t1 = np.array([threshold_otsu(_im) for _im in im])
        return im > t1[:, None, None]
    else:
        raise ValueError("im must be either 2D or 3D ndarray")


def SNR(im, fg, bg):
    """ SNR from An & Psaltis 1995
      assume the image is comprised of foreground and background pixels
      and calculate an SNR based on the contrast of the fg vs background and their internal variances

      fg,bg : binary masks of foreground and background pixels in an image
    """

    m1 = im[fg].mean()
    m2 = im[bg].mean()

    s1 = im[fg].std()
    s2 = im[bg].std()

    return (m1 - m2) / np.sqrt(s1 ** 2 + s2 ** 2)


def calc_SNR(ims, refs):
    assert refs.shape == ims.shape, "images and reference images must have the same shape"

    fgs = get_image_fg(refs)
    bgs = 1 - fgs

    if ims.ndim == 3:
        N = ims.shape[0]
        return np.array([SNR(ims[j], fgs[j], bgs[j]) for j in range(N)])
    elif ims.ndim == 2:
        return SNR(ims, fgs, bgs)
    else:
        raise ValueError("ims must be 2D or 3D")

class SNR_Metric(tf.keras.metrics.Metric):



    def __init__(self, **kwargs):
        '''
        Creates the metric.
        '''
        super(SNR_Metric, self).__init__(**kwargs)
        #self.class_images = class_images
        self.running_snr = 0.0
        self.running_n = 0
        #self.acc_metric = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight = None):
        wanted_images = y_true
        output_images = y_pred
        if output_images.dtype == tf.complex64:
            output_images = tf.math.abs(output_images)
        if wanted_images.dtype == tf.complex64:
            wanted_images = tf.math.abs(wanted_images)
        if not wanted_images.shape[0] == None:
            self.running_snr =  self.running_snr + tf.reduce_sum(tf.numpy_function(calc_SNR, (output_images[:,:,:,0],wanted_images[:,:,:,0]), tf.float64) )
            self.running_n = self.running_n + wanted_images.shape[0]

    def result(self):
        if self.running_n == 0:
            ret =  0
        else:
            ret = self.running_snr/self.running_n
        return ret

    def reset_states(self):
        self.running_snr= 0.0
        self.running_n = 0