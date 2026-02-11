import tensorflow as tf

class GradNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self,alpha = 1.0, **kwargs):
        super(GradNormalizationLayer, self).__init__(**kwargs)
        self.alpha = alpha



    def grad_loss(self, hologram):
        grad = tf.image.image_gradients(tf.math.angle(hologram))
        grad_loss = tf.math.reduce_mean(tf.math.reduce_sum((grad[0]) ** 2, axis=[1, 2, 3]), axis=0)
        grad_loss += tf.math.reduce_mean(tf.math.reduce_sum((grad[1]) ** 2, axis=[1, 2, 3]), axis=0)
        grad_loss = self.alpha * grad_loss
        return grad_loss

    def call(self, Input):
        self.add_loss(self.grad_loss(Input))
        return Input