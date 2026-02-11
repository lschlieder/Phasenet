import tensorflow as tf

class MinMaxConstraint(tf.keras.constraints.Constraint):
  """Constrains weight tensors to be between 0 and 1"""

  def __init__(self, min = 0.0, max = 1.0):
    self.min = min
    self.max = max

  def __call__(self, w):
    if w.dtype == tf.complex64:
        abs = tf.minimum(tf.maximum(tf.math.abs(w), self.min),self.max)
        angle = tf.math.angle(w)
        return tf.complex(abs,0.0) * tf.math.exp(1j*tf.complex(angle, 0.0))
    else: 
        return tf.minimum(tf.maximum(w, self.min),self.max)