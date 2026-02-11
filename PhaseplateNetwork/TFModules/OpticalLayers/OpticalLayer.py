import tensorflow as tf

class OpticalLayer(tf.keras.layers.Layer):
    def __init__(self,output_level = 1, **kwargs):
        super(OpticalLayer,self).__init__(**kwargs)
        self.output_level = output_level

    def get_image_variables(self):
        return

    def call_for_fields(self, input, num = 1):
        return [self.call(input)]
