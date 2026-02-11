import tensorflow as tf

class ForwardOperator(tf.keras.Model):
    '''
    Parent definition for any ForwardOperator. Implements the identity forward operator.
    '''
    def __init__(self):
        super(ForwardOperator, self).__init__()

    def call(self, input):
        return input

    def inverse_call(self,input):
        return input