import unittest
import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL

class TestPaddingCroppingLayers(unittest.TestCase):
    def setUp(self):
        self.ReluLayer1 = OL.ReluLayer(bias = 0.1)
        self.ReluLayer2 = OL.ReluLayer(bias = 0.0)

        self.input1 = 0.2 * tf.math.exp(1j*np.random.uniform(0,np.pi*2))
        self.input2 = tf.ones((32,30,30,1), dtype = tf.complex64)* tf.math.exp( 1j* tf.random.uniform((32,30,30,1), 0, np.pi*2))
        #self.input3 =

    def test_relulayer1(self):
        out1 = self.ReluLayer1(self.input1)
        self.assertEqual(tf.dtype(out1), tf.complex64)
        self.assertAlmostEqual(tf.math.abs(out1), 0.1)

        out2 = self.ReluLayer1(self.input2)
        self.assertAlmostEqual(tf.math.abs(out2), tf.ones((32,30,30,1), dtype = tf.float32)*0.9)

    def test_relulayer2(self):
        out1 = self.ReluLayer2(self.input1)
        self.assertEqual(tf.dtype(out1), tf.complex64)
        self.assertAlmostEqual(out1, self.input2)

        out2 = self.ReluLayer2(self.input2)
        self.assertAlmostEqual(out2, self,input2)




if __name__ == '__main__':
    unittest.main()