import unittest
from PhaseplateNetwork.TFModules.OpticalLayers.PaddingLayer import PaddingLayer
from PhaseplateNetwork.TFModules.OpticalLayers.CropLayer import CropLayer
import tensorflow as tf
import numpy as np

class TestPaddingCroppingLayers(unittest.TestCase):


    def setUp(self):
        self.padding_layer1 = PaddingLayer(100, 0.05, 0.05, 0.0)
        self.padding_layer2 = PaddingLayer((100,90), 0.05, 0.01, 0.0)
        self.padding_layer3 = PaddingLayer((20,10),(0.01,0.011), (0.01,0.01), 1.0)
        self.padding_layer4 = PaddingLayer(360, 0.05+2*0.05, 0.05)
        self.padding_layer5 = PaddingLayer(224, 0.001792, 0.0001)
        #print(0.05+2*0.05)


        self.crop1 = CropLayer(300, 0.05, 0.05)
        self.crop2 = CropLayer((140,126), 0.05, 0.01)
        self.crop3 = CropLayer((60,28), (0.01,0.011), (0.01,0.01))
        self.crop4 = CropLayer((600,600), 0.05+2*0.05,0.05)
        self.crop5 = CropLayer((250,250), 0.001792, 0.0001)

        self.input1 = tf.ones((10,100,100,1))
        self.input2 = tf.ones((10, 100,90, 2))
        self.input3 = tf.ones((2,20,10,1))
        self.input4 = tf.ones((10,360,360,1))
        self.input5 = tf.ones((10,224,224,1))


    def test_padding_1d(self):
        out1 = self.padding_layer1(self.input1)
        self.assertEqual(out1.shape, tf.TensorShape([10,300,300,1]),"padding layer does not pad correctly")

    def test_padding_big(self):
        out4 = self.padding_layer4(self.input4)
        self.assertEqual(out4.shape, tf.TensorShape([10,600,600,1]),"big padding layer does not pad correctly")

    def test_padding_small_dimension(self):
        out5 = self.padding_layer5(self.input5)
        self.assertEqual(out5.shape, tf.TensorShape([10,250,250,1]), "small dimension padding layer does not pad correctly")

    def test_padding_2d(self):
        out2 = self.padding_layer2(self.input2)
        out3 = self.padding_layer3(self.input3)

        self.assertEqual(out2.shape, tf.TensorShape([10, 140, 126, 2]), "padding layer with 2d input shapes does not pad correctly")
        self.assertEqual(out3.shape, tf.TensorShape([2, 60, 28,1]),"padding layer with 2d input shapes does not pad correctly")

    def test_getoutputshape_padding1d(self):
        out_shape = self.padding_layer1.compute_output_shape(self.input1.shape)
        self.assertEqual(out_shape, tf.TensorShape([10,300,300,1]))
        out_shape3 = self.padding_layer4.compute_output_shape(tf.TensorShape([1, 360, 360, 1]))
        self.assertEqual(out_shape3, tf.TensorShape([1,600,600,1]))

    def test_getoutputshape_padding2d(self):
        out_shape2 = self.padding_layer2.compute_output_shape(self.input2.shape)
        out_shape3 = self.padding_layer3.compute_output_shape(self.input3.shape)
        self.assertEqual(out_shape2, tf.TensorShape([10,140,126,2]),"compute_output_shape failed to give correct values")
        self.assertEqual(out_shape3, tf.TensorShape([2,60,28,1]), "compute_output_shape failed to give correct values")

    def test_cropping_1d(self):
        out1 = self.padding_layer1(self.input1)
        self.assertEqual(out1.shape, tf.TensorShape([10,300,300,1]),"padding layer does not pad correctly")
        out_c1 = self.crop1(out1)
        self.assertEqual(out_c1.shape, tf.TensorShape([10,100,100,1]),"Cropping layer does not crop correctly")

    def test_cropping_big(self):
        out4 = self.padding_layer4(self.input4)
        out_c4 = self.crop4(out4)
        self.assertEqual(out_c4.shape, tf.TensorShape([10,360,360,1]))

    def test_cropping_small_dimension(self):
        out5 = self.padding_layer5(self.input5)
        self.assertEqual(out5.shape, tf.TensorShape([10,250,250,1]), "small dimension padding layer does not pad correctly")
        out_c5 = self.crop5(out5)
        self.assertEqual(out_c5.shape, tf.TensorShape([10,224,224,1]))

    def test_cropping_2d(self):
        out2 = self.padding_layer2(self.input2)
        out3 = self.padding_layer3(self.input3)
        out_c2 = self.crop2(out2)
        out_c3 = self.crop3(out3)
        self.assertEqual(out_c2.shape, tf.TensorShape([10,100,90,2]),"Cropping layer does not crop correctly")
        self.assertEqual(out_c3.shape, tf.TensorShape([2,20,10,1]),"Cropping layer with 2d inputs does not crop correctly")

    def test_getoutputshape_cropping1d(self):
        out_shape = self.crop1.compute_output_shape(tf.TensorShape([10,300,300,1]))
        self.assertEqual(out_shape, tf.TensorShape([10,100,100,1]))
    def test_getoutputshape_cropping2d(self):
        out_shape2 = self.crop2.compute_output_shape(tf.TensorShape([10,140,126,2]))
        out_shape3 = self.crop3.compute_output_shape(tf.TensorShape([2,60,28,1]))

        self.assertEqual(out_shape2, tf.TensorShape([10,100,90,2]))
        self.assertEqual(out_shape3, tf.TensorShape([2,20,10,1]))


    def test_random_padding_and_cropping(self):
        inp = tf.random.uniform([2,20,10,1],0, 255.0)
        out = self.crop3(self.padding_layer3(inp))
        self.assertTrue(np.array_equal(inp.numpy(),out.numpy()),"Padding and cropping does not yield the same output")


if __name__ == '__main__':
    unittest.main()


