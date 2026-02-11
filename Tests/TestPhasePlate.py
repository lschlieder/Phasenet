import unittest
from PhaseplateNetwork.TFModules.OpticalLayers.PhasePlate import PhasePlate
import tensorflow as tf
import numpy as np

class TestPhasePlate(unittest.TestCase):
    def setUp(self):
        self.PhasePlate = PhasePlate( (90,90), (60,60))