import numpy as np
import unittest

from context import mospy


class TestVth(unittest.TestCase):
    def setUp(self):
        print("TestVth Start")

    def test_vthon(self):
        def sigmoid(z):
            return 1/(1+np.exp(-z))
        Vg = np.arange(-5, 5, 0.1)
        # TODO! change sigmoid to Ltspice mosfet Id-Vg
        Id = sigmoid(Vg)
        von = mospy.vth.vthon(Vg=Vg, Id=Id)
        expected = -2.0142987926242593
        self.assertEqual(expected, von)


if __name__ == "__main__":
    unittest.main()
