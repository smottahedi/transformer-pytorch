"""Unittest for pointwiseffn"""

import unittest
import torch
from transformer.positionwise_ffn import PositionWiseFFN # pylint: disable=import-error


class TestPointwiseFFN(unittest.TestCase):
    def setUp(self):
        self.ffn = PositionWiseFFN(4, 4, 8)
        self.input = torch.ones(2, 3, 4) # pylint: disable=no-member
    
    def test_positionwise(self):
        self.assertEqual(self.ffn(self.input).shape, torch.Size((2, 3, 8))) # pylint: disable=no-member