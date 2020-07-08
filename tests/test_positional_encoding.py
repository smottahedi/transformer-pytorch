"""Test positional encoding"""

import unittest
import torch
from transformer.positional_encoding import PositionalEncoding # pylint: disable=import-error


class TestPositionalEncoding(unittest.TestCase):
    """unit test for positional encoding"""

    def setUp(self):
        self.pos_enc = PositionalEncoding(20, 0)
        self.input = torch.zeros(1, 100, 20) # pylint: disable=no-member

    def test_positional_encoder(self):
        output = self.pos_enc(self.input)
        self.assertEqual(output.shape, torch.Size((1, 100, 20))) # pylint: disable=no-member