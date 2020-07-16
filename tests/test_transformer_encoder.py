"""Transformer encoder unittest"""

import unittest 
import torch
from transformer.encoder import TransformerEncoder


class TransformerEncoderTest(unittest.TestCase):
    def setUp(self):
        self.encoder = TransformerEncoder(200, 24, 24, 48, 8, 2, 0.5)
        self.inputs = torch.ones(2, 100)
        self.valid_len = torch.Tensor([2, 3])

    def test_transformer_encoder(self):
        output = self.encoder(self.inputs, self.valid_len)
        self.assertEqual(output.shape, torch.Size([2, 100, 24]))