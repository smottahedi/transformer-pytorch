"""Transformer decoder unittest"""

import unittest
from transformer.decoder import TransformerDecoder
from transformer.encoder import TransformerEncoder
import torch


class TestTransformerDecoder(unittest.TestCase):
    def setUp(self):
        self.encoder = TransformerEncoder(200, 24, 24, 48, 8, 2, 0.5)
        self.inputs = torch.ones(2, 100)
        self.valid_len = torch.Tensor([2, 3])


        self.decoder = TransformerDecoder(200, 24, 24, 48, 8, 2, 0.5)
        self.enc_output = self.encoder(self.inputs, self.valid_len)
    
    def test_transformer_decoder(self):
        dec_state = self.decoder.init_state(self.enc_output, self.valid_len)
        output = self.decoder(self.inputs, dec_state)
        self.assertEqual(output[0].shape, torch.Size([2, 100, 200]))


