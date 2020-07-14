"""Decoder unittest"""

import unittest
import torch
from transformer.encoder import EncoderBlock
from transformer.attention import DotProductAttention
from transformer.decoder import DecoderBlock


class TestDecoder(unittest.TestCase):
    def setUp(self):

        dropout = 0.5
        attention = DotProductAttention(dropout)
        encoder = EncoderBlock(embed_dim=24, num_hiddens=24,
                                    seq_len=100, ffn_num_hiddens=48,
                                    num_heads=8, attention=attention,
                                    dropout=dropout, use_bias=False)
        valid_len = torch.tensor([2, 3])
        self.decoder = DecoderBlock(embed_dim=24, num_hiddens=24,
                                    seq_len=100, ffn_num_hiddens=48,
                                     num_heads=8, dropout=dropout, i=0)
        self.inputs = torch.ones([2, 100, 24])
        self.state = [encoder(self.inputs, valid_len), valid_len, [None]]

    def test_decoder(self):
        output = self.decoder(self.inputs, self.state)
        self.assertEqual(output[0].shape, torch.Size([2, 100, 24]))