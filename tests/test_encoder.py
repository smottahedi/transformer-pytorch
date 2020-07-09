"""Encoder unittest"""

import unittest
import torch
from transformer.encoder import EncoderBlock
from transformer.attention import DotProductAttention



class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.ones([2, 100, 24]) # pylint: disable=no-member
        self.valid_len = torch.tensor([2, 3]) # pylint: disable=not-callable
        dropout = 0.5
        attention = DotProductAttention(dropout)
        self.encoder = EncoderBlock(embed_dim=24, num_hiddens=24, 
                                    seq_len=100, ffn_num_hiddens=48,
                                    num_heads=8, attention=attention,
                                    dropout=dropout, use_bias=False)

    def test_encoder(self):
        output = self.encoder(self.inputs, self.valid_len)
        self.assertEqual(output.shape, torch.Size((2, 100, 24)))

if __name__ == '__main__':
    unittest.main()