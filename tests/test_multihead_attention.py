"""Unit test multihead attention."""

import unittest
import torch
from transformer.multihead_attention import MultiheadAttention # pylint: disable=import-error
from transformer.attention import DotProductAttention # pylint: disable=import-error

class TestMultiheadAttention(unittest.TestCase):
    """Test multihead-attention."""

    def setUp(self):
        """setup test input"""
        attention = DotProductAttention(0.5)
        self.cell = MultiheadAttention(100, 90, 9, attention=attention)
        self.input = torch.ones([2, 4, 100]) # pylint: disable=no-member
        self.valid_len = torch.tensor([2, 3]) # pylint: disable=not-callable


    def test_multihead_attention(self):
        """test multihead_attention"""

        self.assertEqual(self.cell(self.input, self.input, self.input,
                                   self.valid_len).shape, torch.Size([2, 4, 90])) # pylint: disable=no-member



if __name__ == '__main__':
    unittest.main()
