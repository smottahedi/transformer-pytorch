"""DotProductAttention unittest"""

import unittest
from transformer.attention import DotProductAttention
import torch 


class TestAttention(unittest.TestCase):
    
    def setUp(self):
        self.attention = DotProductAttention(dropout=0.5)
        self.attention.eval()
        self.keys = torch.ones(2,10,2)
        self.values = torch.arange(40, dtype=torch.float32).reshape(1,10,4).repeat(2,1,1)

    def test_attention(self):
        self.assertEqual(self.attention(torch.ones(2, 1, 2), self.keys, self.values,
            torch.tensor([2, 6])).shape, torch.Size([2, 1, 4]))


if __name__ == '__main__':
    unittest.main()
